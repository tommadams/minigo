# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a reinforcement learning loop to train a Go playing model."""

import sys
sys.path.insert(0, '.')  # nopep8

import asyncio
import logging
import numpy as np
import os
import random
import re
import shutil
import subprocess
import tensorflow as tf
import traceback
import time
import dual_net  # Needed for some FLAGS
from ml_perf.utils import *
from ml_perf.mlp_log import mlperf_print

from absl import app, flags
from rl_loop import example_buffer, fsdb
from tensorflow import gfile

N = int(os.environ.get('BOARD_SIZE', 19))

flags.DEFINE_string('checkpoint_dir', 'ml_perf/checkpoint/{}'.format(N),
                    'The checkpoint directory specify a start model and a set '
                    'of golden chunks used to start training.  If not '
                    'specified, will start from scratch.')

flags.DEFINE_string('target_path', 'ml_perf/target/{}/target.pb'.format(N),
                    'Path to the target model to beat.')

flags.DEFINE_integer('iterations', 100, 'Number of iterations of the RL loop.')

flags.DEFINE_float('gating_win_rate', 0.55,
                   'Win-rate against the current best required to promote a '
                   'model to new best.')

flags.DEFINE_string('flags_dir', None,
                    'Directory in which to find the flag files for each stage '
                    'of the RL loop. The directory must contain the following '
                    'files: bootstrap.flags, selfplay.flags, eval.flags, '
                    'train.flags.')

flags.DEFINE_integer('window_size', 10,
                     'Maximum number of recent selfplay rounds to train on.')

flags.DEFINE_list('tpu_names', None, 'List of TPU names.')

# Eval & selfplay flags.
flags.DEFINE_integer('selfplay_tpu_inference_threads', 16,
                     'Number of inference threads to use for selfplay.')
flags.DEFINE_integer('eval_tpu_inference_threads', 2,
                     'Number of inference threads to use for eval.')
flags.DEFINE_integer('selfplay_num_games', 4096, '')
flags.DEFINE_integer('eval_num_games', 100, '')
flags.DEFINE_float('disable_resign_pct', 0.1,
                   'Fraction of selfplay games to disable resignation for.')

flags.DEFINE_bool('verbose', False,
                  'If true, log all subprocess output to stderr in addition '
                  'to the logfiles.')


FLAGS = flags.FLAGS


# Lock used to serialize writing to log files.
run_lock = asyncio.Lock()


class State:
  """State data used in each iteration of the RL loop.

  Models are named with the current reinforcement learning loop iteration number
  and the model generation (how many models have passed gating). For example, a
  model named "000015-000007" was trained on the 15th iteration of the loop and
  is the 7th models that passed gating.
  Note that we rely on the iteration number being the first part of the model
  name so that the training chunks sort correctly.
  """

  def __init__(self):
    self.start_time = time.time()

    self.iter_num = 0
    self.gen_num = 0
    self._seed = 0

    self.best_model_name = None

    # Number of examples in each training chunk.
    self.num_examples = {}

  @property
  def output_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num)

  @property
  def train_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num + 1)

  @property
  def tmp_model_name(self):
    return '%06d-tmp' % self.iter_num

  @property
  def best_model_path(self):
    if self.best_model_name is None:
      return None
    else:
      return '{}.pb'.format(
         os.path.join(fsdb.models_dir(), self.best_model_name))

  @property
  def train_model_path(self):
    return '{}.pb'.format(
         os.path.join(fsdb.models_dir(), self.train_model_name))

  @property
  def tmp_model_path(self):
    return '{}.pb'.format(
         os.path.join(fsdb.models_dir(), self.tmp_model_name))

  def get_next_seed(self):
    self._seed += 1
    return self._seed


class ColorWinStats:
  """Win-rate stats for a single model & color."""

  def __init__(self, total, both_passed, opponent_resigned, move_limit_reached):
    self.total = total
    self.both_passed = both_passed
    self.opponent_resigned = opponent_resigned
    self.move_limit_reached = move_limit_reached
    # Verify that the total is correct
    assert total == both_passed + opponent_resigned + move_limit_reached


class WinStats:
  """Win-rate stats for a single model."""

  def __init__(self, line):
    pattern = '\s*(\S+)' + '\s+(\d+)' * 8
    match = re.search(pattern, line)
    if match is None:
      raise ValueError('Can\t parse line "{}"'.format(line))
    self.model_name = match.group(1)
    raw_stats = [float(x) for x in match.groups()[1:]]
    self.black_wins = ColorWinStats(*raw_stats[:4])
    self.white_wins = ColorWinStats(*raw_stats[4:])
    self.total_wins = self.black_wins.total + self.white_wins.total


def load_train_times():
  models = []
  path = os.path.join(fsdb.models_dir(), 'train_times.txt')
  with gfile.Open(path, 'r') as f:
    for line in f.readlines():
      line = line.strip()
      if line:
        timestamp, name = line.split(' ')
        path = os.path.join(fsdb.models_dir(), name + '.pb')
        models.append((float(timestamp), name, path))
  return models


def eval_models():
  target = os.path.join(fsdb.models_dir(), 'target.pb')
  models = load_train_times()
  for i, (timestamp, name, path) in enumerate(models):
    win_rate = wait(evaluate_model(path, target, i + 1))
    logging.info('i:%d ts:%s n:%s wr:%f', i, timestamp, name, win_rate)
    if win_rate >= 0.50:
      mlperf_print('eval_result', None, metadata={'iteration': i + 1,
                                                  'timestamp': timestamp})
      return (i, win_rate, timestamp, name, path)
  mlperf_print('eval_result', None, metadata={'iteration': len(models),
                                              'timestamp': 0})
  return None


def initialize_from_checkpoint(state):
  """Initialize the reinforcement learning loop from a checkpoint."""

  # The checkpoint's work_dir should contain the most recently trained model.
  model_paths = tf.gfile.Glob(os.path.join(FLAGS.checkpoint_dir,
                                           'work_dir/model.ckpt-*.pb'))
  if len(model_paths) != 1:
    raise RuntimeError('Expected exactly one model in the checkpoint work_dir, '
                       'got [{}]'.format(', '.join(model_paths)))
  start_model_path = model_paths[0]

  # Copy the latest trained model into the models directory and use it on the
  # first round of selfplay.
  print('Copying checkpoint')
  state.best_model_name = 'checkpoint'
  gfile.Copy(start_model_path,
             os.path.join(fsdb.models_dir(), state.best_model_name + '.pb'))

  # Copy the golden chunks.
  print('Copying golden chunks')
  TF_RECORD_CONFIG = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.ZLIB)
  golden_chunks_dir = os.path.join(FLAGS.checkpoint_dir, 'golden_chunks')
  with tf.Session():
    for basename in os.listdir(golden_chunks_dir):
      src_path = os.path.join(golden_chunks_dir, basename)
      dst_path = os.path.join(fsdb.golden_chunk_dir(), basename)
      gfile.Copy(src_path, dst_path)
      records = tf.python_io.tf_record_iterator(src_path, TF_RECORD_CONFIG)
      state.num_examples[basename] = sum(1 for _ in records)

  # Copy the training files.
  print('Copying training files')
  work_dir = os.path.join(FLAGS.checkpoint_dir, 'work_dir')
  for basename in os.listdir(work_dir):
    src_path = os.path.join(work_dir, basename)
    dst_path = os.path.join(fsdb.working_dir(), basename)
    gfile.Copy(src_path, dst_path)

  print('Checkpoint initialized')

def parse_win_stats_table(stats_str, num_lines):
  result = []
  lines = stats_str.split('\n')
  while True:
    # Find the start of the win stats table.
    assert len(lines) > 1
    if 'Black' in lines[0] and 'White' in lines[0] and 'm.lmt.' in lines[1]:
        break
    lines = lines[1:]

  # Parse the expected number of lines from the table.
  for line in lines[2:2 + num_lines]:
    result.append(WinStats(line))

  return result


async def append_to_log(log_path, cmd_str, output_str):
  async with run_lock:
    with gfile.Open(log_path, 'a') as f:
      f.write(cmd_str)
      f.write('\n')
      f.write(output_str)
      f.write('\n')


async def run(*cmd):
  """Run the given subprocess command in a coroutine.

  Args:
    *cmd: the command to run and its arguments.

  Returns:
    The output that the command wrote to stdout as a list of strings, one line
    per element (stderr output is piped to stdout).

  Raises:
    RuntimeError: if the command returns a non-zero result.
  """

  log_path = os.path.join(FLAGS.base_dir, get_cmd_name(cmd) + '.log')
  cmd_str = expand_cmd_str(cmd)

  try:
    stdout = await checked_run(*cmd, verbose=FLAGS.verbose)
    await append_to_log(log_path, cmd_str, stdout)
  except Exception as e:
    error_str = 'ERROR: {}'.format(traceback.format_exc())
    await append_to_log(log_path, cmd_str, error_str)
    raise

  # Split stdout into lines.
  return stdout.split('\n')


def get_golden_chunk_records():
  """Return up to num_records of golden chunks to train on.

  Returns:
    A list of golden chunks up to num_records in length, sorted by path.
  """

  pattern = os.path.join(fsdb.golden_chunk_dir(), '*.zz')
  return sorted(tf.gfile.Glob(pattern), reverse=True)[:FLAGS.window_size]


# Self-play a number of games.
async def selfplay(state, flagfile='selfplay'):
  """Run selfplay and write a training chunk to the fsdb golden_chunk_dir.

  Args:
    state: the RL loop State instance.
    flagfile: the name of the flagfile to use for selfplay, either 'selfplay'
        (the default) or 'boostrap'.
  """

  output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
  holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)

  tpu_names = FLAGS.tpu_names
     
  num_tpus = len(tpu_names)
  awaitables = []
  for i, tpu_name in enumerate(tpu_names):
    model_path = state.best_model_path
    if model_path:
        model_path = 'tpu:{}:{},{}'.format(
            FLAGS.selfplay_tpu_inference_threads, tpu_name, model_path)
    else:
       model_path = 'random:0,0.4:0.4'

    # Play all calibration (aka resign disabled) games on one TPU with each thread
    # playing one game.
    num_calibration_games = round(
        FLAGS.selfplay_num_games * FLAGS.disable_resign_pct)
    num_normal_games = FLAGS.selfplay_num_games - num_calibration_games
    if i == 0:
      num_games = num_calibration_games
      parallel_games = num_games
      disable_resign_pct=1
    else:
      # Calculate the number of games this TPU needs to play, handling the case
      # where num_normal_games isn't exactly divisble by num_tpus.
      start = ((i - 1) * num_normal_games) // (num_tpus - 1)
      end = (i * num_normal_games) // (num_tpus - 1)
      num_games = end - start
      parallel_games = (num_games + 1) // 2
      disable_resign_pct=0

    awaitables.append(run(
        'bazel-bin/cc/selfplay',
        '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
        '--model={}'.format(model_path),
        '--output_dir={}/{}'.format(output_dir, i),
        '--holdout_dir={}/{}'.format(holdout_dir, i),
        '--disable_resign_pct={}'.format(disable_resign_pct),
        '--num_games={}'.format(num_games),
        '--parallel_games={}'.format(parallel_games),
        '--seed={}'.format(state.get_next_seed())))

  for lines in await asyncio.gather(*awaitables):
    result = '\n'.join(lines[-5:])
    logging.info(result)
    stats = parse_win_stats_table(result, 1)[0]
    num_games = stats.total_wins
    logging.info('Black won %0.3f, white won %0.3f',
                 stats.black_wins.total / num_games,
                 stats.white_wins.total / num_games)

  # Write examples to a single record.
  pattern = os.path.join(output_dir, '*/*', '*.zz')
  seed = state.get_next_seed()
  random.seed(seed)
  tf.set_random_seed(seed)
  np.random.seed(seed)
  # TODO(tommadams): This method of generating one golden chunk per generation
  # is sub-optimal because each chunk gets reused multiple times for training,
  # introducing bias. Instead, a fresh dataset should be uniformly sampled out
  # of *all* games in the training window before the start of each training run.
  buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)

  # TODO(tommadams): parallel_fill is currently non-deterministic. Make it not
  # so.
  logging.info('Writing golden chunk from "{}"'.format(pattern))
  num_examples = buffer.parallel_fill(tf.gfile.Glob(pattern))
  basename = state.output_model_name + '.tfrecord.zz'
  buffer.flush(os.path.join(fsdb.golden_chunk_dir(), basename))
  state.num_examples[basename] = num_examples


async def train(state, tf_records, export_name):
  """Run training and write a new model to the fsdb models_dir.

  Args:
    state: the RL loop State instance.
    tf_records: a list of paths to TensorFlow records to train on.
  """

  num_examples = 0
  for record in tf_records:
    num_examples += state.num_examples[os.path.basename(record)]
  model_path = os.path.join(fsdb.models_dir(), export_name)
  lr_boundaries = ['--lr_boundaries={}'.format(x) for x in FLAGS.lr_boundaries]
  lr_rates = ['--lr_rates={}'.format(x) for x in FLAGS.lr_rates]
  await run(
      'python3', 'train.py', *tf_records, *lr_boundaries, *lr_rates,
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
      '--work_dir={}'.format(fsdb.working_dir()),
      '--train_batch_size={}'.format(FLAGS.train_batch_size),
      '--export_path={}'.format(model_path),
      '--num_examples={}'.format(num_examples),
      '--training_seed={}'.format(state.get_next_seed()),
      '--tpu_name={}'.format(FLAGS.tpu_names[0]),
      '--freeze=true')
  elapsed = time.time() - state.start_time
  return elapsed


def append_timestamp(elapsed, model_name):
  # Append the time elapsed from when the RL was started to when this model
  # was trained. GCS files are immutable, so we have to do the append manually.
  timestamps_path = os.path.join(fsdb.models_dir(), 'train_times.txt')
  try:
    with gfile.Open(timestamps_path, 'r') as f:
      timestamps = f.read()
  except tf.errors.NotFoundError:
    timestamps = ''
  timestamps += '{:.3f} {}\n'.format(elapsed, model_name)
  with gfile.Open(timestamps_path, 'w') as f:
    f.write(timestamps)


async def evaluate_model(eval_model_path, target_model_path, seed):
  """Evaluate one model against a target.
  Args:
    eval_model_path: the path to the model to evaluate.
    target_model_path: the path to the model to compare to. If None, a random
                       model is used.
    seed: random seed to use when running eval.

  Returns:
    The win-rate of eval_model against target_model in the range [0, 1].
  """

  tpu_name = FLAGS.tpu_names[1]
  target_model_path = 'tpu:{}:{},{}'.format(
      FLAGS.eval_tpu_inference_threads, tpu_name, target_model_path)

  eval_model_path = 'tpu:{}:{},{}'.format(
      FLAGS.eval_tpu_inference_threads, tpu_name, eval_model_path)

  lines = await run(
      'bazel-bin/cc/eval',
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'eval.flags')),
      '--model={}'.format(eval_model_path),
      '--model_two={}'.format(target_model_path),
      '--parallel_games={}'.format(FLAGS.eval_num_games),
      '--seed={}'.format(seed))
  result = '\n'.join(lines[-7:])
  logging.info(result)
  eval_stats, target_stats = parse_win_stats_table(result, 2)
  num_games = eval_stats.total_wins + target_stats.total_wins
  win_rate = eval_stats.total_wins / num_games
  logging.info('Win rate %s vs %s: %.3f', eval_stats.model_name,
               target_stats.model_name, win_rate)
  return win_rate


def rl_loop():
  """The main reinforcement learning (RL) loop."""

  state = State()

  if FLAGS.checkpoint_dir:
    # Start from a partially trained model.
    initialize_from_checkpoint(state)
  else:
    # Play the first round of selfplay games with a fake model that returns
    # random noise. We do this instead of playing multiple games using a single
    # model bootstrapped with random noise to avoid any initial bias.
    wait(selfplay(state, 'bootstrap'))

    # Train a real model from the random selfplay games.
    tf_records = get_golden_chunk_records()
    state.iter_num += 1
    wait(train(state, tf_records))

    # Select the newly trained model as the best.
    state.best_model_name = state.train_model_name
    state.gen_num += 1

    # Run selfplay using the new model.
    wait(selfplay(state))

  mlperf_print('init_stop', None)
  mlperf_print('run_start', None)
  state.start_time = time.time()
  # Now start the full training loop.
  while state.iter_num <= FLAGS.iterations:
    mlperf_print('epoch_start', None)
    # Train on shuffled game data from recent selfplay rounds.
    tf_records = get_golden_chunk_records()
    state.iter_num += 1

    if state.iter_num == 1:
      elapsed = wait(train(state, tf_records, state.train_model_name))
      append_timestamp(elapsed, state.train_model_name)
      prev_train_model_name = state.train_model_name
      prev_train_model_path = state.train_model_path
    else:
      elasped, model_win_rate = wait([
          train(state, tf_records, state.tmp_model_name),
          evaluate_model(prev_train_model_path, state.best_model_path,
                         state.get_next_seed())])
      if model_win_rate >= FLAGS.gating_win_rate:
        state.best_model_name = prev_train_model_name
        state.gen_num += 1

      prev_train_model_name = state.train_model_name
      prev_train_model_path = state.train_model_path

      # TODO(tommadams): Move all three files in parallel
      logging.info('Renaming %s to %s', state.tmp_model_name, state.train_model_name)
      for ext in ['.data-00000-of-00001', '.index', '.meta', '.pb']:
        gfile.Rename(os.path.join(fsdb.models_dir(), state.tmp_model_name + ext),
                     os.path.join(fsdb.models_dir(), state.train_model_name + ext))
      append_timestamp(elapsed, state.train_model_name)

    mlperf_print('save_model', None, metadata={'iteration': state.iter_num})

    wait(selfplay(state))
    mlperf_print('epoch_stop', None)


def main(unused_argv):
  """Run the reinforcement learning loop."""

  print('Wiping dir %s' % FLAGS.base_dir, flush=True)
  try:
    gfile.DeleteRecursively(FLAGS.base_dir)
  except:
    pass
  try:
    gfile.DeleteRecursively(os.path.join('gs://' + FLAGS.bucket_name, FLAGS.base_dir))
  except:
    pass

  dirs = [fsdb.models_dir(), fsdb.selfplay_dir(), fsdb.holdout_dir(),
          fsdb.eval_dir(), fsdb.golden_chunk_dir(), fsdb.working_dir()]
  for d in dirs:
    ensure_dir_exists(d);

  logging.getLogger().addHandler(
      logging.FileHandler(os.path.join(FLAGS.base_dir, 'rl_loop.log')))
  formatter = logging.Formatter('')
  for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

  mlperf_print('submission_benchmark', 'minigo')
  mlperf_print('submission_division', 'closed')
  mlperf_print('submission_org', 'google')
  mlperf_print('submission_platform', 'tpu-4x-v3-8')
  mlperf_print('submission_status', 'cloud')
  mlperf_print('cache_clear', None)
  mlperf_print('global_batch_size', FLAGS.num_tpu_cores * FLAGS.train_batch_size)
  mlperf_print('opt_base_learning_rate', FLAGS.lr_rates[0])
  mlperf_print('opt_learning_rate_decay_boundary_steps', FLAGS.lr_boundaries)

  mlperf_print('init_start', None)

  # Copy the flag files so there's no chance of them getting accidentally
  # overwritten while the RL loop is running.
  print('Copying flags')
  flags_dir = os.path.join(FLAGS.base_dir, 'flags')
  shutil.copytree(FLAGS.flags_dir, flags_dir)
  FLAGS.flags_dir = flags_dir

  # Copy the target model to the models directory so we can find it easily.
  print('Copying target model')
  tf.gfile.Copy(FLAGS.target_path, os.path.join(fsdb.models_dir(), 'target.pb'))

  with logged_timer('Total time'):
    try:
      rl_loop()
      result = eval_models()
      if result:
        mlperf_print('run_stop', None, metadata={'status': 'success'})
      else:
        mlperf_print('run_stop', None, metadata={'status': 'aborted'})
    finally:
      asyncio.get_event_loop().close()


if __name__ == '__main__':
  app.run(main)

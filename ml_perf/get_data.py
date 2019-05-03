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

"""Copies & compiles the data required to start the reinforcement loop."""

import sys
sys.path.insert(0, '.')  # nopep8

import asyncio
import glob
import os

from absl import app, flags
from ml_perf import utils

N = os.environ.get('BOARD_SIZE', '19')

flags.DEFINE_string('src_dir', 'gs://minigo-pub/ml_perf/',
                    'Directory on GCS to copy source data from. Files will be '
                    'copied from subdirectories of src_dir corresponding to '
                    'the BOARD_SIZE environment variable (defaults to 19).')

flags.DEFINE_string('dst_dir', 'ml_perf/',
                    'Desitination directory to write to. Files will be written '
                    'to subdirectories of dst_dir corresponding to the '
                    'BOARD_SIZE environment variable (defaults to 19).')

flags.DEFINE_boolean('use_tpu', False,
                     'Set to true to generate models that can run on Cloud TPU')

flags.DEFINE_string('gcs_tmp_dir', None,
                    'If use_tpu true, A temporary GCS directory to copy models '
                    'to for freezing.')

flags.DEFINE_string('tpu_name', '',
                    'If use_tpu is true, TPU name used for freezing.')

FLAGS = flags.FLAGS


def freeze_graph(src_path):
  if FLAGS.use_tpu:
    # Freezing for TPU: copy source files to GCS.
    utils.wait(utils.checked_run(
        'gsutil', '-m', 'cp', src_path + '.*', FLAGS.gcs_tmp_dir))
    freeze_path = os.path.join(FLAGS.gcs_tmp_dir, os.path.basename(src_path))
  else:
    freeze_path = src_path

  utils.wait(utils.checked_run(
      'python', 'freeze_graph.py',
      '--model_path={}'.format(freeze_path),
      '--use_tpu={}'.format(FLAGS.use_tpu),
      '--tpu_name={}'.format(FLAGS.tpu_name)))

  if FLAGS.use_tpu:
    # Freezing for TPU: copy frozen model back from GCS.
    utils.wait(utils.checked_run(
        'gsutil', 'cp', freeze_path + '.pb', os.path.dirname(src_path)))


def main(unused_argv):
  try:
    for d in ['checkpoint', 'target']:
      # Pull the required training checkpoints and models from GCS.
      src = os.path.join(FLAGS.src_dir, d, N)
      dst = os.path.join(FLAGS.dst_dir, d)
      utils.ensure_dir_exists(dst)
      utils.wait(utils.checked_run('gsutil', '-m', 'cp', '-r', src, dst))

    # Freeze the target model.
    freeze_graph(os.path.join(FLAGS.dst_dir, 'target', N, 'target'))

    # Freeze the training checkpoint models.
    pattern = os.path.join(FLAGS.dst_dir, 'checkpoint', N, 'work_dir', '*.index')
    for path in glob.glob(pattern):
      freeze_graph(os.path.splitext(path)[0])

  finally:
    asyncio.get_event_loop().close()


if __name__ == '__main__':
  app.run(main)


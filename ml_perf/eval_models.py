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

"""Evaluates a directory of models against the target model."""

import sys
sys.path.insert(0, '.')  # nopep8

from tensorflow import gfile
import os

from absl import app, flags
from reference_implementation import evaluate_model, load_train_time, eval_models, wait
from rl_loop import fsdb

FLAGS = flags.FLAGS


def main(unused_argv):
  result = eval_models()
  if result:
    iteration, win_rate, timestamp, name, path = result
    print('Model {} beat target after {}s'.format(name, timestamp))
  else:
    print('No model beat the target')


if __name__ == '__main__':
  app.run(main)

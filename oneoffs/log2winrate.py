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
"""Converts the winrates from a directories of eval log to a csv.
"""

from absl import app, flags
import glob
import os
import re
import matplotlib.pyplot as plt

flags.DEFINE_string('dst', None, 'Destination path.')

FLAGS = flags.FLAGS


class Result(object):
    def __init__(self, path):
        self.generations = []
        self.winning_gen = None

        target_name = None
        with open(path, 'r') as f:
            winning_gen = None
            for line in f.readlines():
                m = re.search(r'Win rate (\d+) vs (.*): (\d+\.\d+)', line)
                if m:
                    gen, target, winrate = m.groups()
                    winrate = float(winrate)
                    if winrate >= 0.5 and self.winning_gen is None:
                        self.winning_gen = gen

                    self.generations.append((int(gen), winrate))
                    if target_name is None:
                        target_name = target
                    else:
                        assert target_name == target

        model_name = os.path.splitext(os.path.basename(path))[0]
        self.name = '%s-vs-%s' % (model_name, target_name)


def main(argv):
    assert FLAGS.dst is not None

    paths = []
    for d in argv[1:]:
        pattern = os.path.join(d, '*.log')
        match = glob.glob(pattern)
        paths += match

    results = []
    for path in paths:
        results.append(Result(path))

    for result in results:
        if result.winning_gen is None:
            print('### %s : DIDN\'T BEAT TARGET ###' % result.name)

    for result in results:
        if result.winning_gen is not None:
            print('%s : %s beat target' % (result.name, result.winning_gen))
            x, y = zip(*result.generations)
            plt.plot(x, y, '-', label=result.name)

    plt.yticks((0, 0.5, 1))
    plt.grid(axis='y')
    plt.ylim(0, 1)
    #plt.legend()
    plt.xlabel('generation')
    plt.ylabel('winrate')

    plt.savefig(FLAGS.dst)


if __name__ == '__main__':
    app.run(main)

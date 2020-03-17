// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>

#include "absl/strings/str_format.h"
#include "cc/init.h"
#include "cc/model/loader.h"
#include "cc/model/loader.h"
#include "cc/model/model.h"
#include "cc/model/types.h"
#include "cc/position.h"
#include "gflags/gflags.h"

DEFINE_string(device, "", "Device to run on (e.g. TPU address).");
DEFINE_string(model, "", "Path to a minigo model.");

namespace minigo {
namespace {

void Run() {
  auto model = NewModel(FLAGS_model, FLAGS_device);

  Position position(Color::kBlack);
  ModelInput input;
  input.sym = symmetry::kIdentity;
  input.position_history.push_back(&position);

  ModelOutput output;
  std::vector<const ModelInput*> inputs{&input};
  std::vector<ModelOutput*> outputs{&output};
  model->RunMany(inputs, &outputs, nullptr);

  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      std::cout << absl::StreamFormat("  %.2f", output.policy[Coord(row, col)]);
    }
    std::cout << "\n";
  }
  std::cout << absl::StreamFormat("  %.2f\n", output.policy[Coord::kPass]);
}

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::Run();
  return 0;
}

// Copyright 2019 Google LLC
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

#include <algorithm>
#include <memory>
#include <thread>
#include <vector>

#include "absl/memory/memory.h"
#include "cc/dual_net/factory.h"
#include "cc/logging.h"
#include "gflags/gflags.h"

DEFINE_string(model, "", "");
DEFINE_int32(num_threads, 16, "Number of threads to run in parallel");
DEFINE_int32(batch_size, 128, "Batch size");

namespace minigo {

void MultithreadTest() {
  const auto desc = ParseModelDescriptor(FLAGS_model);

  auto factory = NewDualNetFactory(desc.engine);
  auto model_uptr = factory->NewDualNet(desc.model);
  auto* model = model_uptr.get();

  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id < FLAGS_num_threads; ++thread_id) {
    threads.push_back(std::thread([model, thread_id]() {
      std::vector<DualNet::BoardFeatures> features(FLAGS_batch_size);
      std::vector<DualNet::Output> outputs(FLAGS_batch_size);
      std::vector<const DualNet::BoardFeatures*> feature_ptrs;
      std::vector<DualNet::Output*> output_ptrs;
      for (const auto& f : features) {
        feature_ptrs.push_back(&f);
      }
      for (auto& o : outputs) {
        output_ptrs.push_back(&o);
      }

      for (;;) {
        MG_LOG(INFO) << "Thread " << thread_id << " still running";
        model->RunMany(feature_ptrs, output_ptrs, nullptr);
      }
    }));
  }

  for (auto& t : threads) {
    t.join();
  }
  MG_LOG(INFO) << "DONE";
}

} // namespace minigo

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  minigo::MultithreadTest();

  return 0;
}

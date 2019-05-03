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

#include <array>
#include <algorithm>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gflags/gflags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/session.h"

DEFINE_string(model, "", "");
DEFINE_string(tpu_name, "", "");
DEFINE_int32(num_threads, 16, "");
DEFINE_int32(parallel_games, 2000, "");
DEFINE_int32(virtual_losses, 8, "");
DEFINE_bool(bool_features, false, "");
DEFINE_int32(iterations, 100, "");
DEFINE_string(compression_algorithm, "deflate", "");
DEFINE_int32(compression_level, 1, "");

using tensorflow::DT_FLOAT;
using tensorflow::DT_BOOL;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace minigo {

static constexpr int kN = 9;
static constexpr int kNumMoves = kN * kN + 1;
static constexpr int kMoveHistory = 8;
static constexpr int kNumStoneFeatures = kMoveHistory * 2 + 1;
static constexpr int kNumBoardFeatures = kN * kN * kNumStoneFeatures;

using BoardFeatures = std::array<float, kNumBoardFeatures>;

struct Output {
  std::array<float, kNumMoves> policy;
  float value;
};

constexpr auto kTpuOpsGraphDef = R"(
node {
  name: "ConfigureDistributedTPU"
  op: "ConfigureDistributedTPU"
  device: "/device:TPU_SYSTEM:0"
  attr {
    key: "embedding_config"
    value {
      s: ""
    }
  }
  attr {
    key: "is_global_init"
    value {
      b: false
    }
  }
  attr {
    key: "tpu_embedding_config"
    value {
      s: ""
    }
  }
}
node {
  name: "ShutdownDistributedTPU"
  op: "ShutdownDistributedTPU"
  device: "/device:TPU_SYSTEM:0"
}
library {
}
)";

std::unique_ptr<Session> CreateSession(const GraphDef& graph_def,
                                       const std::string& tpu_name) {
  SessionOptions options;
  options.target = tpu_name;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);
  // options.config.mutable_rpc_options()->set_compression_algorithm(
  //     FLAGS_compression_algorithm);
  // options.config.mutable_rpc_options()->set_compression_level(
  //     FLAGS_compression_level);
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));
  return session;
}

struct Thread {
  Thread(int thread_id, const GraphDef& graph_def, const std::string& tpu_name,
         int num_replicas)
      : thread_id(thread_id) {
    auto batch_size =
        FLAGS_parallel_games * FLAGS_virtual_losses / FLAGS_num_threads;
    auto data_type = FLAGS_bool_features ? DT_BOOL : DT_FLOAT;

    session = CreateSession(graph_def, tpu_name);
    for (int i = 0; i < num_replicas; ++i) {
      output_names.push_back(absl::StrCat("policy_output_", i));
      output_names.push_back(absl::StrCat("value_output_", i));
      inputs.emplace_back(
          absl::StrCat("pos_tensor_", i),
          Tensor(data_type, TensorShape({batch_size, kN, kN,
                                         kNumStoneFeatures})));
    }
    // Warm up inference.
    TF_CHECK_OK(session->Run(inputs, output_names, {}, &outputs));
  }

  void Run() {
    thread = std::thread([this]() {
      for (int i = 0; i < FLAGS_iterations; ++i) {
        if (thread_id == 0) {
	  std::cout << (i + 1) << " / " << FLAGS_iterations << std::endl;
	}
        TF_CHECK_OK(session->Run(inputs, output_names, {}, &outputs));
      }
    });
  }

  void Join() {
    thread.join();
  }

  std::thread thread;
  int thread_id;
  std::unique_ptr<Session> session;
  std::vector<std::pair<std::string, Tensor>> inputs;
  std::vector<std::string> output_names;
  std::vector<Tensor> outputs;
};

void MultithreadTest() {
  GraphDef graph_def;
  ::tensorflow::protobuf::TextFormat::ParseFromString(
      kTpuOpsGraphDef, &graph_def);
  auto main_session = CreateSession(graph_def, FLAGS_tpu_name);
  auto batch_size =
      FLAGS_parallel_games * FLAGS_virtual_losses / FLAGS_num_threads;
  auto total_inferences = batch_size * FLAGS_iterations * FLAGS_num_threads;

  std::cout << "Initializing TPU " << FLAGS_tpu_name << std::endl;
  TF_CHECK_OK(main_session->Run({}, {}, {"ConfigureDistributedTPU"}, nullptr));

  auto* env = Env::Default();
  TF_CHECK_OK(ReadBinaryProto(env, FLAGS_model, &graph_def));

  int num_replicas = 0;
  for (const auto& node : graph_def.node()) {
    absl::string_view name = node.name();
    if (absl::ConsumePrefix(&name, "pos_tensor_")) {
      int replica;
      absl::SimpleAtoi(name, &replica);
      num_replicas = std::max(num_replicas, replica + 1);
    }
  }
  std::cout << "Found " << num_replicas << " model replicas" << std::endl;

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < FLAGS_num_threads; ++i) {
    threads.push_back(absl::make_unique<Thread>(i, graph_def, FLAGS_tpu_name,
                                                num_replicas));
  }
  auto start = absl::Now();
  for (auto& t : threads) {
    t->Run();
  }
  for (auto& t : threads) {
    t->Join();
  }
  auto elapsed = absl::Now() - start;
  std::cout << "Ran " << total_inferences << " inferences in " << elapsed << std::endl;

  std::cout << "Shutting down TPU" << std::endl;
  TF_CHECK_OK(main_session->Run({}, {}, {"ShutdownDistributedTPU"}, nullptr));
  TF_CHECK_OK(main_session->Close());
}

} // namespace minigo

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  minigo::MultithreadTest();

  return 0;
}

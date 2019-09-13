// Copyright 2018 Google LLC
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

#include "cc/dual_net/tpu_round_robin_dual_net.h"

#include <algorithm>
#include <thread>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/logging.h"
#include "cc/model/buffered_model.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "wtf/macros.h"

using tensorflow::DT_FLOAT;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace minigo {

namespace {

constexpr int kNumCores = 8;

// A GraphDef containing the ops required to initialize and shutdown a TPU.
// This proto was generated from the script oneoffs/generate_tpu_graph_def.py.
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

class RoundRobinModel : public Model {
 public:
  RoundRobinModel(std::string name,
                  std::vector<std::unique_ptr<BufferedModel>> impls)
    : Model(std::move(name), kNumCores + 1),
      impls_(std::move(impls)) {
  }

  void RunMany(const std::vector<const Input*>& inputs,
               std::vector<Output*>* outputs,
               std::string* model_name) override {
    auto idx = next_idx_.fetch_add(1);
    impls_[idx % impls_.size()]->RunMany(inputs, outputs, model_name);
  }

 private:
  std::atomic<size_t> next_idx_{0};
  std::vector<std::unique_ptr<BufferedModel>> impls_;
};

void PlaceOnDevice(GraphDef* graph_def, const std::string& device) {
  MG_LOG(INFO) << "PlaceOnDevice(\"" << device << "\")";
  for (auto& node : *graph_def->mutable_node()) {
    // if (node.op() == "Const") {
    //   auto it = node.attr().find("dtype");
    //   if (it != node.attr().end() && it->second.type() == DT_INT32) {
    //     continue;  // Const nodes of type int32 need to be in CPU.
    //   }
    // }
    node.set_device(device);
  }
}

std::unique_ptr<Session> CreateSession(const GraphDef& graph_def,
                                       const std::string& tpu_name) {
  // Make sure tpu_name looks like a valid name.
  MG_CHECK(absl::StartsWith(tpu_name, "grpc://"));

  SessionOptions options;
  options.target = tpu_name;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));
  return session;
}

}  // namespace

TpuRoundRobinDualNet::TpuRoundRobinDualNet(const std::string& tpu_name,
                       const std::string& graph_path,
                       const tensorflow::GraphDef& graph_def,
                       int device)
    : DualNet(std::string(file::Stem(graph_path))),
      device_(device) {
  session_ = CreateSession(graph_def, tpu_name);

  output_names_.emplace_back("tpu_policy_output");
  output_names_.emplace_back("tpu_value_output");

  // Run warm-up inferences on all sessions.
  // Tensorflow lazily initializes the first time Session::Run is called,
  // which can take hundreds of milliseconds. This interfers with time control,
  // so explicitly run inference once during construction.
  /// MG_LOG(INFO) << "Running warm-up inferences";
  /// Position::Stones stones;
  /// Input input;
  /// input.to_play = Color::kBlack;
  /// input.sym = symmetry::kIdentity;
  /// input.position_history.push_back(&stones);
  /// Output output;
  /// std::vector<const Input*> inputs = {&input};
  /// std::vector<Output*> outputs = {&output};
  /// RunMany(inputs, &outputs, nullptr);
}

TpuRoundRobinDualNet::~TpuRoundRobinDualNet() {
  MG_LOG(INFO) << "Closing worker session";
  TF_CHECK_OK(session_->Close());
}

void TpuRoundRobinDualNet::RunManyImpl(std::string* model_name) {
  size_t num_features = features_.size();
  Reserve(num_features);

  auto* feature_data = inputs_[0].second.flat<float>().data();
  // Copy the features into the input tensor.
  for (const auto& feature : features_) {
    feature_data = std::copy(feature.begin(), feature.end(), feature_data);
  }

  // Run the model.
  {
    WTF_SCOPE("Session::Run", size_t, int)(batch_capacity_, device_);
    TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));
  }

  // Copy the policy and value out of the output tensors.
  const auto& policy_tensor = outputs_[0].flat<float>();
  const auto& value_tensor = outputs_[1].flat<float>();
  for (size_t i = 0; i < num_features; ++i) {
    auto& output = raw_outputs_[i];
    memcpy(output.policy.data(), policy_tensor.data() + i * kNumMoves,
           sizeof(output.policy));
    output.value = value_tensor.data()[i];
  }

  if (model_name != nullptr) {
    *model_name = graph_path_;
  }
}

void TpuRoundRobinDualNet::Reserve(size_t capacity) {
  MG_CHECK(capacity > 0);
  if (capacity <= batch_capacity_ && capacity > 3 * batch_capacity_ / 4) {
    return;
  }
  inputs_.clear();
  inputs_.emplace_back(
      "pos_tensor", Tensor(DT_FLOAT, TensorShape({static_cast<int>(capacity),
                                                  kN, kN, kNumStoneFeatures})));
  batch_capacity_ = capacity;
}

TpuRoundRobinDualNetFactory::TpuRoundRobinDualNetFactory(int buffer_count, std::string tpu_name)
    : tpu_name_(std::move(tpu_name)), buffer_count_(buffer_count) {
  // Create a session containing ops for initializing & shutting down a TPU.
  GraphDef graph_def;
  ::tensorflow::protobuf::TextFormat::ParseFromString(kTpuOpsGraphDef,
                                                      &graph_def);
  main_session_ = CreateSession(graph_def, tpu_name_);

  MG_LOG(INFO) << "Initializing TPU " << tpu_name_;
  TF_CHECK_OK(main_session_->Run({}, {}, {"ConfigureDistributedTPU"}, nullptr));
}

TpuRoundRobinDualNetFactory::~TpuRoundRobinDualNetFactory() {
  MG_LOG(INFO) << "Shutting down TPU " << tpu_name_;
  TF_CHECK_OK(main_session_->Run({}, {}, {"ShutdownDistributedTPU"}, nullptr));

  MG_LOG(INFO) << "Closing main session";
  TF_CHECK_OK(main_session_->Close());
}

std::unique_ptr<Model> TpuRoundRobinDualNetFactory::NewModel(
    const std::string& descriptor) {
  GraphDef graph_def;
  auto* env = Env::Default();
  TF_CHECK_OK(ReadBinaryProto(env, descriptor, &graph_def));

  // Check that we're actually loading a TPU model.
  bool found_tpu_op = false;
  for (const auto& node : graph_def.node()) {
    if (absl::StartsWithIgnoreCase(node.name(), "tpu")) {
      found_tpu_op = true;
      break;
    }
  }
  MG_CHECK(found_tpu_op) << "didn't find any ops starting with \"tpu\" this "
                            "model looks like it wasn't compiled for TPU";

  std::vector<std::unique_ptr<BufferedModel>> buffered_models;
  for (int device = 0; device < kNumCores; ++device) {
    PlaceOnDevice(&graph_def, absl::StrCat("/device:TPU:", device));

    std::vector<std::unique_ptr<Model>> models;
    for (int i = 0; i < 2; ++i) {
      models.push_back(absl::make_unique<TpuRoundRobinDualNet>(
            tpu_name_, descriptor, graph_def, device));
    }
    buffered_models.push_back(absl::make_unique<BufferedModel>(
          descriptor, std::move(models)));
  }

  return absl::make_unique<RoundRobinModel>(descriptor,
                                            std::move(buffered_models));
}

}  // namespace minigo

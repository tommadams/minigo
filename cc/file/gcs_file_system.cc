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

#include "cc/file/gcs_file_system.h"

#include <iterator>

#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "cc/logging.h"

namespace minigo {
namespace file {
namespace internal {

namespace {
std::pair<std::string, std::string> ParseGcsPath(std::string path) {
  path = NormalizeSlashesImpl(std::move(path), '\\', '/');
  absl::string_view p = path;
  MG_CHECK(absl::ConsumePrefix(&p, "gs://"));
  std::pair<std::string, std::string> result =
      absl::StrSplit(p, absl::MaxSplits('/', 1));
  return result;
}
}  // namespace

GcsFileSystem::GcsFileSystem() {
  auto status_or_client = google::cloud::storage::Client::CreateDefaultClient();
  MG_CHECK(status_or_client)
      << "Unable to create default GCS client: " << status_or_client.status();
  client_ = std::move(*status_or_client);
}

FileSystem* GcsFileSystem::Get() {
  static auto* impl = new GcsFileSystem();
  return impl;
}

bool GcsFileSystem::RecursivelyCreateDir(std::string path) {
  // GCS doesn't have a concept of directories.
  return true;
}

bool GcsFileSystem::WriteFile(std::string path, absl::string_view contents) {
  auto spec = ParseGcsPath(std::move(path));
  auto stream = client_.WriteObject(spec.first, spec.second);
  stream.write(contents.data(), contents.size());
  stream.Close();
  if (stream.bad()) {
    MG_LOG(ERROR) << stream.status();
    return false;
  }
  return true;
}

bool GcsFileSystem::ReadFile(std::string path, std::string* contents) {
  auto spec = ParseGcsPath(std::move(path));
  auto stream = client_.ReadObject(spec.first, spec.second);
  *contents = std::string(std::istreambuf_iterator<char>{stream}, {});
  if (stream.bad()) {
    MG_LOG(ERROR) << stream.status();
    return false;
  }
  return true;
}

bool GcsFileSystem::GetModTime(std::string path, uint64_t* mtime_usec) {
  auto spec = ParseGcsPath(std::move(path));
  auto metadata = client_.GetObjectMetadata(spec.first, spec.second);
  if (!metadata) {
    MG_LOG(ERROR) << metadata.status();
    return false;
  }
  *mtime_usec = std::chrono::duration_cast<std::chrono::microseconds>(
      metadata.time_storage_class_updated().time_since_epoch());
  return true;
}

bool GcsFileSystem::ListDir(std::string directory,
                            std::vector<std::string>* files) {
  auto spec = ParseGcsPath(std::move(directory));
  for (const auto& metadata : client_.ListObjects(
           spec.first, google::cloud::storage::Prefix(spec.second))) {
    files->push_back(metadata.name());
  }
  return true;
}

std::string GcsFileSystem::NormalizeSlashes(std::string path) {
  return NormalizeSlashesImpl(path, '\\', '/');
}

}  // namespace internal
}  // namespace file
}  // namespace minigo

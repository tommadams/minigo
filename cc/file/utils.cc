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

#include "cc/file/utils.h"

#include "absl/strings/match.h"
#include "cc/file/file_system.h"
#include "cc/file/local_file_system.h"

#ifdef MG_ENABLE_GCS_FILE_SYSTEM
#include "cc/file/gcs_file_system.h"
#endif  // MG_ENABLE_GCS_FILE_SYSTEM

namespace minigo {
namespace file {

namespace {
internal::FileSystem* GetFileSystem(const std::string& path) {
#ifdef MG_ENABLE_GCS_FILE_SYSTEM
  if (absl::StartsWith(path, "gs://")) {
    return internal::GcsFileSystem::Get();
  }
#endif  // MG_ENABLE_GCS_FILE_SYSTEM
  return internal::LocalFileSystem::Get();
}
}  // namespace

bool RecursivelyCreateDir(std::string path) {
  return GetFileSystem(path)->RecursivelyCreateDir(std::move(path));
}

bool WriteFile(std::string path, absl::string_view contents) {
  return GetFileSystem(path)->WriteFile(std::move(path), contents);
}

bool ReadFile(std::string path, std::string* contents) {
  return GetFileSystem(path)->ReadFile(std::move(path), contents);
}

bool GetModTime(std::string path, uint64_t* mtime_usec) {
  return GetFileSystem(path)->GetModTime(std::move(path), mtime_usec);
}

bool ListDir(std::string directory, std::vector<std::string>* files) {
  return GetFileSystem(directory)->ListDir(std::move(directory), files);
}

}  // namespace file
}  // namespace minigo

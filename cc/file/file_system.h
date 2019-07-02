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

#ifndef CC_FILE_UTILS_FILE_SYSTEM_H_
#define CC_FILE_UTILS_FILE_SYSTEM_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace minigo {
namespace file {
namespace internal {

std::string NormalizeSlashesImpl(std::string path, char bad, char good);

class FileSystem {
 public:
  virtual ~FileSystem() = default;
  virtual bool RecursivelyCreateDir(std::string path) = 0;
  virtual bool WriteFile(std::string path, absl::string_view contents) = 0;
  virtual bool ReadFile(std::string path, std::string* contents) = 0;
  virtual bool GetModTime(std::string path, uint64_t* mtime_usec) = 0;
  virtual bool ListDir(std::string directory,
                       std::vector<std::string>* files) = 0;
  virtual std::string NormalizeSlashes(std::string path) = 0;
};

}  // namespace internal
}  // namespace file
}  // namespace minigo

#endif  //

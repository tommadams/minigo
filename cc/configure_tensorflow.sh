#!/bin/bash

set -e

_DEFAULT_CUDA_VERSION=10
_DEFAULT_CUDNN_VERSION=7

# Run the TensorFlow configuration script, setting reasonable values for most
# of the options.
echo "Configuring tensorflow"
CC_OPT_FLAGS="${CC_OPT_FLAGS:--march=native}" \
CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_PATH:-/usr/local/cuda} \
CUDNN_INSTALL_PATH=${CUDNN_INSTALL_PATH:-/usr/local/cuda} \
TF_NEED_JEMALLOC=${TF_NEED_JEMALLOC:-1} \
TF_NEED_GCP=${TF_NEED_GCP:-1} \
TF_CUDA_VERSION=${TF_CUDA_VERSION:-$_DEFAULT_CUDA_VERSION} \
TF_CUDNN_VERSION=${TF_CUDNN_VERSION:-$_DEFAULT_CUDNN_VERSION} \
TF_NEED_HDFS=${TF_NEED_HDFS:-0} \
TF_ENABLE_XLA=${TF_ENABLE_XLA:-1} \
TF_NEED_S3=${TF_NEED_S3:-0} \
TF_NEED_KAFKA=${TF_NEED_KAFKA:-0} \
TF_NEED_CUDA=${TF_NEED_CUDA:-1} \
TF_NEED_GDR=${TF_NEED_GDR:-0} \
TF_NEED_VERBS=${TF_NEED_VERBS:-0} \
TF_NEED_OPENCL_SYCL=${TF_NEED_OPENCL_SYCL:-0} \
TF_CUDA_CLANG=${TF_CUDA_CLANG:-0} \
TF_NEED_ROCM=${TF_NEED_ROCM:-0} \
TF_NEED_TENSORRT=${TF_NEED_TENSORRT:-0} \
TF_NEED_MPI=${TF_NEED_MPI:-0} \
TF_SET_ANDROID_WORKSPACE=${TF_SET_ANDROID_WORKSPACE:-0} \
bazel --bazelrc=/dev/null run @org_tensorflow//:configure

output_base=$(bazel info output_base)
workspace=$(bazel info workspace)

# Copy TensorFlow's bazelrc files to workspace.
cp ${output_base}/external/org_tensorflow/.bazelrc ${workspace}/tensorflow.bazelrc
cp ${output_base}/external/org_tensorflow/.tf_configure.bazelrc ${workspace}/tf_configure.bazelrc

echo "Building tensorflow package"
bazel run -c opt \
  --copt=-Wno-comment \
  --copt=-Wno-deprecated-declarations \
  --copt=-Wno-ignored-attributes \
  --copt=-Wno-maybe-uninitialized \
  --copt=-Wno-sign-compare \
  //cc/tensorflow:build -- ${workspace}/cc/tensorflow

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache License 2.0

cc_library(
    name = "path",
    srcs = ["path.cc"],
    hdrs = ["path.h"],
    deps = [
        "//cc/platform",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file",
    srcs = [
        "utils.cc",
        "file_system.h",
        "file_system.cc",
        "local_file_system.h",
    ] + select({
        "//cc/config:enable_tf": [
            "gcs_file_system.h",
            "gcs_file_system.cc",
        ],
        "//conditions:default": [],
    }) + select({
        "@bazel_tools//src/conditions:windows": [
            "local_file_system_windows.cc",
        ],
        "//conditions:default": [
            "local_file_system_posix.cc",
        ],
    }),
    hdrs = [
        "path.h",
        "utils.h",
    ],
    copts = select({
        "//cc/config:enable_tf": ["-DMG_ENABLE_GCS_FILE_SYSTEM"],
        "//conditions:default": [],
    }),
    deps = [
        ":path",
        "//cc:logging",
        "//cc/platform",
        "@com_google_absl//absl/strings",
    ] + select({
        "//cc/config:enable_tf": [
            "@com_github_googlecloudplatform_google_cloud_cpp//google/cloud/storage:storage_client",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "path_test",
    srcs = ["path_test.cc"],
    deps = [
        ":file",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":file",
        "//cc:base",
        "//cc:logging",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

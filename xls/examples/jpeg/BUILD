# Copyright 2020 The XLS Authors
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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "get_mangled_ir_symbol",
    "xls_dslx_opt_ir",
    "xls_eval_ir_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],
)

xls_dslx_opt_ir(
    name = "idct_chen",
    src = "idct_chen.x",
    ir_file = "idct_chen.ir",
    opt_ir_args = {
        "entry": get_mangled_ir_symbol("idct_chen", "idct"),
    },
    opt_ir_file = "idct_chen.opt.ir",
    # Note: times out in fastbuild 2020-11-09.
    tags = ["optonly"],
)

# TODO: 2021-05-28 Add xls_ir_equivalence_test. Note: times out.
xls_eval_ir_test(
    name = "idct_chen_eval_ir_test",
    src = "idct_chen.ir",
    ir_eval_args = {
        "entry": get_mangled_ir_symbol("idct_chen", "idct"),
    },
    # Note: times out in fastbuild 2020-11-09.
    tags = ["optonly"],
)

cc_xls_ir_jit_wrapper(
    name = "idct_chen_jit_wrapper",
    src = "idct_chen.opt.ir",
    jit_wrapper_args = {
        "class_name": "idct_chen",
    },
)

cc_test(
    name = "idct_chen_jit_wrapper_test",
    srcs = ["idct_chen_jit_wrapper_test.cc"],
    deps = [
        ":idct_chen_jit_wrapper",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

# -- GRM

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "streams",
    srcs = ["streams.cc"],
    hdrs = ["streams.h"],
    deps = [
        ":constants",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
    ],
)

cc_test(
    name = "streams_test",
    srcs = ["streams_test.cc"],
    deps = [
        ":constants",
        ":streams",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

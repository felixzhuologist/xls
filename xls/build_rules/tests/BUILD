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
    "xls_benchmark_ir",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
    "xls_dslx_verilog",
    "xls_eval_ir_test",
    "xls_ir_equivalence_test",
    "xls_ir_jit_wrapper",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

# Ensure that we can take a dep on a genrule .x file and import it.
genrule(
    name = "generated_x",
    outs = ["generated.x"],
    cmd = "echo 'pub const VALUE = u32:42;' > $(OUTS) && " +
          "echo 'fn main() -> u32 { VALUE }' >> $(OUTS) && " +
          "echo '#![test]' >> $(OUTS) && " +
          "echo 'fn test_value() { assert_eq(VALUE, u32:42) }' >> $(OUTS)",
)

# Check that we can generate a block a couple different ways.
xls_dslx_verilog(
    name = "add_one_pipeline_rtl",
    src = "add_one.x",
    codegen_args = {
        "pipeline_stages": "2",
        "delay_model": "unit",
    },
    verilog_file = "add_one_pipeline_rtl.v",
)

sh_test(
    name = "add_one_pipeline_rtl_has_clk_test",
    srcs = ["has_clk.sh"],
    args = ["$(rootpath :add_one_pipeline_rtl.v)"],
    data = [":add_one_pipeline_rtl.v"],
)

xls_dslx_verilog(
    name = "add_one_combinational_rtl",
    src = "add_one.x",
    codegen_args = {
        "generator": "combinational",
    },
    verilog_file = "add_one_combinational_rtl.v",
)

sh_test(
    name = "add_one_combinational_rtl_has_clk_test",
    srcs = ["not_has_clk.sh"],
    args = ["$(rootpath :add_one_combinational_rtl.v)"],
    data = [":add_one_combinational_rtl.v"],
)

# Run the test inside of the generated file.
xls_dslx_test(
    name = "generated_file_as_input_test",
    src = "generated.x",
)

# Encapsulate the generated file as a library.
xls_dslx_library(
    name = "generated_dslx",
    srcs = ["generated.x"],
)

# Reference the library with the generated file inside of a xls_dslx_test rule.
xls_dslx_test(
    name = "dependent_on_library_with_generated_file_test",
    src = "importer.x",
    deps = [
        ":generated_dslx",
    ],
)

xls_dslx_library(
    name = "a_dslx",
    srcs = [
        "a.x",
    ],
)

# xls_dslx_library target depends on xls_dslx_library target 'a_dslx'.
xls_dslx_library(
    name = "bc_dslx",
    srcs = [
        "b.x",
        "c.x",
    ],
    deps = [
        ":a_dslx",
    ],
)

# xls_dslx_test with a transitive dependency.
xls_dslx_test(
    name = "d_dslx_test",
    src = "d.x",
    deps = [
        ":bc_dslx",
    ],
)

################################################################################
# Simple example with rule chaining.
#
# Below is a diagram illustrating the rule chaining example. The build flow
# is on the left and the test flow is on the right. The flow is from top to
# down. The names of the rules target names are displayed.
#
# Overview:
# The 'chaining_example.x' file is the input to the build and test flow. In the
# build flow, the file is the input to a xls_dslx_ir rule named
# 'chaining_example_ir'. The output of the latter is the input to a
# xls_ir_opt_ir rule named 'chaining_example_opt_ir'. The output of the latter
# is the input to a xls_ir_verilog rule named chaining_example_verilog.
#
# In the test flow, the 'chaining_example.x' file is the input to a
# xls_dslx_test rule named 'chaining_example_test'. The output of the
# 'chaining_example_ir' rule is the input to a xls_eval_ir_test rule named
# 'chaining_example_eval_ir_test', and the input to a xls_ir_equivalence_test
# rule named chaining_example_ir_equivalence_test. The output of the
# 'chaining_example_opt_ir' rule is the input to a xls_benchmark_ir rule named
# 'chaining_example_benchmark_ir', and the input to a xls_ir_equivalence_test
# rule named chaining_example_ir_equivalence_test.
#
#           Build Flow                  Test/Execution Flow
#           ----------                  -------------------
#
#                     chaining_example.x
#                             |
#               ---------------------------------
#              |                                |
#              |                      chaining_example_test
#              |
#              |               |--> chaining_example_eval_ir_test
#              |               |
#      chaining_example_ir  ---|
#              |               |
#              |               |----------------------------------------------|
#              |                                                              |
#              |                  |--> chaining_example_ir_equivalence_test <-|
#              |                  |
#     chaining_example_opt_ir  ---|
#              |                  |
#    chaining_example_verilog     |--> chaining_example_benchmark_ir
################################################################################

xls_dslx_test(
    name = "chaining_example_test",
    src = "chaining_example.x",
)

xls_dslx_ir(
    name = "chaining_example_ir",
    src = "chaining_example.x",
)

xls_ir_opt_ir(
    name = "chaining_example_opt_ir",
    src = ":chaining_example_ir",
    opt_ir_file = ":chaining_example_opt_ir.opt.ir",
)

xls_ir_verilog(
    name = "chaining_example_verilog",
    src = ":chaining_example_opt_ir",
    codegen_args = {
        "pipeline_stages": "1",
    },
)

xls_eval_ir_test(
    name = "chaining_example_eval_ir_test",
    src = ":chaining_example_ir",
)

xls_ir_equivalence_test(
    name = "chaining_example_ir_equivalence_test",
    src_0 = ":chaining_example_ir",
    src_1 = ":chaining_example_opt_ir",
)

xls_benchmark_ir(
    name = "chaining_example_benchmark_ir",
    src = ":chaining_example_opt_ir",
)

################################################################################
# An alternate form for xls_dslx_test in the simple example with rule chaining.
#
# Below is a diagram illustrating an alternate form for xls_dslx_test in the
# simple example with rule chaining. The build flow is on the left and the test
# flow is on the right. The flow is from top to down. The names of the rules
# target names are displayed.
#
# Overview:
# The 'chaining_example_ir' target is the input to a xls_dslx_test rule named
# 'chaining_example_alternate_test'. Note for the xls_dslx_test rule, either the
# 'src' or 'dep' attribute must be specified.
#
#           Build Flow
#           ----------
#        chaining_example_ir   (See the simple example with chaining the
#               |                  rules above.)
#               |
#               |
#               |                     Test Flow
#               |                     ---------
#               |
#               |---------- chaining_example_alternate_test
################################################################################

xls_dslx_test(
    name = "chaining_example_alternate_test",
    dep = ":chaining_example_ir",
)

################################################################################
# Extending the simple example with rule chaining.
#
# Below is a diagram illustrating an extension to the build flow of the rule
# chaining example. The names of the rules target names and outputs of the
# targets are displayed. The flow is from top to down.
#
# Overview:
# The 'chaining_example_opt_ir.opt.ir' file, the output from the
# 'chaining_example_opt_ir' target from the simple example with chaining, is the
# input to the cc_xls_ir_jit_wrapper macro named
# 'chaining_example_opt_ir_jit_wrapper_main_cc'.
#
# The 'chaining_example_ir' target is the input to two xls_ir_opt_ir rules named
# 'chaining_example_double_opt_ir' and
# 'chaining_example_triple_run_only_opt_ir'. The output of the
# chaining_example_double_opt_ir target is the input to a
# cc_xls_ir_jit_wrapper macro named
# 'chaining_example_opt_ir_jit_wrapper_double_cc'. Note the reuse of the
# chaining_example_ir target as input to two xls_ir_opt_ir rules using different
# entry points.
#
# The output of the 'chaining_example_triple_run_only_opt_ir' target is the
# input to a xls_ir_opt_ir rule named
# 'chaining_example_triple_skip_passes_opt_ir'. Note the chaining of two
# xls_ir_opt_ir rules.
#
#           Build Flow
#           ----------
#  chaining_example_opt_ir.opt.ir (output from chaining_example_opt_ir target -
#               |                  See the simple example with chaining the
#               |                  rules above.)
#               |
#               |------------->  chaining_example_opt_ir_jit_wrapper_main_cc
#
#
#
#  chaining_example_ir (Target from the simple example with chaining the rules
#               |       shown above.)
#               |
#               |------------->  chaining_example_double_opt_ir
#               |           (Optimizes at the 'double' entry point)
#               |                              |
#               |                              |
#               |                              |
#               |        chaining_example_opt_ir_jit_wrapper_double_cc
#               |
#               |
#               |------------->  chaining_example_triple_run_only_opt_ir
#                                (Optimizes at the 'triple' entry point
#                                          with selected passes.)
#                                                  |
#                                                  |
#                                                  |
#                                chaining_example_triple_skip_passes_opt_ir
#                                (Optimizes by skipping selected passes.)
#
# Below is a diagram illustrating the test flow of an extension to the rule
# chaining example. The names of the rules target names and outputs of the
# targets are displayed. The flow is from top to down.
#
# Overview:
# The output of the 'chaining_example_triple_run_only_opt_ir' target and the
# output of the 'chaining_example_triple_skip_passes_opt_ir' are the inputs to
# an xls_ir_equivalence_test rule named
# 'chaining_example_triple_skip_passes_opt_ir'.
#
#           Test Flow
#           ----------
#    chaining_example_triple_run_only_opt_ir.opt.ir (output from
#               |                  chaining_example_triple_run_only_opt_ir
#               |                  target)
#               |
#               |--> chaining_example_triple_ir_equivalence_test <--|
#                                                                   |
#                                                                   |
#              chaining_example_triple_skip_passes_opt_ir.opt.ir----|
#            (output from chaining_example_triple_skip_passes_opt_ir target)
################################################################################

cc_xls_ir_jit_wrapper(
    name = "chaining_example_opt_ir_jit_wrapper_main_cc",
    src = ":chaining_example_opt_ir.opt.ir",
)

xls_ir_opt_ir(
    name = "chaining_example_double_opt_ir",
    src = ":chaining_example_ir",
    opt_ir_args = {
        "entry": get_mangled_ir_symbol("chaining_example", "double"),
    },
    opt_ir_file = ":chaining_example_double_opt_ir.opt.ir",
)

cc_xls_ir_jit_wrapper(
    name = "chaining_example_opt_ir_jit_wrapper_double_cc",
    src = ":chaining_example_double_opt_ir.opt.ir",
    jit_wrapper_args = {
        "class_name": "chaining_example",
        "function": get_mangled_ir_symbol("chaining_example", "double"),
    },
)

xls_ir_opt_ir(
    name = "chaining_example_triple_run_only_opt_ir",
    src = ":chaining_example_ir",
    opt_ir_args = {
        "entry": get_mangled_ir_symbol("chaining_example", "triple"),
        "run_only_passes": "arith_simp",
    },
    opt_ir_file = ":chaining_example_triple_run_only_opt_ir.opt.ir",
)

xls_ir_opt_ir(
    name = "chaining_example_triple_skip_passes_opt_ir",
    src = ":chaining_example_triple_run_only_opt_ir",
    opt_ir_args = {
        "skip_passes": "arith_simp",
    },
    opt_ir_file = ":chaining_example_triple_skip_passes_opt_ir.opt.ir",
)

xls_ir_equivalence_test(
    name = "chaining_example_triple_ir_equivalence_test",
    src_0 = ":chaining_example_triple_run_only_opt_ir.opt.ir",
    src_1 = ":chaining_example_triple_skip_passes_opt_ir.opt.ir",
)

################################################################################
# Simple example hierarchy layout:
#     one        two        three  -> DSLX libraries (compile and parse)
#      \__________|__________/
#                 |
#               four      -> DSLX interpreter test
#                |
#               five      -> synthesis, JIT wrapper and tests
################################################################################

# xls_dslx_library target grouping DSLX files.
xls_dslx_library(
    name = "simple_example_123_dslx",
    srcs = [
        "simple_example_one.x",
        "simple_example_three.x",
        "simple_example_two.x",
    ],
)

# Execute the DSLX interpreter.
xls_dslx_test(
    name = "simple_example_four_test",
    src = "simple_example_four.x",
    deps = [":simple_example_123_dslx"],
)

# xls_dslx_library target depends on xls_dslx_library target
# 'simple_example_123_dslx'.
xls_dslx_library(
    name = "simple_example_4_dslx",
    srcs = ["simple_example_four.x"],
    deps = [":simple_example_123_dslx"],
)

# Generate verilog from a dslx file.
# The rule performs: DSLX -> IR -> Optimized IR -> Verilog.
xls_dslx_verilog(
    name = "simple_example_5_one_stage",
    src = "simple_example_five.x",
    codegen_args = {
        "pipeline_stages": "1",
    },
    # Output the Optimized IR file to reference.
    opt_ir_file = "simple_example_5_one_stage.opt.ir",
    deps = [":simple_example_4_dslx"],
)

# The xls_ir_jit_wrapper rule using an optimized IR file as input.
# The header and source files are outputs of the rule, they can be referenced
# by other rules.
xls_ir_jit_wrapper(
    name = "simple_example_5_one_stage_jit_wrapper",
    src = ":simple_example_5_one_stage.opt.ir",
    header_file = "simple_example_5_one_stage_jit_wrapper.h",
    source_file = "simple_example_5_one_stage_jit_wrapper.cc",
)

# cc_xls_ir_jit_wrapper macro with a cc_library target with name
# simple_example_5_one_stage_jit_wrapper_cc created.
# The target can be referenced by other cc_* rules.
#
# For example:
#     cc_library (
#         ...
#         deps = [
#             ...
#             ":simple_example_5_one_stage_jit_wrapper_cc",
#             ...
#         ]
#     )
#
cc_xls_ir_jit_wrapper(
    name = "simple_example_5_one_stage_jit_wrapper_cc",
    src = ":simple_example_5_one_stage.opt.ir",
)

# The rule performs tests using the DSLX, IR and Optimized IR files from the
# xls_dslx_verilog. See the xls_dslx_opt_ir_test rule description for a list of
# tests.
# Note the rule can depend on a xls_dslx_verilog target because the target
# generates all the data that the xls_dslx_opt_ir_test rule requires for its
# tests.
xls_dslx_opt_ir_test(
    name = "simple_example_5_dslx_opt_ir_test",
    dep = ":simple_example_5_one_stage",
)

################################################################################
# Examples using entry points.
################################################################################
# Base rule for entry points.
# The rule performs: DSLX -> IR -> Optimized IR.
xls_dslx_opt_ir(
    name = "simple_example_5_entry_five",
    src = "simple_example_five.x",
    ir_conv_args = {
        "entry": "five",
    },
    # Output the Optimized IR file to reference.
    deps = [":simple_example_4_dslx"],
)

# Universal entry example: The value is provided by the
# 'simple_example_5_entry_five' target.
xls_dslx_opt_ir_test(
    name = "simple_example_5_one_stage_entry_five_universal_test",
    dep = ":simple_example_5_entry_five",
)

# Global entry example: The value is provided by the rule's context using the
# 'entry' attribute. The global entry overwrites the universal entry.
xls_dslx_opt_ir_test(
    name = "simple_example_5_one_stage_entry_five_global_test",
    dep = ":simple_example_5_entry_five",
    entry = get_mangled_ir_symbol("simple_example_five", "five"),
)

# Local entry example: The value is provided by the arguments. The local entry
# overwrites the universal entry and global entry.
xls_dslx_opt_ir_test(
    name = "simple_example_5_one_stage_entry_five_local_test",
    benchmark_ir_args = {
        "entry": get_mangled_ir_symbol("simple_example_five", "five"),
    },
    dep = ":simple_example_5_entry_five",
    entry = "Global entry gets overwritten by local arguments.",
    ir_equivalence_args = {
        "function": get_mangled_ir_symbol("simple_example_five", "five"),
    },
    ir_eval_args = {
        "entry": get_mangled_ir_symbol("simple_example_five", "five"),
    },
)

################################################################################
# An example of xls_dslx_test using the 'dep' attribute on a target with a
# transitive dependency on a xls_dslx_library.
################################################################################

xls_dslx_test(
    name = "simple_example_5_one_stage_dslx_test",
    dep = ":simple_example_5_one_stage",
)

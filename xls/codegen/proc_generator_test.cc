// Copyright 2020 The XLS Authors
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

#include "xls/codegen/proc_generator.h"

#include "xls/codegen/signature_generator.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;

constexpr char kTestName[] = "proc_generator_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

class ProcGeneratorTest : public VerilogTestBase {
 protected:
  CodegenOptions codegen_options() {
    return CodegenOptions().use_system_verilog(UseSystemVerilog());
  }
};

TEST_P(ProcGeneratorTest, APlusB) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_ch,
      package.CreatePortChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_ch,
      package.CreatePortChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      package.CreatePortChannel("sum", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue a = pb.Receive(a_ch);
  BValue b = pb.Receive(b_ch);
  pb.Send(output_ch, pb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(codegen_options(), proc));

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(codegen_options(), proc));

  ModuleTestbench tb(verilog, sig, GetSimulator());

  tb.ExpectX("sum");
  // The combinational module doesn't a connected clock, but the clock can still
  // be used to sequence events in time.
  tb.NextCycle().Set("a", 0).Set("b", 0).ExpectEq("sum", 0);
  tb.NextCycle().Set("a", 100).Set("b", 42).ExpectEq("sum", 142);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ProcGeneratorTest, PipelinedAPlusB) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_ch,
      package.CreatePortChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_ch,
      package.CreatePortChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * p0_a_ch,
      package.CreateRegisterChannel("p0_a", u32,
                                    /*reset_value=*/Value(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * p0_b_ch,
      package.CreateRegisterChannel("p0_b", u32,
                                    /*reset_value=*/Value(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * p1_sum_ch,
      package.CreateRegisterChannel("p1_sum", u32,
                                    /*reset_value=*/Value(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      package.CreatePortChannel("sum", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);

  BValue a = pb.Receive(a_ch);
  BValue b = pb.Receive(b_ch);

  // Pipeline register 0.
  pb.Send(p0_a_ch, a);
  BValue p0_a = pb.Receive(p0_a_ch);
  pb.Send(p0_b_ch, b);
  BValue p0_b = pb.Receive(p0_b_ch);

  // Pipeline register 1.
  pb.Send(p1_sum_ch, pb.Add(p0_a, p0_b));
  BValue p1_sum = pb.Receive(p1_sum_ch);

  pb.Send(output_ch, p1_sum);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  CodegenOptions options = codegen_options()
                               .clock_name("the_clock")
                               .reset("the_reset", /*asynchronous=*/false,
                                      /*active_low=*/false);
  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog, GenerateVerilog(options, proc));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(options, proc));
  ModuleTestbench tb(verilog, sig, GetSimulator());

  tb.ExpectX("sum");
  tb.Set("a", 0).Set("b", 0);
  tb.AdvanceNCycles(2).ExpectEq("sum", 0);

  tb.Set("a", 100).Set("b", 42);
  tb.AdvanceNCycles(2).ExpectEq("sum", 142);

  tb.Set("the_reset", 1).NextCycle();
  tb.ExpectEq("sum", 0);

  tb.Set("the_reset", 0);
  tb.AdvanceNCycles(2).ExpectEq("sum", 142);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ProcGeneratorTest, RegisteredInputNoReset) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * input_ch,
      package.CreatePortChannel("foo", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * reg_ch,
                           package.CreateRegisterChannel("foo_reg", u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      package.CreatePortChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue data = pb.Receive(input_ch);
  pb.Send(reg_ch, data);
  BValue reg_data = pb.Receive(reg_ch);
  pb.Send(output_ch, reg_data);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  CodegenOptions options = codegen_options().clock_name("foo_clock");
  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog, GenerateVerilog(options, proc));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(options, proc));

  ModuleTestbench tb(verilog, sig, GetSimulator());

  tb.ExpectX("out");
  tb.Set("foo", 42).NextCycle().ExpectEq("out", 42);
  tb.Set("foo", 100).ExpectEq("out", 42).NextCycle().ExpectEq("out", 100);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ProcGeneratorTest, Accumulator) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * input_ch,
      package.CreatePortChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * accum_ch,
      package.CreateRegisterChannel("accum", u32,
                                    /*reset_value=*/Value(UBits(10, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      package.CreatePortChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue input = pb.Receive(input_ch);
  BValue accum = pb.Receive(accum_ch);
  BValue next_accum = pb.Add(input, accum);
  pb.Send(accum_ch, next_accum);
  pb.Send(output_ch, accum);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  CodegenOptions options =
      codegen_options().clock_name("clk").reset("rst_n", /*asynchronous=*/false,
                                                /*active_low=*/true);
  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog, GenerateVerilog(options, proc));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(options, proc));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb(verilog, sig, GetSimulator());

  tb.Set("in", 0).Set("rst_n", 0).NextCycle().Set("rst_n", 1);

  tb.ExpectEq("out", 10);
  tb.Set("in", 42).NextCycle().ExpectEq("out", 52);
  tb.Set("in", 100).NextCycle().ExpectEq("out", 152);

  tb.Set("in", 0).Set("rst_n", 0).NextCycle().Set("rst_n", 1);
  tb.ExpectEq("out", 10);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ProcGeneratorTest, ProcWithNonNilState) {
  Package package(TestBaseName());
  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value(UBits(42, 32)),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  EXPECT_THAT(
      GenerateVerilog(codegen_options(), proc).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("The proc state must be an empty tuple for codegen")));
}

TEST_P(ProcGeneratorTest, ProcWithStreamingChannel) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch,
      package.CreateStreamingChannel("a", ChannelOps::kReceiveOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value(UBits(42, 32)),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue sum = pb.Add(pb.GetStateParam(), pb.Receive(ch));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(sum));

  EXPECT_THAT(
      GenerateVerilog(codegen_options(), proc).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Only register and port channel are supported in codegen")));
}

TEST_P(ProcGeneratorTest, ResetValueWithoutResetSignal) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      package.CreatePortChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * reg_ch, package.CreateRegisterChannel(
                            "reg", u32, /*reset_value=*/Value(UBits(123, 32))));
  TokenlessProcBuilder pb(TestBaseName(),
                          /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue in = pb.Receive(in_ch);
  BValue reg_d = pb.Receive(reg_ch);
  pb.Send(reg_ch, pb.Add(in, reg_d));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  EXPECT_THAT(
      GenerateVerilog(codegen_options().clock_name("clk"), proc).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Must specify a reset signal if registers have a reset value")));
}

TEST_P(ProcGeneratorTest, ProcWithAssertNoLabel) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch,
      package.CreatePortChannel("a", ChannelOps::kReceiveOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  pb.Assert(pb.ULt(pb.Receive(ch), pb.Literal(UBits(42, 32))),
            "a is not greater than 42");

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  {
    // No format string.
    XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                             GenerateVerilog(codegen_options(), proc));
    if (UseSystemVerilog()) {
      EXPECT_THAT(
          verilog,
          HasSubstr(
              R"(assert ($isunknown(a < 32'h0000_002a) || a < 32'h0000_002a) else $fatal(0, "a is not greater than 42"))"));
    } else {
      EXPECT_THAT(verilog, Not(HasSubstr("assert")));
    }
  }

  {
    // With format string, no label.
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string verilog,
        GenerateVerilog(
            codegen_options()
                .reset("my_rst", /*asynchronous=*/false, /*active_low=*/false)
                .clock_name("my_clk")
                .assert_format(
                    R"(`MY_ASSERT({condition}, "{message}", {clk}, {rst}))"),
            proc));
    if (UseSystemVerilog()) {
      EXPECT_THAT(
          verilog,
          HasSubstr(
              R"(`MY_ASSERT(a < 32'h0000_002a, "a is not greater than 42", my_clk, my_rst))"));
    } else {
      EXPECT_THAT(verilog, Not(HasSubstr("assert")));
    }
  }

  // Format string with label but assert doesn't have label.
  EXPECT_THAT(
      GenerateVerilog(codegen_options()
                          .reset("my_rst", /*asynchronous=*/false,
                                 /*active_low=*/false)
                          .clock_name("my_clk")
                          .assert_format(R"({label} foobar)"),
                      proc),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Assert format string has '{label}' placeholder, "
                         "but assert operation has no label")));

  // Format string with clock but block doesn't have clock.
  EXPECT_THAT(
      GenerateVerilog(codegen_options()
                          .reset("my_rst", /*asynchronous=*/false,
                                 /*active_low=*/false)
                          .assert_format(R"({clk} foobar)"),
                      proc),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Assert format string has '{clk}' placeholder, "
                         "but block has no clock signal")));

  // Format string with reset but block doesn't have reset.
  EXPECT_THAT(
      GenerateVerilog(codegen_options().assert_format(R"({rst} foobar)"), proc),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Assert format string has '{rst}' placeholder, "
                         "but block has no reset signal")));

  // Format string with invalid placeholder.
  EXPECT_THAT(
      GenerateVerilog(
          codegen_options().assert_format(R"({foobar} blargfoobar)"), proc),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Invalid placeholder '{foobar}' in assert format string. "
                    "Supported placeholders: {clk}, {condition}, {label}, "
                    "{message}, {rst}")));
}

TEST_P(ProcGeneratorTest, ProcWithAssertWithLabel) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch,
      package.CreatePortChannel("a", ChannelOps::kReceiveOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  pb.Assert(pb.ULt(pb.Receive(ch), pb.Literal(UBits(42, 32))),
            "a is not greater than 42", "the_label");

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  {
    // No format string.
    XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                             GenerateVerilog(codegen_options(), proc));
    if (UseSystemVerilog()) {
      EXPECT_THAT(
          verilog,
          HasSubstr(
              R"(assert ($isunknown(a < 32'h0000_002a) || a < 32'h0000_002a) else $fatal(0, "a is not greater than 42"))"));
    } else {
      EXPECT_THAT(verilog, Not(HasSubstr("assert")));
    }
  }

  {
    // With format string.
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string verilog,
        GenerateVerilog(
            codegen_options()
                .reset("my_rst", /*asynchronous=*/false, /*active_low=*/false)
                .clock_name("my_clk")
                .assert_format(
                    R"({label}: `MY_ASSERT({condition}, "{message}", {clk}, {rst}) // {label})"),
            proc));
    if (UseSystemVerilog()) {
      EXPECT_THAT(
          verilog,
          HasSubstr(
              R"(the_label: `MY_ASSERT(a < 32'h0000_002a, "a is not greater than 42", my_clk, my_rst) // the_label)"));
    } else {
      EXPECT_THAT(verilog, Not(HasSubstr("assert")));
    }
  }
}

TEST_P(ProcGeneratorTest, ProcWithEmptyTupleElementInOutput) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      package.CreatePortChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      package.CreatePortChannel(
          "result", ChannelOps::kSendOnly,
          package.GetTupleType({package.GetTupleType({}), u32})));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue empty_tuple = pb.Tuple({});
  pb.Send(out_ch, pb.Tuple({empty_tuple, pb.Receive(in_ch)}));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  XLS_LOG(INFO) << package.DumpIr();
  XLS_ASSERT_OK(GenerateVerilog(codegen_options(), proc).status());
}

TEST_P(ProcGeneratorTest, PortOrderTest) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      PortChannel * a_ch,
      package.CreatePortChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      PortChannel * b_ch,
      package.CreatePortChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      PortChannel * c_ch,
      package.CreatePortChannel("c", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      PortChannel * output_ch,
      package.CreatePortChannel("sum", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue a = pb.Receive(a_ch);
  BValue b = pb.Receive(b_ch);
  BValue c = pb.Receive(c_ch);
  pb.Send(output_ch, pb.Add(pb.Add(a, b), c));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  {
    a_ch->SetPosition(0);
    b_ch->SetPosition(1);
    c_ch->SetPosition(2);
    output_ch->SetPosition(3);
    XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                             GenerateVerilog(codegen_options(), proc));
    EXPECT_THAT(verilog,
                HasSubstr("input wire [31:0] a,\n  input wire [31:0] b,\n  "
                          "input wire [31:0] c,\n  output wire [31:0] sum"));
  }

  {
    a_ch->SetPosition(2);
    b_ch->SetPosition(0);
    c_ch->SetPosition(1);
    output_ch->SetPosition(3);
    XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                             GenerateVerilog(codegen_options(), proc));
    EXPECT_THAT(verilog,
                HasSubstr("input wire [31:0] b,\n  input wire [31:0] c,\n  "
                          "input wire [31:0] a,\n  output wire [31:0] sum"));
  }

  {
    a_ch->SetPosition(3);
    b_ch->SetPosition(0);
    c_ch->SetPosition(1);
    output_ch->SetPosition(2);
    EXPECT_THAT(GenerateVerilog(codegen_options(), proc),
                StatusIs(absl::StatusCode::kUnimplemented,
                         HasSubstr("Output ports must be ordered after all "
                                   "input ports in the proc.")));
  }
}

TEST_P(ProcGeneratorTest, LoadEnables) {
  // Construct a block with two parallel data paths: "a" and "b". Each consists
  // of a single register with a load enable. Verify that the two load enables
  // work as expected.
  Package package(TestBaseName());

  Type* u1 = package.GetBitsType(1);
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_ch,
      package.CreatePortChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_le_ch,
      package.CreatePortChannel("a_le", ChannelOps::kReceiveOnly, u1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_ch,
      package.CreatePortChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_le_ch,
      package.CreatePortChannel("b_le", ChannelOps::kReceiveOnly, u1));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_reg,
      package.CreateRegisterChannel("a_reg", u32,
                                    /*reset_value=*/Value(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_reg,
      package.CreateRegisterChannel("b_reg", u32,
                                    /*reset_value=*/Value(UBits(43, 32))));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_out,
      package.CreatePortChannel("a_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_out,
      package.CreatePortChannel("b_out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);

  BValue a = pb.Receive(a_ch);
  BValue a_le = pb.Receive(a_le_ch);
  BValue b = pb.Receive(b_ch);
  BValue b_le = pb.Receive(b_le_ch);

  pb.SendIf(a_reg, a_le, a);
  BValue a_d = pb.Receive(a_reg);
  pb.SendIf(b_reg, b_le, b);
  BValue b_d = pb.Receive(b_reg);

  pb.Send(a_out, a_d);
  pb.Send(b_out, b_d);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  CodegenOptions options =
      codegen_options().clock_name("clk").reset("rst", /*asynchronous=*/false,
                                                /*active_low=*/false);
  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog, GenerateVerilog(options, proc));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(options, proc));
  ModuleTestbench tb(verilog, sig, GetSimulator());

  // Set inputs to zero and disable load-enables.
  tb.Set("a", 100).Set("b", 200).Set("a_le", 0).Set("b_le", 0).Set("rst", 1);
  tb.NextCycle();
  tb.Set("rst", 0);
  tb.NextCycle();

  // Outputs should be at the reset value.
  tb.ExpectEq("a_out", 42).ExpectEq("b_out", 43);

  // Outputs should remain at reset values after clocking because load enables
  // are unasserted.
  tb.NextCycle();
  tb.ExpectEq("a_out", 42).ExpectEq("b_out", 43);

  // Assert load enable of 'a'. Load enable of 'b' remains unasserted.
  tb.Set("a_le", 1);
  tb.NextCycle();
  tb.ExpectEq("a_out", 100).ExpectEq("b_out", 43);

  // Assert load enable of 'b'. Deassert load enable of 'a' and change a's
  // input. New input of 'a' should not propagate.
  tb.Set("a", 101).Set("a_le", 0).Set("b_le", 1);
  tb.NextCycle();
  tb.ExpectEq("a_out", 100).ExpectEq("b_out", 200);

  // Assert both load enables.
  tb.Set("b", 201).Set("a_le", 1).Set("b_le", 1);
  tb.NextCycle();
  tb.ExpectEq("a_out", 101).ExpectEq("b_out", 201);

  XLS_ASSERT_OK(tb.Run());
}

INSTANTIATE_TEST_SUITE_P(ProcGeneratorTestInstantiation, ProcGeneratorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<ProcGeneratorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls

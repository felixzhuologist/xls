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

#include "xls/passes/tuple_simplification_pass.h"

#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class TupleSimplificationPassTest : public IrTestBase {
 protected:
  TupleSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         TupleSimplificationPass().RunOnFunctionBase(
                             f, PassOptions(), &results));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                            .RunOnFunctionBase(f, PassOptions(), &results)
                            .status());
    // Return whether tuple simplification changed anything.
    return changed;
  }
};

TEST_F(TupleSimplificationPassTest, SingleSimplification) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x:bits[2], y:bits[42]) -> bits[42] {
        tuple.1: (bits[2], bits[42]) = tuple(x, y)
        ret tuple_index.2: bits[42] = tuple_index(tuple.1, index=1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 4);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 2);
  EXPECT_THAT(f->return_value(), m::Param("y"));
}

TEST_F(TupleSimplificationPassTest, NoSimplification) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: (bits[2], bits[42])) -> bits[42] {
        ret tuple_index.2: bits[42] = tuple_index(x, index=1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 2);
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(f->node_count(), 2);
}

TEST_F(TupleSimplificationPassTest, NestedSimplification) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[42], y: bits[3], z: bits[73]) -> bits[73] {
        tuple.1: (bits[42], bits[73]) = tuple(x, z)
        tuple.2: ((bits[42], bits[73]), bits[3]) = tuple(tuple.1, y)
        tuple.3: ((bits[42], bits[73]), ((bits[42], bits[73]), bits[3])) = tuple(tuple.1, tuple.2)
        tuple_index.4: ((bits[42], bits[73]), bits[3]) = tuple_index(tuple.3, index=1)
        tuple_index.5: (bits[42], bits[73]) = tuple_index(tuple_index.4, index=0)
        ret tuple_index.6: bits[73] = tuple_index(tuple_index.5, index=1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 9);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 3);
  EXPECT_THAT(f->return_value(), m::Param("z"));
}

TEST_F(TupleSimplificationPassTest, ChainOfTuplesSimplification) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[42], y: bits[3]) -> bits[42] {
        tuple.1: (bits[42], bits[3]) = tuple(x, y)
        tuple_index.2: bits[42] = tuple_index(tuple.1, index=0)
        tuple.3: (bits[42], bits[3]) = tuple(tuple_index.2, y)
        tuple_index.4: bits[42] = tuple_index(tuple.3, index=0)
        tuple.5: (bits[42], bits[3]) = tuple(tuple_index.4, y)
        ret tuple_index.6: bits[42] = tuple_index(tuple.5, index=0)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 8);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 2);
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(TupleSimplificationPassTest, SimpleUnboxingArray) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(x: bits[2]) -> bits[2] {
  array.2: bits[2][1] = array(x)
  literal.3: bits[1] = literal(value=0)
  ret array_index.4: bits[2] = array_index(array.2, literal.3)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(TupleSimplificationPassTest, UnboxingLiteralArray) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func() -> bits[2] {
  literal.1: bits[2][2] = literal(value=[0b00, 0b01])
  literal.2: bits[1] = literal(value=0)
  literal.3: bits[1] = literal(value=1)
  array_index.4: bits[2] = array_index(literal.1, literal.2)
  array_index.5: bits[2] = array_index(literal.1, literal.3)
  ret add.6: bits[2] = add(array_index.4, array_index.5)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Literal(0), m::Literal(1)));
}

TEST_F(TupleSimplificationPassTest, Reconstruct) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(x: bits[1], y: bits[2], z: bits[3]) -> (bits[1], bits[2], bits[3]) {
  tuple.1: (bits[1], bits[2], bits[3]) = tuple(x, y, z)
  tuple_index.2: bits[1] = tuple_index(tuple.1, index=0)
  tuple_index.3: bits[2] = tuple_index(tuple.1, index=1)
  tuple_index.4: bits[3] = tuple_index(tuple.1, index=2)
  ret tuple.5: (bits[1], bits[2], bits[3]) = tuple(tuple_index.2, tuple_index.3, tuple_index.4)
 }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 8);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 4);
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Param("x"), m::Param("y"), m::Param("z")));
}

TEST_F(TupleSimplificationPassTest, ReconstructParamTuple) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(data: (bits[1], bits[2], bits[3])) -> (bits[1], bits[2], bits[3]) {
  tuple_index.1: bits[1] = tuple_index(data, index=0)
  tuple_index.2: bits[2] = tuple_index(data, index=1)
  tuple_index.3: bits[3] = tuple_index(data, index=2)
  ret tuple.4: (bits[1], bits[2], bits[3]) = tuple(tuple_index.1, tuple_index.2, tuple_index.3)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("data"));
}

TEST_F(TupleSimplificationPassTest, ReconstructDifferentSize) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(data: (bits[1], bits[2], bits[3])) -> (bits[1], bits[2]) {
  tuple_index.1: bits[1] = tuple_index(data, index=0)
  tuple_index.2: bits[2] = tuple_index(data, index=1)
  ret tuple.3: (bits[1], bits[2]) = tuple(tuple_index.1, tuple_index.2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls

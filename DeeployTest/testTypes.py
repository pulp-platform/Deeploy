# ----------------------------------------------------------------------
#
# File: testTypes.py
#
# Last edited: 15.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle

import pytest

from Deeploy.AbstractDataTypes import PointerClass, StructClass
from Deeploy.CommonExtensions.DataTypes import IntegerDataTypes, int8_t, int16_t, int32_t
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, StructBuffer, TransientBuffer, VariableBuffer


def testImmediateSerialization():
    for _type in IntegerDataTypes:
        val = _type(17)
        assert pickle.loads(pickle.dumps(val)) == val, f"Serialized value {val} should be equal!"
        assert pickle.loads(pickle.dumps(_type))(17) == _type(17), f"Serialized type {_type} should be equal to type!"
    return True


def testStructSerialization():
    structType = {"f1": int32_t, "f2": int8_t}
    s = StructClass("s", structType)
    val = {"f1": 7, "f2": 8}

    assert pickle.loads(pickle.dumps(s))(val) == s(val), f"Serialized type {s} should be equal to type!"
    assert pickle.loads(pickle.dumps(s(val))) == s(val), f"Serialized type {s} should be equal to type!"
    globals()[s] = None
    return True


def testImmediateTypeEquivalence():
    for _type in IntegerDataTypes:
        val = _type(17)
        assert _type(17) == val, f"Type {_type} should be equal if content is the same!"
    return True


def testStructTypeEquivalence():
    structType = {"f1": int32_t, "f2": int8_t}
    s = StructClass("s4", structType)
    val = {"f1": 7, "f2": 8}
    assert s(val) == s(val), f"Type {s} should be equal if content is same!"

    return True


def testImmediatePromotion():
    with pytest.raises(Exception):
        _ = int8_t(2**7)
        _ = int16_t(2**15)
        _ = int32_t(2**31)
    a = int8_t(2**7 - 1)
    b = int16_t(2**15 - 1)
    c = int32_t(2**31 - 1)

    with pytest.raises(Exception):
        _ = int8_t(b)
        _ = int8_t(c)
        _ = int16_t(c)

    _ = int8_t(a)
    _ = int16_t(a)
    _ = int32_t(a)

    _ = int16_t(b)
    _ = int32_t(b)

    _ = int32_t(c)

    return True


def generateTestStruct() -> StructClass:
    testStructType = {"f1": int32_t, "f2": int8_t}
    s1 = StructClass("s2", testStructType)
    structType = {"f1": int32_t, "struct": s1}
    s = StructClass("s3", structType)
    return s, s1


def testStructPromotion():

    s, s1 = generateTestStruct()

    _ = s({"f1": 15, "struct": s1({"f1": 8, "f2": 8})})

    globals()["s"] = None
    globals()["s1"] = None


def testStructKeyChecking():

    s, _ = generateTestStruct()
    with pytest.raises(Exception):
        _ = s({"struct": {"f1": 2540, "f2": 17}})
    with pytest.raises(Exception):
        _ = s({"f1": 2**14, "struct": {"f2": 17}})
    with pytest.raises(Exception):
        _ = s({"f1": 2**14, "struct": {"f1": 2540, "f3": 18}})
    with pytest.raises(Exception):
        _ = s({"f1": 2**14, "strct": {"f1": 2540, "f3": 18}})

    _ = s({"struct": {"f1": 2540, "f2": 17}, "f1": 2**14})
    _ = s({"struct": {"f2": 17, "f1": 2540}, "f1": 2**14})

    return True


def testStructRecursiveEquivalence():

    s, _ = generateTestStruct()

    with pytest.raises(Exception):
        _ = s({"f1": 2**14, "struct": {"f1": 2540, "f2": 2**8}})

    _ = s({"f1": 2**14, "struct": {"f1": 2540, "f2": 18}})
    return True


def generateTestCtxt() -> NetworkContext:
    testCtxt = NetworkContext(VariableBuffer, ConstantBuffer, StructBuffer, TransientBuffer)

    var = ConstantBuffer("testConstant", shape = [
        16,
    ], values = [14] * 16)
    testCtxt.add(var, 'global')

    return testCtxt


def testPointerPromotion():

    testCtxt = generateTestCtxt()
    i8p = PointerClass(int8_t)
    i16p = PointerClass(int16_t)
    sp = PointerClass(generateTestStruct()[0])
    testCtxt.annotateType(name = "testConstant", _type = i8p)

    with pytest.raises(Exception):
        _ = i16p("testConstant", testCtxt)
    with pytest.raises(Exception):
        _ = sp("testConstant", testCtxt)
    _ = i8p("testConstant", testCtxt)

    return True


def testPointerSerialization():

    testCtxt = generateTestCtxt()

    i8p = PointerClass(int8_t)
    testCtxt.annotateType(name = "testConstant", _type = i8p)

    _ = i8p("testConstant", testCtxt)
    _ = i8p("testConstant", pickle.loads(pickle.dumps(testCtxt)))
    _ = pickle.loads(pickle.dumps(i8p))("testConstant", testCtxt)
    _ = pickle.loads(pickle.dumps(i8p))("testConstant", pickle.loads(pickle.dumps(testCtxt)))

    return True


def testPointerTypeEquivalence():

    testCtxt = generateTestCtxt()

    i8p = PointerClass(int8_t)
    i16p = PointerClass(int16_t)

    var = ConstantBuffer("testConstant2", shape = [
        16,
    ], values = [14] * 16)
    testCtxt.add(var, 'global')

    testCtxt.annotateType(name = "testConstant", _type = i8p)
    testCtxt.annotateType(name = "testConstant2", _type = i8p)

    with pytest.raises(Exception):
        _ = i16p("testConstant2", testCtxt)
        _ = i16p("testConstant", testCtxt)

    assert i8p("testConstant",
               testCtxt) != i8p("testConstant2",
                                testCtxt), "Pointers testConstant and testConstant2 should not be equal!"
    assert i8p("testConstant", testCtxt) == i8p("testConstant",
                                                testCtxt), "Pointers constructed from same reference should be equal!"

    assert pickle.loads(pickle.dumps(i8p("testConstant", testCtxt))) == i8p(
        "testConstant", testCtxt), "Pointers constructed from same reference should be equal!"

    assert pickle.loads(pickle.dumps(i8p("testConstant", pickle.loads(pickle.dumps(testCtxt))))) == i8p(
        "testConstant", testCtxt), "Pointers constructed from same reference should be equal!"
    assert pickle.loads(pickle.dumps(i8p("testConstant", testCtxt))) == i8p(
        "testConstant", testCtxt), "Pointers constructed from same reference should be equal!"

    return True


if __name__ == "__main__":
    testImmediateSerialization()
    testImmediatePromotion()
    testImmediateTypeEquivalence()

    testStructSerialization()
    testStructPromotion()
    testStructTypeEquivalence()
    testStructKeyChecking()
    testStructRecursiveEquivalence()

    testPointerSerialization()
    testPointerPromotion()
    testPointerTypeEquivalence()

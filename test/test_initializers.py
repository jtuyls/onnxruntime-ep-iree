#!/usr/bin/env python3
"""Test initializer handling: small (inline) vs large (IRPA parameter)."""

import glob
import os
import pathlib
import sys
import tempfile

import numpy as np
from onnx import TensorProto, helper

import test_utils


def get_iree_files():
    """Return current sets of IREE temp MLIR and IRPA files."""
    temp_dir = tempfile.gettempdir()
    mlir = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))
    irpa = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.irpa")))
    return mlir, irpa


def cleanup_files(files):
    """Remove a set of files."""
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass


def test_small_initializer_inline():
    """Small constant (16 bytes) should be inlined via dense_resource."""
    print("\n=== test_small_initializer_inline ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    # 2x2 float32 = 16 bytes, well under 256 threshold.
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[10, 20], [30, 40]], dtype=np.float32)

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=[2, 2],
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "D"], outputs=["C"])
    graph = helper.make_graph(
        [add_node, constant_node], "test_graph", [input_a], [output]
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    mlir_before, irpa_before = get_iree_files()
    model_path = test_utils.save_model(model)

    try:
        session = test_utils.create_session(
            model_path, device, {"target_arch": "host", "save_intermediates": "1"}
        )
        result = session.run(None, {"A": a})[0]
        expected = a + b

        if not np.allclose(result, expected, rtol=1e-5, atol=1e-5):
            print(f"FAIL: Values mismatch\n  Expected: {expected}\n  Got: {result}")
            return False
        print("  Inference result correct")

        # Check MLIR content.
        mlir_after, irpa_after = get_iree_files()
        new_mlir = mlir_after - mlir_before
        new_irpa = irpa_after - irpa_before

        if not new_mlir:
            print("FAIL: No MLIR file saved")
            return False

        mlir_content = open(list(new_mlir)[0]).read()

        if "dense_resource<" not in mlir_content:
            print("FAIL: MLIR should contain dense_resource<")
            return False
        if "dialect_resources" not in mlir_content:
            print("FAIL: MLIR should contain dialect_resources section")
            return False
        if "flow.parameter.named" in mlir_content:
            print("FAIL: MLIR should NOT contain flow.parameter.named for small init")
            return False
        print("  MLIR content verified (inline dense_resource)")

        # IRPA should exist but be empty (no parameters needed).
        if new_irpa:
            irpa_size = os.path.getsize(list(new_irpa)[0])
            if irpa_size > 0:
                print(f"FAIL: IRPA file should be empty, got {irpa_size} bytes")
                return False
            print(f"  IRPA file empty as expected")

        cleanup_files(new_mlir | new_irpa)
        # Also clean up VMFB files.
        temp_dir = tempfile.gettempdir()
        vmfb_after = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.vmfb")))
        cleanup_files(vmfb_after)

        print("PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_large_initializer_parameter():
    """Large constant (16384 bytes) should use IRPA parameter."""
    print("\n=== test_large_initializer_parameter ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    # 64x64 float32 = 16384 bytes, well over 256 threshold.
    a = np.random.rand(64, 64).astype(np.float32)
    b = np.random.rand(64, 64).astype(np.float32)

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [64, 64])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [64, 64])
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=[64, 64],
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "D"], outputs=["C"])
    graph = helper.make_graph(
        [add_node, constant_node], "test_graph", [input_a], [output]
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    mlir_before, irpa_before = get_iree_files()
    model_path = test_utils.save_model(model)

    try:
        session = test_utils.create_session(
            model_path, device, {"target_arch": "host", "save_intermediates": "1"}
        )
        result = session.run(None, {"A": a})[0]
        expected = a + b

        if not np.allclose(result, expected, rtol=1e-5, atol=1e-5):
            print(f"FAIL: Values mismatch")
            return False
        print("  Inference result correct")

        # Check MLIR content.
        mlir_after, irpa_after = get_iree_files()
        new_mlir = mlir_after - mlir_before
        new_irpa = irpa_after - irpa_before

        if not new_mlir:
            print("FAIL: No MLIR file saved")
            return False

        mlir_content = open(list(new_mlir)[0]).read()

        if 'flow.parameter.named<"model"::' not in mlir_content:
            print("FAIL: MLIR should contain flow.parameter.named")
            return False
        if "dense_resource<" in mlir_content:
            print("FAIL: MLIR should NOT contain dense_resource< for large init")
            return False
        if "dialect_resources" in mlir_content:
            print("FAIL: MLIR should NOT contain dialect_resources for large init")
            return False
        print("  MLIR content verified (parameter reference)")

        # IRPA should exist and have content.
        if not new_irpa:
            print("FAIL: No IRPA file was created")
            return False
        irpa_size = os.path.getsize(list(new_irpa)[0])
        if irpa_size == 0:
            print("FAIL: IRPA file should not be empty")
            return False
        print(f"  IRPA file size: {irpa_size} bytes")

        cleanup_files(new_mlir | new_irpa)
        temp_dir = tempfile.gettempdir()
        vmfb_after = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.vmfb")))
        cleanup_files(vmfb_after)

        print("PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_mixed_initializers():
    """Model with both small and large constants."""
    print("\n=== test_mixed_initializers ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    # Small: 1x16 float32 = 64 bytes (inline, broadcasts to [16,16]).
    # Large: 16x16 float32 = 1024 bytes (parameter).
    # Graph: C = (A + D_small) + D_large
    shape = [16, 16]
    a = np.random.rand(*shape).astype(np.float32)
    b_small = np.random.rand(1, 16).astype(np.float32)
    b_large = np.random.rand(*shape).astype(np.float32)

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, shape)
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, shape)

    const_small = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D_small"],
        value=helper.make_tensor(
            name="small_const",
            data_type=TensorProto.FLOAT,
            dims=[1, 16],
            vals=b_small.flatten().tolist(),
        ),
    )
    const_large = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D_large"],
        value=helper.make_tensor(
            name="large_const",
            data_type=TensorProto.FLOAT,
            dims=shape,
            vals=b_large.flatten().tolist(),
        ),
    )

    add1_out = helper.make_tensor_value_info("T", TensorProto.FLOAT, shape)
    add1 = helper.make_node("Add", inputs=["A", "D_small"], outputs=["T"])
    add2 = helper.make_node("Add", inputs=["T", "D_large"], outputs=["C"])

    graph = helper.make_graph(
        [add1, add2, const_small, const_large],
        "test_graph",
        [input_a],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    mlir_before, irpa_before = get_iree_files()
    model_path = test_utils.save_model(model)

    try:
        session = test_utils.create_session(
            model_path, device, {"target_arch": "host", "save_intermediates": "1"}
        )
        result = session.run(None, {"A": a})[0]
        expected = (a + b_small) + b_large

        if not np.allclose(result, expected, rtol=1e-5, atol=1e-5):
            print(f"FAIL: Values mismatch")
            return False
        print("  Inference result correct")

        # Check MLIR content.
        mlir_after, irpa_after = get_iree_files()
        new_mlir = mlir_after - mlir_before
        new_irpa = irpa_after - irpa_before

        if not new_mlir:
            print("FAIL: No MLIR file saved")
            return False

        mlir_content = open(list(new_mlir)[0]).read()

        if "dense_resource<" not in mlir_content:
            print("FAIL: MLIR should contain dense_resource< for small init")
            return False
        if 'flow.parameter.named<"model"::' not in mlir_content:
            print("FAIL: MLIR should contain flow.parameter.named for large init")
            return False
        if "dialect_resources" not in mlir_content:
            print("FAIL: MLIR should contain dialect_resources section")
            return False
        print("  MLIR content verified (both inline and parameter)")

        # IRPA should exist and have content.
        if not new_irpa:
            print("FAIL: No IRPA file was created")
            return False
        irpa_size = os.path.getsize(list(new_irpa)[0])
        if irpa_size == 0:
            print("FAIL: IRPA file should not be empty")
            return False
        print(f"  IRPA file size: {irpa_size} bytes")

        cleanup_files(new_mlir | new_irpa)
        temp_dir = tempfile.gettempdir()
        vmfb_after = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.vmfb")))
        cleanup_files(vmfb_after)

        print("PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_small_initializer_no_save():
    """Small constant without save_intermediates — verify inference works."""
    print("\n=== test_small_initializer_no_save ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[10, 20], [30, 40]], dtype=np.float32)

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=[2, 2],
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "D"], outputs=["C"])
    graph = helper.make_graph(
        [add_node, constant_node], "test_graph", [input_a], [output]
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    model_path = test_utils.save_model(model)
    try:
        session = test_utils.create_session(model_path, device, {"target_arch": "host"})
        result = session.run(None, {"A": a})[0]
        expected = a + b

        if not np.allclose(result, expected, rtol=1e-5, atol=1e-5):
            print(f"FAIL: Values mismatch\n  Expected: {expected}\n  Got: {result}")
            return False

        print("  Inference result correct")
        print("PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_large_initializer_no_save():
    """Large constant without save_intermediates — verify inference works."""
    print("\n=== test_large_initializer_no_save ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    a = np.random.rand(64, 64).astype(np.float32)
    b = np.random.rand(64, 64).astype(np.float32)

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [64, 64])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [64, 64])
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=[64, 64],
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "D"], outputs=["C"])
    graph = helper.make_graph(
        [add_node, constant_node], "test_graph", [input_a], [output]
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    model_path = test_utils.save_model(model)
    try:
        session = test_utils.create_session(model_path, device, {"target_arch": "host"})
        result = session.run(None, {"A": a})[0]
        expected = a + b

        if not np.allclose(result, expected, rtol=1e-5, atol=1e-5):
            print(f"FAIL: Values mismatch")
            return False

        print("  Inference result correct")
        print("PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def main():
    """Run all initializer tests."""
    print("Testing initializer handling (inline vs IRPA parameter)")
    print("=" * 60)

    test_utils.register_ep()

    results = []
    results.append(("small_inline", test_small_initializer_inline()))
    results.append(("large_parameter", test_large_initializer_parameter()))
    results.append(("mixed", test_mixed_initializers()))
    results.append(("small_no_save", test_small_initializer_no_save()))
    results.append(("large_no_save", test_large_initializer_no_save()))

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n=== All tests PASSED ===")
        return 0
    else:
        print("\n=== Some tests FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())

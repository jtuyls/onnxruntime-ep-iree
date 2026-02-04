#!/usr/bin/env python3
"""Test that intermediate file saving and cleanup works correctly."""

import glob
import os
import pathlib
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
import iree_onnx_ep


def get_iree_temp_files():
    """Return list of IREE temp files (mlir and vmfb) in the temp directory."""
    temp_dir = tempfile.gettempdir()
    mlir_files = glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir"))
    vmfb_files = glob.glob(os.path.join(temp_dir, "iree_ep_*.vmfb"))
    return mlir_files, vmfb_files


def cleanup_iree_temp_files():
    """Remove any leftover IREE temp files from previous runs."""
    mlir_files, vmfb_files = get_iree_temp_files()
    for f in mlir_files + vmfb_files:
        try:
            os.remove(f)
        except OSError:
            pass


def create_simple_model():
    """Create a simple Add model for testing."""
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[10, 20], [30, 40]], dtype=np.float32)

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["B"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=[2, 2],
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])

    graph = helper.make_graph(
        [add_node, constant_node], "test_graph", [input_a], [output]
    )
    model = helper.make_model(
        graph, producer_name="iree_test", opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model, a, b


def get_iree_device():
    """Get the IREE device (CPU)."""
    ep_devices = ort.get_ep_devices()
    for dev in ep_devices:
        if dev.ep_name == "IREE" and dev.device.type == ort.OrtHardwareDeviceType.CPU:
            return dev
    return None


def run_inference(model_path, iree_device, save_intermediates):
    """Run inference with the given save_intermediates setting."""
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0

    provider_options = {
        "device": "local-task",
        "opt_level": "O0",
    }
    if save_intermediates:
        provider_options["save_intermediates"] = "1"

    sess_options.add_provider_for_devices([iree_device], provider_options)

    session = ort.InferenceSession(model_path, sess_options=sess_options)

    # Run inference to ensure compilation happens.
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    session.run(None, {"A": a})

    # Explicitly delete the session to trigger cleanup.
    del session


def test_save_intermediates_enabled():
    """Test that intermediate files are saved when save_intermediates=1."""
    print("\n=== Testing save_intermediates=1 (files should be kept) ===")

    iree_device = get_iree_device()
    if not iree_device:
        print("ERROR: IREE EP not found")
        return False

    # Clean up any leftover files from previous runs.
    cleanup_iree_temp_files()

    # Get baseline file counts.
    mlir_before, vmfb_before = get_iree_temp_files()

    model, _, _ = create_simple_model()

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = f.name
        onnx.save(model, model_path)

    try:
        run_inference(model_path, iree_device, save_intermediates=True)

        # Check that intermediate files were created and kept.
        mlir_after, vmfb_after = get_iree_temp_files()

        new_mlir = set(mlir_after) - set(mlir_before)
        new_vmfb = set(vmfb_after) - set(vmfb_before)

        if not new_mlir:
            print("FAIL: No MLIR file was saved")
            return False
        if not new_vmfb:
            print("FAIL: No VMFB file was saved")
            return False

        print(f"  MLIR file saved: {list(new_mlir)[0]}")
        print(f"  VMFB file saved: {list(new_vmfb)[0]}")

        # Verify files have content.
        for f in new_mlir:
            size = os.path.getsize(f)
            if size == 0:
                print(f"FAIL: MLIR file is empty: {f}")
                return False
            print(f"  MLIR file size: {size} bytes")

        for f in new_vmfb:
            size = os.path.getsize(f)
            if size == 0:
                print(f"FAIL: VMFB file is empty: {f}")
                return False
            print(f"  VMFB file size: {size} bytes")

        # Clean up the saved files.
        for f in list(new_mlir) + list(new_vmfb):
            os.remove(f)

        print("PASS: Intermediate files were saved correctly")
        return True

    finally:
        pathlib.Path(model_path).unlink()


def test_save_intermediates_disabled():
    """Test that intermediate files are cleaned up when save_intermediates is not set."""
    print("\n=== Testing save_intermediates=0 (files should be cleaned up) ===")

    iree_device = get_iree_device()
    if not iree_device:
        print("ERROR: IREE EP not found")
        return False

    # Clean up any leftover files from previous runs.
    cleanup_iree_temp_files()

    # Get baseline file counts.
    mlir_before, vmfb_before = get_iree_temp_files()

    model, _, _ = create_simple_model()

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = f.name
        onnx.save(model, model_path)

    try:
        run_inference(model_path, iree_device, save_intermediates=False)

        # Check that intermediate files were cleaned up.
        mlir_after, vmfb_after = get_iree_temp_files()

        new_mlir = set(mlir_after) - set(mlir_before)
        new_vmfb = set(vmfb_after) - set(vmfb_before)

        if new_mlir:
            print(f"FAIL: MLIR file was not cleaned up: {list(new_mlir)}")
            # Clean up for next test.
            for f in new_mlir:
                os.remove(f)
            return False

        if new_vmfb:
            print(f"FAIL: VMFB file was not cleaned up: {list(new_vmfb)}")
            # Clean up for next test.
            for f in new_vmfb:
                os.remove(f)
            return False

        print("PASS: Intermediate files were cleaned up correctly")
        return True

    finally:
        pathlib.Path(model_path).unlink()


def main():
    """Run all tests."""
    print("Testing intermediate file handling")
    print("=" * 50)

    ep_lib_path = iree_onnx_ep.get_library_path()
    ort.register_execution_provider_library(iree_onnx_ep.get_ep_name(), ep_lib_path)

    results = []

    # Test with save_intermediates enabled.
    results.append(("save_intermediates=1", test_save_intermediates_enabled()))

    # Test with save_intermediates disabled.
    results.append(("save_intermediates=0", test_save_intermediates_disabled()))

    print("\n" + "=" * 50)
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

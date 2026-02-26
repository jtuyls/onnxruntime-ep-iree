"""Test EPContext caching for compiled artifacts.

Tests three model types:
- Small: weights inlined in MLIR, no IRPA generated
- Large: weights externalized to IRPA via parameter archive
- External: ONNX model with external data files, verifying that EPContext
  cache is self-contained (works after removing original external data files)
"""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper
from onnx.numpy_helper import from_array

np.random.seed(42)


def _create_matmul_add_model(model_dir, weight_size):
    """Create a MatMul + Add ONNX model with inline weights."""
    model_path = os.path.join(model_dir, "model.onnx")
    W = np.random.randn(weight_size, weight_size).astype(np.float32)
    B = np.random.randn(weight_size).astype(np.float32)

    X_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, weight_size])
    Y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, weight_size])
    W_init = from_array(W, name="W")
    B_init = from_array(B, name="B")

    matmul = helper.make_node("MatMul", ["X", "W"], ["XW"])
    add = helper.make_node("Add", ["XW", "B"], ["Y"])
    graph = helper.make_graph(
        [matmul, add], "test", [X_info], [Y_info], [W_init, B_init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, model_path)
    return model_path, W, B


def _create_matmul_add_model_external(model_dir, weight_size):
    """Create a MatMul + Add ONNX model with external data files."""
    model_path = os.path.join(model_dir, "model.onnx")
    W = np.random.randn(weight_size, weight_size).astype(np.float32)
    B = np.random.randn(weight_size).astype(np.float32)

    X_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, weight_size])
    Y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, weight_size])
    W_init = from_array(W, name="W")
    B_init = from_array(B, name="B")

    matmul = helper.make_node("MatMul", ["X", "W"], ["XW"])
    add = helper.make_node("Add", ["XW", "B"], ["Y"])
    graph = helper.make_graph(
        [matmul, add], "test", [X_info], [Y_info], [W_init, B_init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.save(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
    )
    return model_path, W, B


def _create_session(model_path, device, target_arch, extra_config=None):
    """Create an ORT session with the IREE EP."""
    so = ort.SessionOptions()
    for k, v in (extra_config or {}).items():
        so.add_session_config_entry(k, v)
    so.add_provider_for_devices([device], {"target_arch": target_arch})
    return ort.InferenceSession(model_path, sess_options=so)


def test_ep_context_small(iree_device):
    """EPContext round-trip with small model (no IRPA, weights inlined)."""
    weight_size = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, W, B = _create_matmul_add_model(tmpdir, weight_size)
        X = np.random.randn(2, weight_size).astype(np.float32)
        expected = X @ W + B

        # Generate EPContext cache.
        ctx_path = os.path.join(tmpdir, "model_ctx.onnx")
        sess = _create_session(
            model_path,
            iree_device,
            "host",
            {"ep.context_enable": "1", "ep.context_file_path": ctx_path},
        )
        result = sess.run(None, {"X": X})[0]
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
        del sess

        # Verify: no IRPA for small models (weights inlined in MLIR).
        assert os.path.exists(ctx_path)
        assert os.path.exists(os.path.join(tmpdir, "model_ctx.vmfb"))
        assert not os.path.exists(os.path.join(tmpdir, "model_ctx.irpa"))

        # Load EPContext cache and verify inference.
        sess = _create_session(ctx_path, iree_device, "host")
        result = sess.run(None, {"X": X})[0]
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
        del sess


def test_ep_context_large(iree_device):
    """EPContext round-trip with large model (weights archived to IRPA)."""
    weight_size = 256
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, W, B = _create_matmul_add_model(tmpdir, weight_size)
        X = np.random.randn(2, weight_size).astype(np.float32)
        expected = X @ W + B

        # Generate EPContext cache.
        ctx_path = os.path.join(tmpdir, "model_ctx.onnx")
        sess = _create_session(
            model_path,
            iree_device,
            "host",
            {"ep.context_enable": "1", "ep.context_file_path": ctx_path},
        )
        result = sess.run(None, {"X": X})[0]
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
        del sess

        # Verify: IRPA should exist for large models.
        assert os.path.exists(ctx_path)
        assert os.path.exists(os.path.join(tmpdir, "model_ctx.vmfb"))
        irpa_path = os.path.join(tmpdir, "model_ctx.irpa")
        assert os.path.exists(irpa_path)
        assert os.path.getsize(irpa_path) > 0

        # Load EPContext cache and verify inference.
        sess = _create_session(ctx_path, iree_device, "host")
        result = sess.run(None, {"X": X})[0]
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
        del sess


def test_ep_context_external_data(iree_device):
    """EPContext round-trip with external data files.

    Verifies that the IRPA cache is self-contained: after generating the
    EPContext cache, the original model and external data files are removed,
    and inference from the cached _ctx.onnx still produces correct results.
    """
    weight_size = 256
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, W, B = _create_matmul_add_model_external(tmpdir, weight_size)
        ext_data_path = os.path.join(tmpdir, "model.data")
        assert os.path.exists(ext_data_path)

        X = np.random.randn(2, weight_size).astype(np.float32)
        expected = X @ W + B

        # Generate EPContext cache.
        ctx_path = os.path.join(tmpdir, "model_ctx.onnx")
        sess = _create_session(
            model_path,
            iree_device,
            "host",
            {"ep.context_enable": "1", "ep.context_file_path": ctx_path},
        )
        result = sess.run(None, {"X": X})[0]
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
        del sess

        # Verify cache files.
        assert os.path.exists(ctx_path)
        assert os.path.exists(os.path.join(tmpdir, "model_ctx.vmfb"))
        irpa_path = os.path.join(tmpdir, "model_ctx.irpa")
        assert os.path.exists(irpa_path)
        assert os.path.getsize(irpa_path) > 0

        # Remove original model files — cache must be self-contained.
        os.remove(model_path)
        os.remove(ext_data_path)
        assert not os.path.exists(model_path)
        assert not os.path.exists(ext_data_path)

        # Load EPContext cache without original files and verify inference.
        sess = _create_session(ctx_path, iree_device, "host")
        result = sess.run(None, {"X": X})[0]
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
        del sess

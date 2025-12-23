// mlir_gen.h - MLIR text generation from OrtGraph.
//
// Generates torch-mlir ONNX dialect MLIR text directly from an OrtGraph,
// bypassing the need for ONNX protobuf dependencies.
//
// Note that we make a conscious decision to avoid using ONNX protobuf here.
// There are some utilities to convert OrtGraph to ONNX protobuf implemented in
// https://github.com/microsoft/onnxruntime/pull/25292, but even they recommend
// that EP's should directly convert OrtGraph to their own internal
// representation. We also want to be light on dependencies and don't want to
// take a dependency on protobuf and onnx. Since we don't use ONNX protobuf, we
// have to directly emit torch onnx dialect instead of using torch-mlir's onnx
// importer.

#ifndef IREE_ONNX_EP_SRC_MLIR_GEN_H_
#define IREE_ONNX_EP_SRC_MLIR_GEN_H_

#include <string>

#include "ort_import.h"

namespace iree_onnx_ep {

// Generates MLIR text from an OrtGraph and writes it to the specified file.
// Returns nullptr on success, or an OrtStatus with error details on failure.
OrtStatus* GenerateMlir(const Ort::ConstGraph& graph, const OrtApi& ort_api,
                        const std::string& output_path);

}  // namespace iree_onnx_ep

#endif  // IREE_ONNX_EP_SRC_MLIR_GEN_H_

//===- mlir_gen.cc --------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// MLIR text generation from OrtGraph.
//
//===----------------------------------------------------------------------===//

#include "mlir_gen.h"

#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "iree/io/file_handle.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree_ort_utils.h"

namespace onnxruntime::iree {
namespace {

// Initializers smaller than this are inlined via dense<> DenseElementsAttr.
// Larger ones become IREE parameters backed by an IRPA archive.
constexpr size_t kMaxInlineInitializerSize = 256;

// Encodes raw bytes as a hex string: "0xAABBCC...".
std::string HexEncode(const uint8_t* data, size_t size) {
  constexpr char hex_chars[] = "0123456789abcdef";
  std::string result;
  result.reserve(2 + size * 2);
  result = "0x";
  for (size_t i = 0; i < size; ++i) {
    result += hex_chars[(data[i] >> 4) & 0xF];
    result += hex_chars[data[i] & 0xF];
  }
  return result;
}

// Tracks a large initializer that will become an IREE parameter.
struct ParameterInitializer {
  std::string sanitized_name;
  size_t initializer_index;  // Index into initializers_ vector.
};

// Callback for iree_io_build_parameter_archive to create the IRPA file.
iree_status_t IrpaFileOpenCallback(void* user_data, iree_io_physical_offset_t,
                                   iree_io_physical_size_t,
                                   iree_io_file_handle_t** out_file_handle) {
  auto* path = static_cast<const std::string*>(user_data);
  return iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_OVERWRITE,
      iree_make_string_view(path->data(), path->size()),
      iree_allocator_system(), out_file_handle);
}

// Sanitizes an ONNX name to be a valid MLIR SSA identifier.
// MLIR identifiers must match [a-zA-Z_][a-zA-Z0-9_$]*.
std::string SanitizeName(const std::string& name) {
  assert(!name.empty() && "Unexpected empty name");
  std::string result;
  result.reserve(name.size());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$') {
      result += c;
    } else {
      result += '_';
    }
  }
  // Ensure starts with letter or underscore.
  if (!result.empty() && std::isdigit(static_cast<unsigned char>(result[0]))) {
    result = "_" + result;
  }
  return result.empty() ? "_unnamed" : result;
}

// Joins a vector of strings with a separator.
std::string Join(const std::vector<std::string>& parts,
                 const std::string& sep) {
  std::ostringstream ss;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      ss << sep;
    }
    ss << parts[i];
  }
  return ss.str();
}

// Returns MLIR element type string for an ONNX tensor element type.
// If signless is true, returns signless types (i64) for all integers.
// Otherwise returns signed (si64) or unsigned (ui64) types for torch dialect.
std::string GetElementType(ONNXTensorElementDataType dtype,
                           bool signless = false) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "f32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "f64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "f16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bf16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return signless ? "i8" : "si8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return signless ? "i16" : "si16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return signless ? "i32" : "si32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return signless ? "i64" : "si64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return signless ? "i8" : "ui8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return signless ? "i16" : "ui16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return signless ? "i32" : "ui32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return signless ? "i64" : "ui64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "i1";
    default:
      return "NYI";
  }
}

// Formats a tensor type as !torch.vtensor<[dims],dtype>.
std::string FormatTensorType(const Ort::ConstTypeInfo& type_info) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return "NYI";  // NYI: non-tensor types.
  }

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  auto dtype = tensor_info.GetElementType();

  std::ostringstream ss;
  ss << "!torch.vtensor<[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    if (shape[i] < 0) {
      // TODO: Ensure that dynamic dimensions are actually represented as -1. I
      // checked and they seem to but an example to check would be good.
      ss << "?";
    } else {
      ss << shape[i];
    }
  }
  ss << "]," << GetElementType(dtype) << ">";
  return ss.str();
}

// Formats a tensor type as tensor<dimsxdtype> (standard MLIR format).
// Uses signless integer types as required by MLIR tensor dialect.
std::string FormatMlirTensorType(const Ort::ConstTypeInfo& type_info) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return "NYI";
  }

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  auto dtype = tensor_info.GetElementType();

  std::ostringstream ss;
  ss << "tensor<";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      ss << "?";
    } else {
      ss << shape[i];
    }
    ss << "x";
  }
  ss << GetElementType(dtype, /*signless=*/true) << ">";
  return ss.str();
}

// Tries to parse an entire string as a non-negative integer.
// Returns true and sets value on success. Returns false on any invalid input
// (empty, non-digit characters, overflow). Uses std::from_chars — no
// exceptions.
bool ParseUnsigned(std::string_view s, size_t& value) {
  if (s.empty()) return false;
  size_t parsed{};
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), parsed);
  if (ec != std::errc{} || ptr != s.data() + s.size()) return false;
  value = parsed;
  return true;
}

// Parses an input reference of the form "$N" where N is a non-negative integer.
// Returns true and sets input_idx = N. Returns false for literal values
// (e.g., "4") or malformed specs (e.g., "$abc", "$").
bool ParseInputRef(const std::string& spec, size_t& input_idx) {
  if (spec.size() < 2 || spec[0] != '$') return false;
  return ParseUnsigned(std::string_view(spec).substr(1), input_idx);
}

// MLIR generator class.
class MlirGenerator {
 public:
  MlirGenerator(const Ort::ConstGraph& graph, std::ostream& out,
                const std::string& irpa_path, TargetConfig target_config)
      : graph_(graph),
        out_(out),
        irpa_path_(irpa_path),
        target_config_(std::move(target_config)) {}

  OrtStatus* Generate() {
    CollectMetadata();
    BuildShapeRefinementMaps();
    OrtStatus* status = EmitModuleHeader();
    if (status) return status;
    EmitSymbolicShapeBindings();
    status = EmitFunctionBody();
    if (status) return status;
    EmitModuleFooter();
    return nullptr;
  }

  // Builds an IRPA parameter archive for large initializers and creates a
  // parameter provider. Call after Generate(). If no parameters are needed,
  // the output pointers remain null.
  OrtStatus* BuildParameterArchive(ParameterIndexPtr& out_index,
                                   ParameterProviderPtr& out_provider);

 private:
  void CollectMetadata() {
    // Get graph name.
    graph_name_ = SanitizeName(graph_.GetName());
    if (graph_name_.empty()) {
      graph_name_ = "main";
    }

    // Get IR version.
    ir_version_ = graph_.GetOnnxIRVersion();

    // Get opset version (find default domain).
    auto opsets = graph_.GetOperatorSets();
    for (const auto& opset : opsets) {
      if (opset.domain.empty() || opset.domain == "ai.onnx") {
        opset_version_ = opset.version;
        break;
      }
    }

    // Collect inputs (excluding initializers).
    auto inputs = graph_.GetInputs();
    auto initializers = graph_.GetInitializers();

    // Build set of initializer names.
    std::unordered_set<std::string> init_names;
    for (const auto& init : initializers) {
      init_names.insert(init.GetName());
    }

    // Graph inputs are those not in initializers.
    for (const auto& input : inputs) {
      std::string name = input.GetName();
      if (init_names.find(name) == init_names.end()) {
        graph_inputs_.push_back(input);
      }
    }

    // Graph outputs.
    graph_outputs_ = graph_.GetOutputs();

    // Initializers.
    initializers_ = initializers;
  }

  OrtStatus* EmitModuleHeader() {
    // Build function arguments.
    std::ostringstream args;
    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (i > 0) {
        args << ", ";
      }
      std::string name = SanitizeName(graph_inputs_[i].GetName());
      std::string type = FormatTensorType(graph_inputs_[i].TypeInfo());
      args << "%" << name << ": " << type;
    }

    // Build return types (use refined types if available).
    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_types << ", ";
      }
      std::string out_name = graph_outputs_[i].GetName();
      ret_types << GetVtensorType(out_name, graph_outputs_[i].TypeInfo());
    }

    // Define the #executable_target alias for hal.dispatch.extern objects()
    // clauses. This is harmless when no extern dispatches are present.
    // We intentionally do NOT set hal.device.targets on the module — device
    // targeting is handled by iree-compile CLI flags (--iree-hal-target-device,
    // --iree-hip-target, etc.).
    if (!target_config_.hal_backend.empty()) {
      out_ << "#executable_target = #hal.executable.target<\""
           << target_config_.hal_backend << "\", \""
           << target_config_.hal_format << "\", {target_arch = \""
           << target_config_.target_arch << "\"}>\n";
    }
    out_ << "module {\n";

    // Emit util.global declarations for stateful KV cache.
    // Pre-scan nodes for com.iree:LoadGlobal to discover globals.
    auto nodes = graph_.GetNodes();
    for (const auto& node : nodes) {
      if (node.GetDomain() == "com.iree" &&
          node.GetOperatorType() == "LoadGlobal") {
        // Get global name from attribute.
        std::string global_name;
        for (const auto& attr : node.GetAttributes()) {
          if (attr.GetName() == "global_name") {
            attr.GetValue(global_name);
          }
        }
        if (!global_name.empty() &&
            declared_globals_.find(global_name) == declared_globals_.end()) {
          // Get output type to determine the tensor shape.
          auto outputs = node.GetOutputs();
          if (outputs.size() > 0 && outputs[0]) {
            std::string tensor_type = GetVtensorType(
                outputs[0].GetName(), outputs[0].TypeInfo());
            // Convert vtensor type to builtin tensor type for the global.
            std::vector<int64_t> dims;
            std::string elem;
            std::string builtin_type;
            if (ParseVtensorType(tensor_type, dims, elem)) {
              std::ostringstream ss;
              ss << "tensor<";
              for (size_t d = 0; d < dims.size(); ++d) {
                if (dims[d] <= 0) ss << "?";
                else ss << dims[d];
                ss << "x";
              }
              ss << elem << ">";
              builtin_type = ss.str();
            } else {
              builtin_type = FormatMlirTensorType(outputs[0].TypeInfo());
            }
            out_ << std::format(
                "  util.global private mutable @{} = dense<0.0> : {}\n",
                global_name, builtin_type);
            declared_globals_[global_name] = builtin_type;
          }
        }
      }
    }

    constexpr std::string_view func_schema =
        R"(  func.func @{0}({1}) -> ({2})
      attributes {{
        torch.onnx_meta.ir_version = {3} : si64,
        torch.onnx_meta.opset_version = {4} : si64,
        torch.onnx_meta.producer_name = "onnxruntime-ep-iree",
        torch.onnx_meta.producer_version = ""
      }} {{
)";

    out_ << std::format(func_schema,
                        graph_name_,      // {0}
                        args.str(),       // {1}
                        ret_types.str(),  // {2}
                        ir_version_,      // {3}
                        opset_version_);  // {4}
    return nullptr;
  }

  // Emits torch.symbolic_int + torch.bind_symbolic_shape ops for input tensors
  // with dynamic dimensions when seq_len_divisor is configured. This tells
  // IREE's BindSymbolicShapes pass to insert util.assume.int<udiv=N> ops,
  // enabling better tiling/vectorization for dynamic shapes.
  void EmitSymbolicShapeBindings() {
    int divisor = target_config_.seq_len_divisor;
    if (divisor <= 0) return;

    // Find input tensors with dynamic dimensions and collect unique symbols.
    // We use one symbol per unique dynamic dimension position across all inputs.
    // For LLM models, there's typically just one dynamic dim: seq_len.
    int sym_count = 0;
    struct Binding {
      size_t input_idx;
      std::string name;
      std::string type;
      std::vector<int64_t> shape;
      std::vector<int> dyn_dim_symbols;  // symbol index per dim (-1 = static)
    };
    std::vector<Binding> bindings;

    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      auto type_info = graph_inputs_[i].TypeInfo();
      if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) continue;

      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();

      bool has_dynamic = false;
      for (auto dim : shape) {
        if (dim < 0) { has_dynamic = true; break; }
      }
      if (!has_dynamic) continue;

      Binding b;
      b.input_idx = i;
      b.name = SanitizeName(graph_inputs_[i].GetName());
      b.type = FormatTensorType(graph_inputs_[i].TypeInfo());
      b.shape = shape;
      for (auto dim : shape) {
        if (dim < 0) {
          // Assign a symbol. For now, all dynamic dims share symbol 0
          // (typical for LLMs where the only dynamic dim is seq_len).
          b.dyn_dim_symbols.push_back(0);
          sym_count = std::max(sym_count, 1);
        } else {
          b.dyn_dim_symbols.push_back(-1);
        }
      }
      bindings.push_back(std::move(b));
    }

    if (bindings.empty()) return;

    // Emit symbolic_int declarations.
    // max_val = reasonable upper bound / divisor (e.g., 4096/128 = 32).
    int max_sym_val = 4096 / divisor;
    for (int s = 0; s < sym_count; ++s) {
      out_ << std::format(
          "    %__sym_s{0} = torch.symbolic_int \"s{0}\" "
          "{{min_val = 1, max_val = {1}}} : !torch.int\n",
          s, max_sym_val);
    }

    // Emit bind_symbolic_shape for each tensor with dynamic dims.
    for (const auto& b : bindings) {
      // Collect unique symbol references used by this binding.
      std::vector<int> used_syms;
      for (int sym : b.dyn_dim_symbols) {
        if (sym >= 0 &&
            std::find(used_syms.begin(), used_syms.end(), sym) ==
                used_syms.end()) {
          used_syms.push_back(sym);
        }
      }

      // Build symbol list: [%__sym_s0, %__sym_s1, ...]
      std::ostringstream sym_refs;
      bool first = true;
      for (int s : used_syms) {
        if (!first) sym_refs << ", ";
        first = false;
        sym_refs << "%__sym_s" << s;
      }

      // Build affine map: ()[s0] -> (static_dim, s0 * divisor, ...)
      std::ostringstream map_params, map_results;
      first = true;
      for (int s : used_syms) {
        if (!first) map_params << ", ";
        first = false;
        map_params << "s" << s;
      }

      first = true;
      for (size_t d = 0; d < b.shape.size(); ++d) {
        if (!first) map_results << ", ";
        first = false;
        if (b.dyn_dim_symbols[d] >= 0) {
          map_results << "s" << b.dyn_dim_symbols[d] << " * " << divisor;
        } else {
          map_results << b.shape[d];
        }
      }

      out_ << std::format(
          "    torch.bind_symbolic_shape %{0}, [{1}], "
          "affine_map<()[{2}] -> ({3})> : {4}\n",
          b.name, sym_refs.str(), map_params.str(),
          map_results.str(), b.type);
    }
  }

  OrtStatus* EmitFunctionBody() {
    // Emit initializers as flow.tensor.constant ops.
    for (size_t i = 0; i < initializers_.size(); ++i) {
      EmitInitializer(initializers_[i], i);
    }

    // Emit nodes.
    auto nodes = graph_.GetNodes();
    for (const auto& node : nodes) {
      OrtStatus* status = EmitNode(node);
      if (status) return status;
    }

    // Emit return.
    EmitReturn();
    return nullptr;
  }

  // Emits an initializer as a flow.tensor.constant with a
  // torch_c.from_builtin_tensor cast. Small initializers use dense<> with
  // inline hex-encoded data. Large initializers use #flow.parameter.named
  // (data stored in IRPA archive).
  //
  // Output format (small):
  //   %__raw_name = flow.tensor.constant dense<"0x..."> : tensor<...>
  //   %name = torch_c.from_builtin_tensor %__raw_name : tensor<...>
  //       -> !torch.vtensor<[...],dtype>
  //
  // Output format (large):
  //   %__raw_name = flow.tensor.constant
  //       #flow.parameter.named<"model"::"name"> : tensor<...>
  //   %name = torch_c.from_builtin_tensor %__raw_name : tensor<...>
  //       -> !torch.vtensor<[...],dtype>
  void EmitInitializer(const Ort::ConstValueInfo& init, size_t init_index) {
    std::string name = SanitizeName(init.GetName());
    std::string vtensor_type = FormatTensorType(init.TypeInfo());
    std::string tensor_type = FormatMlirTensorType(init.TypeInfo());

    auto tensor_info = init.TypeInfo().GetTensorTypeAndShapeInfo();
    size_t byte_size = tensor_info.GetElementCount() *
                       OnnxElementTypeSize(tensor_info.GetElementType());

    if (byte_size <= kMaxInlineInitializerSize) {
      // Small: inline with dense<> DenseElementsAttr.
      Ort::ConstValue tensor_value{nullptr};
      auto status = init.GetInitializer(tensor_value);
      if (!status.IsOK()) {
        return;
      }
      const auto* data =
          static_cast<const uint8_t*>(tensor_value.GetTensorRawData());
      std::string hex = HexEncode(data, tensor_value.GetTensorSizeInBytes());

      constexpr std::string_view schema =
          R"(    %__raw_{0} = flow.tensor.constant dense<"{3}"> : {1}
    %{0} = torch_c.from_builtin_tensor %__raw_{0} : {1} -> {2}
)";
      out_ << std::format(schema, name, tensor_type, vtensor_type, hex);
    } else {
      // Large: parameter reference. Data stored in IRPA archive.
      constexpr std::string_view schema =
          R"(    %__raw_{0} = flow.tensor.constant #flow.parameter.named<"model"::"{0}"> : {1}
    %{0} = torch_c.from_builtin_tensor %__raw_{0} : {1} -> {2}
)";
      out_ << std::format(schema, name, tensor_type, vtensor_type);
      parameter_initializers_.push_back({name, init_index});
    }
  }

  OrtStatus* EmitNode(const Ort::ConstNode& node) {
    if (node.GetDomain() == "com.iree") {
      if (node.GetOperatorType() == "ExternDispatch") {
        return EmitExternDispatch(node, extern_id_++);
      }
      if (node.GetOperatorType() == "LoadGlobal") {
        return EmitLoadGlobal(node);
      }
      if (node.GetOperatorType() == "StoreGlobal") {
        return EmitStoreGlobal(node);
      }
    }

    std::string op_type = node.GetOperatorType();
    auto inputs = node.GetInputs();
    auto outputs = node.GetOutputs();
    auto attrs = node.GetAttributes();

    // Track Shape ops for later Gather(Shape(tensor), idx) → size.int rewrite.
    if (op_type == "Shape" && inputs.size() >= 1 && inputs[0] &&
        outputs.size() >= 1 && outputs[0]) {
      std::string shape_out = outputs[0].GetName();
      std::string shape_src = inputs[0].GetName();
      if (!shape_out.empty() && !shape_src.empty()) {
        shape_source_map_[shape_out] = {
            shape_src, GetVtensorType(shape_src, inputs[0].TypeInfo())};
      }
    }

    // Rewrite Gather(Shape(tensor), const_idx) → torch.aten.size.int +
    // torch.prim.NumToTensor.Scalar. This gives IREE a direct reference to the
    // input tensor's dimension, enabling workgroup count resolution.
    if (op_type == "Gather" && inputs.size() >= 2 && inputs[0] && inputs[1] &&
        outputs.size() >= 1 && outputs[0]) {
      std::string shape_name = inputs[0].GetName();
      std::string idx_name = inputs[1].GetName();
      auto shape_it = shape_source_map_.find(shape_name);
      auto idx_it = known_int64_vectors_.find(idx_name);
      if (shape_it != shape_source_map_.end() && idx_it != known_int64_vectors_.end() &&
          idx_it->second.size() == 1) {
        // This is Gather(Shape(tensor), const_idx) — emit size.int instead.
        const auto& [src_name, src_type] = shape_it->second;
        int64_t dim_idx = idx_it->second[0];
        std::string out_name = SanitizeName(outputs[0].GetName());
        std::string int_name = out_name + "_int";
        out_ << std::format(
            R"(    %{0} = torch.constant.int {1}
    %{2} = torch.aten.size.int %{3}, %{0} : {4}, !torch.int -> !torch.int
    %{5} = torch.prim.NumToTensor.Scalar %{2} : !torch.int -> !torch.vtensor<[],si64>
)",
            out_name + "_dim_idx",  // {0} constant dim index
            dim_idx,                // {1} dim value
            int_name,               // {2} int result name
            SanitizeName(src_name), // {3} source tensor SSA
            src_type,               // {4} source tensor type
            out_name);              // {5} output scalar tensor
        // Track the size.int SSA name for later use by ConstantOfShape.
        size_int_ssa_map_[outputs[0].GetName()] = int_name;
        return nullptr;
      }
    }

    // Propagate size_int_ssa_map_ through Unsqueeze (tensor wrapping).
    if (op_type == "Unsqueeze" && inputs.size() >= 1 && inputs[0] &&
        outputs.size() >= 1 && outputs[0]) {
      auto sit = size_int_ssa_map_.find(inputs[0].GetName());
      if (sit != size_int_ssa_map_.end()) {
        size_int_ssa_map_[outputs[0].GetName()] = sit->second;
      }
    }

    // Rewrite ConstantOfShape to torch.aten.full when shape elements can be
    // resolved to size.int values or constants. This gives IREE direct
    // traceable references to input tensor dimensions.
    if (op_type == "ConstantOfShape" && inputs.size() >= 1 && inputs[0] &&
        outputs.size() >= 1 && outputs[0]) {
      std::string shape_name = inputs[0].GetName();
      auto prod_it = producer_info_.find(shape_name);
      if (prod_it != producer_info_.end() && prod_it->second.op_type == "Concat") {
        const auto& concat_inputs = prod_it->second.input_names;
        std::vector<std::string> dim_ssa_names;
        bool can_rewrite = true;
        int counter = 0;
        for (const auto& inp_name : concat_inputs) {
          // Check if this concat input traces to a size.int.
          auto sit = size_int_ssa_map_.find(inp_name);
          if (sit != size_int_ssa_map_.end()) {
            dim_ssa_names.push_back(sit->second);
          } else {
            // Check if it's a known constant.
            auto cit = known_int64_vectors_.find(inp_name);
            if (cit != known_int64_vectors_.end() && cit->second.size() == 1) {
              std::string cname = SanitizeName(outputs[0].GetName()) +
                                  "_dim_c" + std::to_string(counter++);
              out_ << std::format("    %{} = torch.constant.int {}\n",
                                  cname, cit->second[0]);
              dim_ssa_names.push_back(cname);
            } else {
              can_rewrite = false;
              break;
            }
          }
        }
        if (can_rewrite && !dim_ssa_names.empty()) {
          std::string out_name = SanitizeName(outputs[0].GetName());
          // Use refined type if available, otherwise ORT type.
          auto ref_it = refined_types_.find(outputs[0].GetName());
          std::string out_type = (ref_it != refined_types_.end())
              ? ref_it->second
              : FormatTensorType(outputs[0].TypeInfo());

          // Build list construct.
          std::ostringstream dim_refs, dim_types;
          for (size_t i = 0; i < dim_ssa_names.size(); ++i) {
            if (i > 0) { dim_refs << ", "; dim_types << ", "; }
            dim_refs << "%" << dim_ssa_names[i];
            dim_types << "!torch.int";
          }
          std::string list_name = out_name + "_shape_list";
          out_ << std::format(
              "    %{0} = torch.prim.ListConstruct {1} : ({2}) -> !torch.list<int>\n",
              list_name, dim_refs.str(), dim_types.str());

          // Extract fill value from the value attribute (default 0).
          // The attr is dense<"0x0000"> : tensor<1xf16> or similar.
          std::string fill_name = out_name + "_fill";
          out_ << std::format("    %{} = torch.constant.float 0.000000e+00\n",
                              fill_name);
          std::string none_name = out_name + "_none";
          out_ << std::format("    %{} = torch.constant.none\n", none_name);

          // Emit torch.aten.full.
          out_ << std::format(
              "    %{0} = torch.aten.full %{1}, %{2}, %{3}, %{3}, %{3}, %{3} : "
              "!torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> {4}\n",
              out_name, list_name, fill_name, none_name, out_type);
          return nullptr;
        }
      }
    }

    // Handle DequantizeLinear with block_size by decomposing to arithmetic ops.
    // torch-mlir's ONNX conversion doesn't support the block_size attribute,
    // so we decompose into: reshape → cast → sub(zero) → mul(scale) → reshape.
    // The resulting arith pattern (extui → uitofp → subf → mulf) matches IREE's
    // FuseDequantizationMatmul pass for optimized quantized inference.
    if (op_type == "DequantizeLinear") {
      int64_t block_size = 0;
      for (const auto& a : attrs) {
        if (a.GetName() == "block_size") a.GetValue(block_size);
      }
      if (block_size > 0) {
        return EmitBlockedDequantizeLinear(node);
      }
    }

    // Fused Softmax→Cast→TopK → extern dispatch for MoE routing.
    // Replaces the decomposed pattern with a single GPU kernel that directly
    // outputs top-k values and indices, eliminating ~216 slow_memcpy dispatches
    // from IREE's sort-based TopK decomposition.
    if (op_type == "TopK" && !target_config_.hal_backend.empty() &&
        inputs.size() >= 1 && inputs[0] && outputs.size() >= 2 &&
        outputs[0] && outputs[1]) {
      std::string topk_input_name = inputs[0].GetName();
      auto cast_it = producer_info_.find(topk_input_name);
      // Check: TopK input ← Cast ← Softmax
      if (cast_it != producer_info_.end() &&
          cast_it->second.op_type == "Cast" &&
          !cast_it->second.input_names.empty()) {
        std::string cast_input_name = cast_it->second.input_names[0];
        auto softmax_it = producer_info_.find(cast_input_name);
        if (softmax_it != producer_info_.end() &&
            softmax_it->second.op_type == "Softmax" &&
            !softmax_it->second.input_names.empty()) {
          // Found pattern: Softmax(x) → Cast(f16→f32) → TopK
          // x is the gate logits in f16 from the router MatMul.
          std::string gate_logits_name = softmax_it->second.input_names[0];

          // Get the gate logits type info from refined types.
          std::string gate_type;
          auto ref_it = refined_types_.find(gate_logits_name);
          if (ref_it != refined_types_.end()) {
            gate_type = ref_it->second;
          }

          // Parse gate logits shape: expect [?, num_experts]
          std::vector<int64_t> gate_dims;
          std::string gate_elem;
          if (!gate_type.empty() &&
              ParseVtensorType(gate_type, gate_dims, gate_elem) &&
              gate_dims.size() == 2 && gate_dims[1] > 0 &&
              gate_elem == "f16") {
            int64_t num_experts = gate_dims[1];

            // Parse TopK k value from input[1] (constant tensor).
            std::string k_name = inputs[1].GetName();
            auto k_it = known_int64_vectors_.find(k_name);
            if (k_it != known_int64_vectors_.end() &&
                k_it->second.size() == 1 && k_it->second[0] > 0) {
              int64_t top_k = k_it->second[0];

              // Build kernel object filename from target arch.
              std::string kernel_obj = "moe_topk_" +
                  target_config_.target_arch + ".co";

              // Output SSA names.
              std::string val_name = SanitizeName(outputs[0].GetName());
              std::string idx_name = SanitizeName(outputs[1].GetName());
              std::string prefix = std::format("__topk_{}", extern_id_++);

              // Bridge gate logits input to builtin tensor.
              std::string gate_ssa = SanitizeName(gate_logits_name);
              std::string dim0_str = gate_dims[0] > 0
                  ? std::to_string(gate_dims[0]) : "?";
              std::string gate_tensor_type = std::format(
                  "tensor<{}x{}xf16>", dim0_str, num_experts);
              out_ << std::format(
                  "    %{0}_in = torch_c.to_builtin_tensor %{1} : {2} -> {3}\n",
                  prefix, gate_ssa, gate_type, gate_tensor_type);

              // Get dynamic dim (N = number of tokens) via tensor.dim
              // on the bridged builtin tensor — gives index directly.
              out_ << std::format(
                  "    %{0}_c0 = arith.constant 0 : index\n"
                  "    %{0}_dim_n = tensor.dim %{0}_in, %{0}_c0 : {1}\n",
                  prefix, gate_tensor_type);

              // Push constants: N (i32), num_experts (i32), top_k (i32).
              out_ << std::format(
                  "    %{0}_pc_n = arith.index_cast %{0}_dim_n : index to i32\n"
                  "    %{0}_pc_e = arith.constant {1} : i32\n"
                  "    %{0}_pc_k = arith.constant {2} : i32\n",
                  prefix, num_experts, top_k);

              // Workload: ceil(N / 64) for 64 threads per workgroup.
              out_ << std::format(
                  "    %{0}_c64 = arith.constant 64 : index\n"
                  "    %{0}_c63 = arith.constant 63 : index\n"
                  "    %{0}_n_plus = arith.addi %{0}_dim_n, %{0}_c63 : index\n"
                  "    %{0}_wl = arith.divui %{0}_n_plus, %{0}_c64 : index\n",
                  prefix);

              // Output types — use static dim if known.
              std::string val_tensor = std::format(
                  "tensor<{}x{}xf32>", dim0_str, top_k);
              std::string idx_tensor = std::format(
                  "tensor<{}x{}xi64>", dim0_str, top_k);
              std::string val_vtensor = std::format(
                  "!torch.vtensor<[{},{}],f32>", dim0_str, top_k);
              std::string idx_vtensor = std::format(
                  "!torch.vtensor<[{},{}],si64>", dim0_str, top_k);

              // Dynamic dim annotations: only needed when dim0 is dynamic.
              bool dim0_dynamic = (gate_dims[0] <= 0);
              std::string dim_annot = dim0_dynamic
                  ? std::format("{{%{}_dim_n}}", prefix) : "";

              // Emit hal.dispatch.extern.
              out_ << std::format(
                  R"(    %{0}_out_v, %{0}_out_i = hal.dispatch.extern "moe_topk"[%{0}_wl](%{0}_pc_n, %{0}_pc_e, %{0}_pc_k, %{0}_in)
        : (i32, i32, i32, {1}{5}) -> ({2}{5}, {3}{5})
      count(%device: !hal.device, %workload: index) -> (index, index, index) {{
        %c1 = arith.constant 1 : index
        hal.return %workload, %c1, %c1 : index, index, index
      }}
      layout(#hal.pipeline.layout<constants = 3, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>,
        #hal.pipeline.binding<storage_buffer>
      ]>)
      objects({{
        #executable_target ordinal(0) = [
          #hal.executable.object<{{path = "{4}"}}>
        ]
      }})
      attributes {{workgroup_size = [64 : index, 1 : index, 1 : index]}}
)",
                  prefix,             // {0}
                  gate_tensor_type,   // {1} input type
                  val_tensor,         // {2} values output type
                  idx_tensor,         // {3} indices output type
                  kernel_obj,         // {4} kernel object path
                  dim_annot);         // {5} dynamic dim annotation

              // Bridge outputs back to torch types.
              out_ << std::format(
                  "    %{0} = torch_c.from_builtin_tensor %{1}_out_v : {2} -> {3}\n"
                  "    %{4} = torch_c.from_builtin_tensor %{1}_out_i : {5} -> {6}\n",
                  val_name, prefix, val_tensor, val_vtensor,
                  idx_name, idx_tensor, idx_vtensor);

              // Store refined types for downstream propagation.
              refined_types_[outputs[0].GetName()] = val_vtensor;
              refined_types_[outputs[1].GetName()] = idx_vtensor;
              return nullptr;
            }
          }
        }
      }
    }

    // ScatterElements → tensor.insert_slice for KV cache updates.
    // Detects the pattern:
    //   ScatterElements(data=past_kv, indices=Expand(Reshape(Cast(Add(
    //     Range(0,seq_len,1), offset)))), updates=new_kv, axis=2)
    // which is a contiguous slice write: past_kv[:,:,pos:pos+seq_len,:] = new_kv.
    // Emitting tensor.insert_slice instead eliminates the 4D index tensor
    // decomposition that IREE lowers to ~240 slow_memcpy dispatches.
    if (op_type == "ScatterElements" && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] &&
        outputs.size() >= 1 && outputs[0]) {
      // Check axis attribute.
      int64_t axis = 0;
      for (const auto& attr : attrs) {
        if (attr.GetName() == "axis") {
          attr.GetValue(axis);
          break;
        }
      }

      // Only handle axis=2 KV cache scatter pattern for now.
      if (axis == 2) {
        // Trace index chain backwards from indices to find:
        //   Expand ← (Reshape|Cast)* ← Add(Range, offset)
        std::string idx_name = inputs[1].GetName();
        auto expand_it = producer_info_.find(idx_name);
        if (expand_it != producer_info_.end() &&
            expand_it->second.op_type == "Expand" &&
            !expand_it->second.input_names.empty()) {
          // Walk backwards through Reshape/Cast to find the Add.
          std::string cur = expand_it->second.input_names[0];
          const ProducerInfo* add_info = nullptr;
          for (int depth = 0; depth < 5; ++depth) {
            auto it = producer_info_.find(cur);
            if (it == producer_info_.end() || it->second.input_names.empty())
              break;
            if (it->second.op_type == "Add") {
              add_info = &it->second;
              break;
            }
            if (it->second.op_type == "Reshape" ||
                it->second.op_type == "Cast" ||
                it->second.op_type == "Unsqueeze") {
              cur = it->second.input_names[0];
              continue;
            }
            break;  // Unexpected op — stop.
          }

          if (add_info && add_info->input_names.size() >= 2) {
            // Found: Add(Range, offset). Determine which input is Range
            // and which is the scalar offset (seq_position).
            std::string offset_name;
            for (const auto& add_inp : add_info->input_names) {
              auto range_it = producer_info_.find(add_inp);
              if (range_it != producer_info_.end() &&
                  range_it->second.op_type == "Range") {
                continue;  // Skip the Range input.
              }
              offset_name = add_inp;  // The other input is the offset.
            }

            if (!offset_name.empty()) {
                  // Pattern matched! Emit tensor.insert_slice.
                  std::string data_name = inputs[0].GetName();
                  std::string update_name = inputs[2].GetName();
                  std::string out_name = outputs[0].GetName();

                  std::string data_ssa = SanitizeName(data_name);
                  std::string update_ssa = SanitizeName(update_name);
                  std::string offset_ssa = SanitizeName(offset_name);
                  std::string out_ssa = SanitizeName(out_name);

                  // Get types.
                  std::string data_vtype =
                      GetVtensorType(data_name, inputs[0].TypeInfo());
                  std::string update_vtype =
                      GetVtensorType(update_name, inputs[2].TypeInfo());
                  auto update_info =
                      inputs[2].TypeInfo().GetTensorTypeAndShapeInfo();
                  auto data_info =
                      inputs[0].TypeInfo().GetTensorTypeAndShapeInfo();
                  auto update_shape = update_info.GetShape();
                  auto data_shape = data_info.GetShape();

                  // Get refined update type if available.
                  auto ref_it = refined_types_.find(update_name);
                  if (ref_it != refined_types_.end()) {
                    update_vtype = ref_it->second;
                  }
                  auto ref_data = refined_types_.find(data_name);
                  if (ref_data != refined_types_.end()) {
                    data_vtype = ref_data->second;
                  }

                  // Parse shapes.
                  std::vector<int64_t> data_dims, update_dims;
                  std::string data_elem, update_elem;
                  ParseVtensorType(data_vtype, data_dims, data_elem);
                  ParseVtensorType(update_vtype, update_dims, update_elem);

                  if (data_dims.size() == 4 && update_dims.size() == 4) {
                    // Build builtin tensor types from refined dims
                    // (ORT type info may have all-dynamic shapes).
                    std::string elem_type = GetElementType(
                        data_info.GetElementType(), /*signless=*/true);
                    auto fmt_ttype = [&](const std::vector<int64_t>& dims) {
                      std::ostringstream ss;
                      ss << "tensor<";
                      for (size_t i = 0; i < dims.size(); ++i) {
                        if (dims[i] < 0) ss << "?"; else ss << dims[i];
                        ss << "x";
                      }
                      ss << elem_type << ">";
                      return ss.str();
                    };
                    std::string data_ttype = fmt_ttype(data_dims);
                    std::string update_ttype = fmt_ttype(update_dims);

                    std::string prefix = std::format("__kv_update_{}", extern_id_++);

                    // Bridge data (past_kv) and updates (new_kv) to builtin.
                    out_ << std::format(
                        "    %{0}_data = torch_c.to_builtin_tensor %{1}"
                        " : {2} -> {3}\n",
                        prefix, data_ssa, data_vtype, data_ttype);
                    out_ << std::format(
                        "    %{0}_src = torch_c.to_builtin_tensor %{1}"
                        " : {2} -> {3}\n",
                        prefix, update_ssa, update_vtype, update_ttype);

                    // Extract offset as index.
                    // seq_position is a scalar i64 vtensor — extract and
                    // convert to index.
                    out_ << std::format(
                        "    %{0}_off_t = torch_c.to_builtin_tensor"
                        " %{1} : !torch.vtensor<[],si64> -> tensor<i64>\n"
                        "    %{0}_off_i64 = tensor.extract %{0}_off_t[]"
                        " : tensor<i64>\n"
                        "    %{0}_off = arith.index_cast %{0}_off_i64"
                        " : i64 to index\n",
                        prefix, offset_ssa);

                    // Get dynamic dim for seq_len (axis 2 of updates).
                    // Use tensor.dim on the builtin update tensor.
                    out_ << std::format(
                        "    %{0}_c2 = arith.constant 2 : index\n"
                        "    %{0}_seq_len = tensor.dim %{0}_src, %{0}_c2"
                        " : {1}\n",
                        prefix, update_ttype);

                    // Build static sizes for non-axis dims.
                    // data shape: [1, 16, 256, 128] (all static)
                    // update shape: [1, 16, ?, 128]
                    // insert_slice offsets: [0, 0, %off, 0]
                    // insert_slice sizes: [1, 16, %seq_len, 128]
                    // insert_slice strides: [1, 1, 1, 1]
                    std::string offsets = std::format(
                        "0, 0, %{}_off, 0", prefix);
                    // If seq_len dim (axis 2) is static, use the constant;
                    // otherwise use the dynamic SSA value.
                    std::string seq_len_str =
                        update_dims[2] > 0
                            ? std::to_string(update_dims[2])
                            : std::format("%{}_seq_len", prefix);
                    std::string sizes = std::format(
                        "{}, {}, {}, {}",
                        update_dims[0] > 0 ? std::to_string(update_dims[0]) : "1",
                        update_dims[1] > 0 ? std::to_string(update_dims[1]) : "16",
                        seq_len_str,
                        update_dims[3] > 0 ? std::to_string(update_dims[3]) : "128");

                    // Emit tensor.insert_slice.
                    out_ << std::format(
                        "    %{0}_result = tensor.insert_slice %{0}_src"
                        " into %{0}_data[{1}] [{2}] [1, 1, 1, 1]"
                        " : {3} into {4}\n",
                        prefix, offsets, sizes, update_ttype, data_ttype);

                    // Bridge result back to vtensor.
                    std::string out_vtype = data_vtype;
                    out_ << std::format(
                        "    %{0} = torch_c.from_builtin_tensor %{1}_result"
                        " : {2} -> {3}\n",
                        out_ssa, prefix, data_ttype, out_vtype);

                    // Store refined type.
                    refined_types_[out_name] = out_vtype;
                    return nullptr;
                  }
                }
            }
          }
        }
      }

    // Handle ReduceMean/ReduceSum with tensor axes input (opset 18+).
    // torch-mlir expects axes as an attribute (opset 17 format), so convert
    // the tensor input back to an attribute when the axes are known constants.
    if ((op_type == "ReduceMean" || op_type == "ReduceSum" ||
         op_type == "ReduceMax" || op_type == "ReduceMin") &&
        inputs.size() >= 2 && inputs[1] && !inputs[1].GetName().empty()) {
      auto axes_it = known_int64_vectors_.find(inputs[1].GetName());
      if (axes_it != known_int64_vectors_.end()) {
        return EmitReduceWithAxesAttr(node, op_type, axes_it->second);
      }
    }

    // For Transpose: emit torch.aten.permute instead of generic onnx.Transpose.
    // torch.aten.permute is better handled by IREE's fusion passes, allowing
    // the permutation to be folded into adjacent producer/consumer kernels
    // rather than materializing as a separate slow_memcpy dispatch.
    if (op_type == "Transpose" && inputs.size() >= 1 && inputs[0] &&
        outputs.size() >= 1 && outputs[0]) {
      // Extract perm attribute.
      std::vector<int64_t> perm;
      for (const auto& attr : attrs) {
        if (attr.GetName() == "perm" &&
            attr.GetType() == ORT_OP_ATTR_INTS) {
          attr.GetValueArray<int64_t>(perm);
          break;
        }
      }
      if (!perm.empty()) {
        std::string out_name = SanitizeName(outputs[0].GetName());
        std::string data_name = SanitizeName(inputs[0].GetName());
        std::string data_type = GetVtensorType(inputs[0].GetName(),
                                                inputs[0].TypeInfo());
        // Compute output type by permuting input dims.
        std::string in_type = data_type;
        std::vector<int64_t> in_dims;
        std::string elem_type;
        std::string out_type;
        if (ParseVtensorType(in_type, in_dims, elem_type) &&
            perm.size() == in_dims.size()) {
          std::vector<int64_t> out_dims(perm.size());
          for (size_t i = 0; i < perm.size(); ++i) {
            out_dims[i] = in_dims[perm[i]];
          }
          out_type = BuildVtensorType(out_dims, elem_type);
        } else {
          out_type = FormatTensorType(outputs[0].TypeInfo());
        }
        // Store refined type for downstream propagation.
        refined_types_[outputs[0].GetName()] = out_type;

        // Emit perm constants and ListConstruct.
        std::ostringstream dim_refs, dim_types;
        for (size_t i = 0; i < perm.size(); ++i) {
          std::string pname = out_name + "_p" + std::to_string(i);
          out_ << std::format("    %{} = torch.constant.int {}\n",
                              pname, perm[i]);
          if (i > 0) { dim_refs << ", "; dim_types << ", "; }
          dim_refs << "%" << pname;
          dim_types << "!torch.int";
        }
        std::string list_name = out_name + "_perm";
        out_ << std::format(
            "    %{0} = torch.prim.ListConstruct {1} : ({2}) -> !torch.list<int>\n",
            list_name, dim_refs.str(), dim_types.str());
        out_ << std::format(
            "    %{0} = torch.aten.permute %{1}, %{2} : {3}, !torch.list<int> -> {4}\n",
            out_name, data_name, list_name, data_type, out_type);
        return nullptr;
      }
    }

    // For Reshape, try to refine the output shape by tracing constant values
    // through the shape tensor (ORT loses this info through Concat chains).
    std::string refined_reshape_type;
    if (op_type == "Reshape") {
      refined_reshape_type = RefineReshapeOutputType(node);
    }

    // Build output SSA names and types.
    // First, collect output names for later refinement storage.
    std::vector<std::string> output_names;
    std::vector<std::string> output_type_strs;
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (!outputs[i]) { output_names.push_back(""); output_type_strs.push_back(""); continue; }
      std::string name = outputs[i].GetName();
      output_names.push_back(name);
      if (i == 0 && !refined_reshape_type.empty()) {
        output_type_strs.push_back(refined_reshape_type);
      } else {
        output_type_strs.push_back(FormatTensorType(outputs[i].TypeInfo()));
      }
    }

    // Propagate refined shapes through downstream ops.
    PropagateRefinedTypes(op_type, inputs, outputs, attrs, output_names,
                          output_type_strs);

    std::ostringstream out_names;
    std::ostringstream out_types;
    bool first_output = true;
    size_t valid_output_count = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (output_names[i].empty()) continue;
      if (!first_output) {
        out_names << ", ";
        out_types << ", ";
      }
      first_output = false;
      valid_output_count++;
      out_names << "%" << SanitizeName(output_names[i]);
      out_types << output_type_strs[i];
    }

    // For Reshape: if the shape tensor comes from a Concat, emit
    // torch.aten.reshape with a list of ints instead of onnx.Reshape with
    // a shape tensor. For known values, use torch.constant.int. For unknown
    // values, use the tracked size.int SSA name. This gives IREE direct
    // traceable shape values.
    if (op_type == "Reshape" && inputs.size() >= 2 && inputs[1]) {
      std::string shape_name = inputs[1].GetName();
      auto prod_it = producer_info_.find(shape_name);
      if (prod_it != producer_info_.end() &&
          prod_it->second.op_type == "Concat") {
        const auto& concat_inputs = prod_it->second.input_names;
        std::vector<std::string> dim_ssa_names;
        bool can_rewrite = true;
        int counter = 0;
        for (const auto& inp_name : concat_inputs) {
          auto cit = known_int64_vectors_.find(inp_name);
          if (cit != known_int64_vectors_.end() && cit->second.size() == 1) {
            // Known constant (including ONNX -1 "infer").
            std::string cname = SanitizeName(output_names[0]) +
                                "_rs_c" + std::to_string(counter++);
            out_ << std::format("    %{} = torch.constant.int {}\n",
                                cname, cit->second[0]);
            dim_ssa_names.push_back(cname);
          } else {
            // Try size.int reference for unknown dims.
            auto sit = size_int_ssa_map_.find(inp_name);
            if (sit != size_int_ssa_map_.end()) {
              dim_ssa_names.push_back(sit->second);
            } else {
              can_rewrite = false;
              break;
            }
          }
        }
        if (can_rewrite && !dim_ssa_names.empty()) {
          std::string out_name = SanitizeName(output_names[0]);
          std::string data_name = SanitizeName(inputs[0].GetName());
          std::string data_type = GetVtensorType(inputs[0].GetName(),
                                                  inputs[0].TypeInfo());
          // Build list construct.
          std::ostringstream dim_refs, dim_types;
          for (size_t i = 0; i < dim_ssa_names.size(); ++i) {
            if (i > 0) { dim_refs << ", "; dim_types << ", "; }
            dim_refs << "%" << dim_ssa_names[i];
            dim_types << "!torch.int";
          }
          std::string list_name = out_name + "_rs_shape";
          out_ << std::format(
              "    %{0} = torch.prim.ListConstruct {1} : ({2}) -> !torch.list<int>\n",
              list_name, dim_refs.str(), dim_types.str());
          out_ << std::format(
              "    %{0} = torch.aten.reshape %{1}, %{2} : {3}, !torch.list<int> -> {4}\n",
              out_name, data_name, list_name, data_type,
              output_type_strs[0]);
          return nullptr;
        }
      }
      // Fallback: if shape is a known constant vector (not from Concat),
      // try direct constant rewrite (max 1 unknown allowed).
      if (!shape_name.empty()) {
        std::vector<int64_t> shape_vals;
        int unknown_count = 0;
        if (ResolveShapeVector(shape_name, shape_vals, &unknown_count) &&
            !shape_vals.empty() && unknown_count == 0) {
          // All values known (may include ONNX -1).
          std::string out_name = SanitizeName(output_names[0]);
          std::string data_name = SanitizeName(inputs[0].GetName());
          std::string data_type = GetVtensorType(inputs[0].GetName(),
                                                  inputs[0].TypeInfo());
          std::string unique_pfx = out_name + "_rs_";
          std::vector<std::string> dim_names;
          std::ostringstream dim_types;
          for (size_t i = 0; i < shape_vals.size(); ++i) {
            std::string dname = unique_pfx + "d" + std::to_string(i);
            out_ << std::format("    %{} = torch.constant.int {}\n",
                                dname, shape_vals[i]);
            dim_names.push_back(dname);
            if (i > 0) dim_types << ", ";
            dim_types << "!torch.int";
          }
          std::ostringstream dim_refs;
          for (size_t i = 0; i < dim_names.size(); ++i) {
            if (i > 0) dim_refs << ", ";
            dim_refs << "%" << dim_names[i];
          }
          std::string list_name = unique_pfx + "shape";
          out_ << std::format(
              "    %{0} = torch.prim.ListConstruct {1} : ({2}) -> !torch.list<int>\n",
              list_name, dim_refs.str(), dim_types.str());
          out_ << std::format(
              "    %{0} = torch.aten.reshape %{1}, %{2} : {3}, !torch.list<int> -> {4}\n",
              out_name, data_name, list_name, data_type,
              output_type_strs[0]);
          return nullptr;
        }
      }
    }

    // For Unsqueeze: extract axes from the constant tensor input and emit
    // torch.aten.unsqueeze. ONNX Unsqueeze takes axes as a tensor input,
    // but torch-mlir expects an integer attribute.
    if (op_type == "Unsqueeze" && inputs.size() >= 2 && inputs[1]) {
      std::string axes_name = inputs[1].GetName();
      auto cit = known_int64_vectors_.find(axes_name);
      if (cit != known_int64_vectors_.end() && cit->second.size() == 1) {
        int64_t axis = cit->second[0];
        std::string out_name = SanitizeName(output_names[0]);
        std::string data_name = SanitizeName(inputs[0].GetName());
        std::string data_type = GetVtensorType(inputs[0].GetName(),
                                                inputs[0].TypeInfo());
        std::string dim_name = out_name + "_axis";
        out_ << std::format("    %{} = torch.constant.int {}\n",
                            dim_name, axis);
        out_ << std::format(
            "    %{0} = torch.aten.unsqueeze %{1}, %{2} : {3}, !torch.int -> {4}\n",
            out_name, data_name, dim_name, data_type,
            output_type_strs[0]);
        return nullptr;
      }
      // Multi-axis unsqueeze: emit sequential unsqueezes.
      if (cit != known_int64_vectors_.end() && cit->second.size() > 1) {
        std::string data_name = SanitizeName(inputs[0].GetName());
        std::string data_type = GetVtensorType(inputs[0].GetName(),
                                                inputs[0].TypeInfo());
        auto axes = cit->second;
        // Sort axes ascending so insertion positions are correct.
        std::sort(axes.begin(), axes.end());
        std::string cur_name = data_name;
        std::string cur_type = data_type;
        for (size_t ai = 0; ai < axes.size(); ++ai) {
          std::string step_name = (ai + 1 == axes.size())
              ? SanitizeName(output_names[0])
              : SanitizeName(output_names[0]) + "_usq" + std::to_string(ai);
          std::string step_type = (ai + 1 == axes.size())
              ? output_type_strs[0]
              // Intermediate type: we don't track it precisely, use output
              // type as approximation for last step; for intermediates, use
              // a generic dynamic type.
              : output_type_strs[0];
          std::string dim_name = step_name + "_axis";
          out_ << std::format("    %{} = torch.constant.int {}\n",
                              dim_name, axes[ai]);
          out_ << std::format(
              "    %{0} = torch.aten.unsqueeze %{1}, %{2} : {3}, !torch.int -> {4}\n",
              step_name, cur_name, dim_name, cur_type, step_type);
          cur_name = step_name;
          cur_type = step_type;
        }
        return nullptr;
      }
    }

    // MatMul f16→f32 accumulation promotion for gfx1100 (RDNA3).
    // Disabled: causes codegen distribution failures with static shapes.
    // The model produces correct text output without this promotion.
    // TODO: Re-enable selectively for problematic matmul shapes.
    if (false && op_type == "MatMul" && !target_config_.target_arch.empty() &&
        inputs.size() >= 2 && inputs[0] && inputs[1] &&
        outputs.size() >= 1 && outputs[0]) {
      auto info_a = inputs[0].TypeInfo().GetTensorTypeAndShapeInfo();
      auto info_b = inputs[1].TypeInfo().GetTensorTypeAndShapeInfo();
      if (info_a.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
          info_b.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        std::string out_name = SanitizeName(output_names[0]);
        std::string a_name = SanitizeName(inputs[0].GetName());
        std::string b_name = SanitizeName(inputs[1].GetName());
        std::string a_type = GetVtensorType(inputs[0].GetName(), inputs[0].TypeInfo());
        std::string b_type = GetVtensorType(inputs[1].GetName(), inputs[1].TypeInfo());
        // Build f32 type strings by replacing "f16" with "f32".
        auto replace_f16 = [](std::string s) {
          size_t pos = s.rfind("f16");
          if (pos != std::string::npos) s.replace(pos, 3, "f32");
          return s;
        };
        std::string a_f32_type = replace_f16(a_type);
        std::string b_f32_type = replace_f16(b_type);
        std::string out_f32_type = replace_f16(output_type_strs[0]);
        std::string out_f16_type = output_type_strs[0];

        // Cast inputs to f32.
        out_ << std::format(
            "    %{0}_a_f32 = torch.operator \"onnx.Cast\"(%{1}) "
            "{{torch.onnx.to = 1 : si64}} : ({2}) -> {3}\n",
            out_name, a_name, a_type, a_f32_type);
        out_ << std::format(
            "    %{0}_b_f32 = torch.operator \"onnx.Cast\"(%{1}) "
            "{{torch.onnx.to = 1 : si64}} : ({2}) -> {3}\n",
            out_name, b_name, b_type, b_f32_type);
        // MatMul in f32.
        out_ << std::format(
            "    %{0}_f32 = torch.operator \"onnx.MatMul\""
            "(%{0}_a_f32, %{0}_b_f32) {{}} : ({1}, {2}) -> {3}\n",
            out_name, a_f32_type, b_f32_type, out_f32_type);
        // Cast back to f16.
        out_ << std::format(
            "    %{0} = torch.operator \"onnx.Cast\"(%{0}_f32) "
            "{{torch.onnx.to = 10 : si64}} : ({1}) -> {2}\n",
            out_name, out_f32_type, out_f16_type);
        return nullptr;
      }
    }

    // Constant shape tensor for non-Reshape ops (currently unused, kept for
    // potential future ConstantOfShape use).
    std::string const_shape_ssa_name;
    size_t const_shape_input_idx = SIZE_MAX;

    // Build input SSA references.
    std::ostringstream in_names;
    std::ostringstream in_types;
    bool first_input = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i]) {
        // Skip invalid inputs (optional inputs can be empty/null).
        continue;
      }
      std::string input_name = inputs[i].GetName();
      if (input_name.empty()) {
        continue;
      }
      if (!first_input) {
        in_names << ", ";
        in_types << ", ";
      }
      first_input = false;
      // Use the constant shape tensor if we emitted one for this input.
      if (i == const_shape_input_idx && !const_shape_ssa_name.empty()) {
        in_names << "%" << const_shape_ssa_name;
        in_types << std::format("!torch.vtensor<[{}],si64>",
            inputs[i].TypeInfo().GetTensorTypeAndShapeInfo().GetShape()[0]);
      } else {
        in_names << "%" << SanitizeName(input_name);
        in_types << GetVtensorType(input_name, inputs[i].TypeInfo());
      }
    }

    // Build attributes.
    std::string attr_str = FormatAttributes(attrs);

    // Format output types: wrap in parentheses if multiple outputs.
    std::string out_types_str = out_types.str();
    if (valid_output_count > 1 && !out_types_str.empty()) {
      out_types_str = "(" + out_types_str + ")";
    }

    // Emit the operator.
    constexpr std::string_view schema =
        R"(    {0} = torch.operator "onnx.{1}"({2}) {{{3}}} : ({4}) -> {5}
)";
    out_ << std::format(schema,
                        out_names.str(),  // {0}
                        op_type,          // {1}
                        in_names.str(),   // {2}
                        attr_str,         // {3}
                        in_types.str(),   // {4}
                        out_types_str);   // {5}
    return nullptr;
  }

  std::string FormatAttributes(const std::vector<Ort::ConstOpAttr>& attrs) {
    if (attrs.empty()) {
      return "";
    }

    std::ostringstream ss;
    for (size_t i = 0; i < attrs.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << FormatAttribute(attrs[i]);
    }
    return ss.str();
  }

  std::string FormatAttribute(const Ort::ConstOpAttr& attr) {
    std::string name = attr.GetName();
    OrtOpAttrType type = attr.GetType();

    switch (type) {
      case ORT_OP_ATTR_INT: {
        int64_t value = 0;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = {1} : si64", name, value);
      }
      case ORT_OP_ATTR_FLOAT: {
        float value = 0.0f;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = {1:e} : f32", name, value);
      }
      case ORT_OP_ATTR_STRING: {
        std::string value;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = \"{1}\"", name, value);
      }
      case ORT_OP_ATTR_INTS: {
        std::vector<int64_t> values;
        attr.GetValueArray<int64_t>(values);
        std::vector<std::string> str_values(values.size());
        std::transform(values.begin(), values.end(), str_values.begin(),
                       [](int64_t v) { return std::format("{0} : si64", v); });
        return std::format("torch.onnx.{0} = [{1}]", name,
                           Join(str_values, ", "));
      }
      case ORT_OP_ATTR_TENSOR: {
        Ort::Value tensor_value{nullptr};
        auto status = attr.GetTensorAttributeAsOrtValue(tensor_value);
        if (!status.IsOK()) {
          return std::format("torch.onnx.{0} = \"NYI_TENSOR_ERROR\"", name);
        }
        auto type_info = tensor_value.GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();
        auto dtype = type_info.GetElementType();
        size_t count = type_info.GetElementCount();
        const auto* raw =
            static_cast<const uint8_t*>(tensor_value.GetTensorRawData());

        // Format as dense<value> : tensor<shape x dtype>.
        std::ostringstream dense;
        dense << "dense<";
        if (count == 0) {
          dense << "[]";
        } else if (count == 1) {
          switch (dtype) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
              float v = *reinterpret_cast<const float*>(raw);
              if (v == 0.0f) {
                dense << "0.000000e+00";
              } else if (std::isinf(v) || std::isnan(v)) {
                // Special float values must use hex encoding in MLIR.
                dense << "\"" << HexEncode(raw, 4) << "\"";
              } else {
                dense << std::format("{:e}", v);
              }
              break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
              dense << *reinterpret_cast<const int64_t*>(raw);
              break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
              dense << *reinterpret_cast<const int32_t*>(raw);
              break;
            default:
              // Hex encoding is the universal fallback for MLIR's dense<>
              // syntax — it works for any element type (f16, bf16, f64, etc.)
              // without needing type-specific formatting.
              dense << "\""
                    << HexEncode(raw, count * OnnxElementTypeSize(dtype))
                    << "\"";
              break;
          }
        } else {
          // Multi-element tensors: always hex-encode. ONNX tensor attributes
          // are almost always single-element (e.g., ConstantOfShape fill
          // value), so readability of multi-element cases isn't a priority.
          dense << "\"" << HexEncode(raw, count * OnnxElementTypeSize(dtype))
                << "\"";
        }
        dense << ">";

        // Build tensor type using signed integer types (matching
        // torch.onnx.value format).
        std::ostringstream type_ss;
        type_ss << "tensor<";
        for (size_t j = 0; j < shape.size(); ++j) {
          type_ss << shape[j] << "x";
        }
        type_ss << GetElementType(dtype, /*signless=*/false) << ">";

        return std::format("torch.onnx.{0} = {1} : {2}", name, dense.str(),
                           type_ss.str());
      }
      default:
        return std::format("torch.onnx.{0} = \"NYI\"", name);
    }
  }

  // Emits ops to resolve a spec string to an SSA value. The spec is either:
  // - A literal integer (emits arith.constant)
  // - "$N" input reference (extracts scalar from input tensor N)
  // as_i32: if true, result is i32; otherwise index.
  // context: used in error messages, e.g. "push_constants[0]".
  OrtStatus* EmitResolvedValue(
      const std::string& spec, const std::string& result_name, bool as_i32,
      const std::vector<std::string>& raw_input_names,
      const std::vector<std::string>& raw_input_types,
      const std::vector<std::string>& raw_input_elem_types,
      const std::vector<size_t>& raw_input_ranks, const std::string& context) {
    size_t input_idx;
    if (ParseInputRef(spec, input_idx)) {
      if (input_idx >= raw_input_names.size()) {
        return MakeError(
            "ExternDispatch: {} = '{}' references input[{}] but only {} "
            "inputs available",
            context, spec, input_idx, raw_input_names.size());
      }
      if (raw_input_names[input_idx].empty()) {
        return MakeError(
            "ExternDispatch: {} = '{}' references input[{}] which is "
            "null or empty",
            context, spec, input_idx);
      }
      if (raw_input_ranks[input_idx] != 0) {
        return MakeError(
            "ExternDispatch: {} = '{}' references input[{}] which has "
            "rank {} (must be a scalar tensor, rank 0)",
            context, spec, input_idx, raw_input_ranks[input_idx]);
      }
      const std::string& elem_type = raw_input_elem_types[input_idx];
      if (elem_type != "i64" && elem_type != "i32") {
        return MakeError(
            "ExternDispatch: {} = '{}' references input[{}] with element "
            "type {} (must be i32 or i64)",
            context, spec, input_idx, elem_type);
      }
      // Extract scalar from the rank-0 tensor.
      if (as_i32 && elem_type == "i32") {
        // Already i32 — extract directly into result, no cast needed.
        out_ << std::format("    %{} = tensor.extract %{}[] : {}\n",
                            result_name, raw_input_names[input_idx],
                            raw_input_types[input_idx]);
      } else {
        std::string scalar = result_name + "_scalar";
        out_ << std::format("    %{} = tensor.extract %{}[] : {}\n", scalar,
                            raw_input_names[input_idx],
                            raw_input_types[input_idx]);
        // Cast to the target type.
        if (as_i32) {
          // elem_type is i64 here (i32 case handled above).
          out_ << std::format("    %{} = arith.trunci %{} : i64 to i32\n",
                              result_name, scalar);
        } else {
          out_ << std::format("    %{} = arith.index_cast %{} : {} to index\n",
                              result_name, scalar, elem_type);
        }
      }
    } else {
      size_t value;
      if (!ParseUnsigned(spec, value)) {
        return MakeError(
            "ExternDispatch: {} = '{}' is not a valid integer literal "
            "or input reference ($N)",
            context, spec);
      }
      if (as_i32 && value > static_cast<size_t>(INT32_MAX)) {
        return MakeError("ExternDispatch: {} = '{}' exceeds i32 range (max {})",
                         context, spec, INT32_MAX);
      }
      out_ << std::format("    %{} = arith.constant {} : {}\n", result_name,
                          spec, as_i32 ? "i32" : "index");
    }
    return nullptr;
  }

  // Emits a hal.dispatch.extern for a com.iree:ExternDispatch ONNX node.
  // Emits a util.global.load for a com.iree:LoadGlobal node.
  // LoadGlobal has one output (the loaded tensor) and a "global_name" attribute.
  OrtStatus* EmitLoadGlobal(const Ort::ConstNode& node) {
    auto outputs = node.GetOutputs();
    if (outputs.empty() || !outputs[0])
      return MakeError("LoadGlobal: missing output");

    std::string global_name;
    for (const auto& attr : node.GetAttributes()) {
      if (attr.GetName() == "global_name") attr.GetValue(global_name);
    }
    if (global_name.empty())
      return MakeError("LoadGlobal: missing 'global_name' attribute");

    auto it = declared_globals_.find(global_name);
    if (it == declared_globals_.end())
      return MakeError("LoadGlobal: global '{}' not declared", global_name);

    std::string builtin_type = it->second;
    std::string out_name = SanitizeName(outputs[0].GetName());
    std::string vtensor_type = GetVtensorType(
        outputs[0].GetName(), outputs[0].TypeInfo());

    // Emit: %raw = util.global.load @global_name : tensor<...>
    //       %out = torch_c.from_builtin_tensor %raw : tensor<...> -> vtensor
    out_ << std::format(
        "    %{0}_raw = util.global.load @{1} : {2}\n"
        "    %{0} = torch_c.from_builtin_tensor %{0}_raw : {2} -> {3}\n",
        out_name, global_name, builtin_type, vtensor_type);

    // Store refined type for downstream ops.
    refined_types_[outputs[0].GetName()] = vtensor_type;
    return nullptr;
  }

  // Emits a util.global.store for a com.iree:StoreGlobal node.
  // StoreGlobal has one input (the tensor to store) and a "global_name" attribute.
  OrtStatus* EmitStoreGlobal(const Ort::ConstNode& node) {
    auto inputs = node.GetInputs();
    if (inputs.empty() || !inputs[0])
      return MakeError("StoreGlobal: missing input");

    std::string global_name;
    for (const auto& attr : node.GetAttributes()) {
      if (attr.GetName() == "global_name") attr.GetValue(global_name);
    }
    if (global_name.empty())
      return MakeError("StoreGlobal: missing 'global_name' attribute");

    auto it = declared_globals_.find(global_name);
    if (it == declared_globals_.end())
      return MakeError("StoreGlobal: global '{}' not declared", global_name);

    std::string builtin_type = it->second;
    std::string in_name = SanitizeName(inputs[0].GetName());
    std::string vtensor_type = GetVtensorType(
        inputs[0].GetName(), inputs[0].TypeInfo());

    // Emit: %raw = torch_c.to_builtin_tensor %in : vtensor -> tensor<...>
    //       util.global.store %raw, @global_name : tensor<...>
    out_ << std::format(
        "    %{0}_store_raw = torch_c.to_builtin_tensor %{0} : {1} -> {2}\n"
        "    util.global.store %{0}_store_raw, @{3} : {2}\n",
        in_name, vtensor_type, builtin_type, global_name);

    return nullptr;
  }

  OrtStatus* EmitExternDispatch(const Ort::ConstNode& node, int extern_id) {
    std::string prefix = std::format("__extern_{}", extern_id);
    auto inputs = node.GetInputs();
    auto outputs = node.GetOutputs();

    // Parse ExternDispatch attributes.
    std::string kernel_name;
    std::string kernel_object;
    std::vector<int64_t> workgroup_size;
    std::vector<std::string> push_constants;
    std::vector<std::string> workgroup_count;

    for (const auto& attr : node.GetAttributes()) {
      std::string name = attr.GetName();
      if (name == "kernel_name") {
        attr.GetValue(kernel_name);
      } else if (name == "kernel_object") {
        attr.GetValue(kernel_object);
      } else if (name == "workgroup_size") {
        attr.GetValueArray<int64_t>(workgroup_size);
      } else if (name == "push_constants") {
        attr.GetValueArray<std::string>(push_constants);
      } else if (name == "workgroup_count") {
        attr.GetValueArray<std::string>(workgroup_count);
      }
    }

    // Validate backend supports extern dispatch (must have a HAL target).
    if (target_config_.hal_backend.empty()) {
      return MakeError(
          "ExternDispatch: backend '{}' does not support extern dispatch",
          target_config_.backend);
    }

    // Validate required attributes.
    if (kernel_name.empty()) {
      return MakeError(
          "ExternDispatch: missing required 'kernel_name' attribute");
    }
    if (kernel_object.empty()) {
      return MakeError(
          "ExternDispatch: missing required 'kernel_object' attribute");
    }
    if (workgroup_size.size() != 3) {
      return MakeError(
          "ExternDispatch: 'workgroup_size' must have exactly 3 elements "
          "(X, Y, Z), got {}",
          workgroup_size.size());
    }
    for (size_t i = 0; i < 3; ++i) {
      if (workgroup_size[i] <= 0) {
        return MakeError(
            "ExternDispatch: workgroup_size[{}] = {} must be positive", i,
            workgroup_size[i]);
      }
    }
    if (workgroup_count.size() != 3) {
      return MakeError(
          "ExternDispatch: 'workgroup_count' must have "
          "exactly 3 elements (X, Y, Z), got {}",
          workgroup_count.size());
    }

    // Pre-scan push_constants and workgroup_count for $N input references.
    // Inputs referenced by $N are scalar tensors consumed as values, NOT
    // data buffers — they must be excluded from dispatch args and bindings.
    std::unordered_set<size_t> scalar_input_indices;
    for (const auto& spec : push_constants) {
      size_t idx;
      if (ParseInputRef(spec, idx)) scalar_input_indices.insert(idx);
    }
    for (const auto& spec : workgroup_count) {
      size_t idx;
      if (ParseInputRef(spec, idx)) scalar_input_indices.insert(idx);
    }

    // Step 1: Bridge ALL inputs from torch to builtin tensor.
    // Vectors are indexed by original ONNX input position so that $N input
    // refs resolve correctly. Scalar inputs (referenced by $N) are bridged
    // too — they need builtin tensors for tensor.extract.
    std::vector<std::string> raw_input_names(inputs.size());
    std::vector<std::string> raw_input_types(inputs.size());
    std::vector<std::string> raw_input_elem_types(inputs.size());
    std::vector<size_t> raw_input_ranks(inputs.size(), 0);
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i]) continue;
      std::string input_name = inputs[i].GetName();
      if (input_name.empty()) continue;

      std::string ssa = SanitizeName(input_name);
      std::string vtensor_type = GetVtensorType(input_name, inputs[i].TypeInfo());
      std::string tensor_type = FormatMlirTensorType(inputs[i].TypeInfo());
      auto tensor_info = inputs[i].TypeInfo().GetTensorTypeAndShapeInfo();

      // Use refined type info for the builtin tensor type. ORT's TypeInfo may
      // have all-dynamic dims even when the EP has refined shapes. Parse the
      // vtensor type (which uses refined shapes) and rebuild the tensor type.
      std::vector<int64_t> refined_dims;
      std::string refined_elem;
      if (ParseVtensorType(vtensor_type, refined_dims, refined_elem)) {
        std::string signless_elem =
            GetElementType(tensor_info.GetElementType(), /*signless=*/true);
        std::ostringstream refined_ss;
        refined_ss << "tensor<";
        for (size_t d = 0; d < refined_dims.size(); ++d) {
          if (refined_dims[d] <= 0) refined_ss << "?";
          else refined_ss << refined_dims[d];
          if (d + 1 < refined_dims.size()) refined_ss << "x";
          else refined_ss << "x" << signless_elem;
        }
        refined_ss << ">";
        tensor_type = refined_ss.str();
      }

      std::string raw = std::format("{}_raw_{}", prefix, i);
      raw_input_names[i] = raw;
      raw_input_types[i] = tensor_type;
      raw_input_elem_types[i] =
          GetElementType(tensor_info.GetElementType(), /*signless=*/true);
      raw_input_ranks[i] = tensor_info.GetShape().size();

      out_ << std::format(
          "    %{} = torch_c.to_builtin_tensor %{} : {} -> {}\n", raw, ssa,
          vtensor_type, tensor_type);
    }

    // Step 2: Emit push constants.
    std::vector<std::string> pc_names;
    for (size_t i = 0; i < push_constants.size(); ++i) {
      std::string pc = std::format("{}_pc{}", prefix, i);
      pc_names.push_back(pc);
      OrtStatus* status = EmitResolvedValue(
          push_constants[i], pc, /*as_i32=*/true, raw_input_names,
          raw_input_types, raw_input_elem_types, raw_input_ranks,
          std::format("push_constants[{}]", i));
      if (status) return status;
    }

    // Step 3: Compute workload X from workgroup_count[0].
    // X is passed as the workload argument to hal.dispatch.extern and becomes
    // %workload inside the count region. Y and Z must be literals since the
    // count region is isolated and can only reference its own arguments.
    std::string workload_x_name = std::format("{}_workload", prefix);
    {
      OrtStatus* status = EmitResolvedValue(
          workgroup_count[0], workload_x_name, /*as_i32=*/false,
          raw_input_names, raw_input_types, raw_input_elem_types,
          raw_input_ranks, "workgroup_count[0]");
      if (status) return status;
    }

    // Validate Y and Z workgroup counts are literals (count region is
    // isolated).
    for (size_t i = 1; i < 3; ++i) {
      size_t unused_idx;
      if (ParseInputRef(workgroup_count[i], unused_idx)) {
        return MakeError(
            "ExternDispatch: workgroup_count[{}] = '{}' is dynamic, but only "
            "workgroup_count[0] (X) can be dynamic — Y and Z must be integer "
            "literals",
            i, workgroup_count[i]);
      }
      size_t unused;
      if (!ParseUnsigned(workgroup_count[i], unused)) {
        return MakeError(
            "ExternDispatch: workgroup_count[{}] = '{}' is not a valid "
            "integer literal",
            i, workgroup_count[i]);
      }
    }

    // Step 4: Build output names and types.
    // ExternDispatch is opaque — ORT cannot infer its output shapes. The ONNX
    // model must declare explicit type info for all ExternDispatch outputs
    // (via graph value_info entries).
    std::vector<std::string> out_raw_names;
    std::vector<std::string> out_raw_types;
    std::vector<std::string> out_vtensor_types;
    std::vector<std::string> out_ssa_names;
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (!outputs[i]) continue;
      std::string output_name = outputs[i].GetName();
      if (output_name.empty()) continue;
      out_raw_names.push_back(std::format("{}_out{}", prefix, i));
      out_raw_types.push_back(FormatMlirTensorType(outputs[i].TypeInfo()));
      out_vtensor_types.push_back(GetVtensorType(output_name, outputs[i].TypeInfo()));
      out_ssa_names.push_back(SanitizeName(output_name));
    }

    if (out_raw_names.empty()) {
      return MakeError(
          "ExternDispatch: node has no valid outputs (all outputs are "
          "null or have empty names)");
    }

    // Step 5: Emit hal.dispatch.extern.
    // Pre-build variable parts for the dispatch template.
    std::vector<std::string> dispatch_outs;
    for (const auto& name : out_raw_names) {
      dispatch_outs.push_back("%" + name);
    }

    std::vector<std::string> dispatch_args;
    for (const auto& pc : pc_names) {
      dispatch_args.push_back("%" + pc);
    }
    for (size_t i = 0; i < raw_input_names.size(); ++i) {
      if (!raw_input_names[i].empty() && !scalar_input_indices.count(i)) {
        dispatch_args.push_back("%" + raw_input_names[i]);
      }
    }

    // For dynamic tensor types (containing '?'), we need to extract the
    // dynamic dimensions and annotate them in the dispatch signature with
    // {%dim0, %dim1, ...} syntax. This is required by hal.dispatch.extern.
    // We collect all dynamic dims from inputs and outputs into a shared pool
    // keyed by the raw tensor name.
    auto emit_dynamic_dims = [&](const std::string& raw_name,
                                  const std::string& tensor_type)
        -> std::string {
      // Count '?' occurrences in the tensor type.
      size_t ndyn = 0;
      for (char c : tensor_type) if (c == '?') ndyn++;
      if (ndyn == 0) return "";
      // Emit tensor.dim ops for each dynamic dimension.
      std::vector<std::string> dim_names;
      size_t dim_idx = 0;
      for (size_t pos = 0; pos < tensor_type.size(); ++pos) {
        if (tensor_type[pos] == '?') {
          std::string dname = raw_name + "_d" + std::to_string(dim_idx);
          out_ << std::format(
              "    %{0}_cidx = arith.constant {1} : index\n"
              "    %{0} = tensor.dim %{2}, %{0}_cidx : {3}\n",
              dname, dim_idx, raw_name, tensor_type);
          dim_names.push_back("%" + dname);
          dim_idx++;
        }
      }
      return "{" + Join(dim_names, ", ") + "}";
    };

    std::vector<std::string> dispatch_arg_types;
    for (size_t i = 0; i < pc_names.size(); ++i) {
      dispatch_arg_types.push_back("i32");
    }
    for (size_t i = 0; i < raw_input_types.size(); ++i) {
      if (!raw_input_types[i].empty() && !scalar_input_indices.count(i)) {
        std::string annot = emit_dynamic_dims(
            raw_input_names[i], raw_input_types[i]);
        dispatch_arg_types.push_back(raw_input_types[i] + annot);
      }
    }

    // For output types, we need dim annotations too. Outputs don't have
    // raw tensors yet (they're created by the dispatch), so we tie their
    // dynamic dims to the first input tensor that shares the same dynamic
    // dimension. For typical MoE patterns, all dynamic dims are N (token
    // count) from the first input.
    std::string ret_types_str;
    std::vector<std::string> annotated_out_types;
    for (size_t i = 0; i < out_raw_types.size(); ++i) {
      size_t ndyn = 0;
      for (char c : out_raw_types[i]) if (c == '?') ndyn++;
      if (ndyn == 0) {
        annotated_out_types.push_back(out_raw_types[i]);
      } else {
        // Find the first input with a dynamic dim to share.
        // If no input is dynamic, resolve the output ? to static dim from
        // the first non-scalar input's dim[0].
        std::string annot;
        for (size_t j = 0; j < raw_input_types.size() && annot.empty(); ++j) {
          if (raw_input_types[j].empty() || scalar_input_indices.count(j))
            continue;
          size_t inp_ndyn = 0;
          for (char c : raw_input_types[j]) if (c == '?') inp_ndyn++;
          if (inp_ndyn > 0) {
            std::string dname = raw_input_names[j] + "_d0";
            std::vector<std::string> parts(ndyn, "%" + dname);
            annot = "{" + Join(parts, ", ") + "}";
          }
        }
        if (annot.empty()) {
          // All inputs are static. Replace ? in output type with the
          // known static dim from the first non-scalar input's first dim.
          std::string resolved = out_raw_types[i];
          for (size_t j = 0; j < raw_input_types.size(); ++j) {
            if (raw_input_types[j].empty() || scalar_input_indices.count(j))
              continue;
            // Parse first dim from input type: tensor<DIMx...>
            auto tp = raw_input_types[j];
            size_t start = tp.find('<');
            size_t end = tp.find('x', start);
            if (start != std::string::npos && end != std::string::npos) {
              std::string dim_str = tp.substr(start + 1, end - start - 1);
              if (dim_str != "?") {
                // Replace each ? in output with this static dim
                size_t pos = 0;
                while ((pos = resolved.find('?', pos)) != std::string::npos) {
                  resolved.replace(pos, 1, dim_str);
                  pos += dim_str.length();
                }
              }
            }
            break;
          }
          annotated_out_types.push_back(resolved);
        } else {
          annotated_out_types.push_back(out_raw_types[i] + annot);
        }
      }
    }
    if (annotated_out_types.size() == 1) {
      ret_types_str = annotated_out_types[0];
    } else {
      ret_types_str = "(" + Join(annotated_out_types, ", ") + ")";
    }

    std::vector<std::string> bindings;
    for (size_t i = 0; i < raw_input_names.size(); ++i) {
      if (!raw_input_names[i].empty() && !scalar_input_indices.count(i)) {
        bindings.push_back(
            "        #hal.pipeline.binding<storage_buffer, ReadOnly>");
      }
    }
    for (size_t i = 0; i < out_raw_names.size(); ++i) {
      bindings.push_back("        #hal.pipeline.binding<storage_buffer>");
    }

    std::vector<std::string> wg_size_parts;
    for (auto s : workgroup_size) {
      wg_size_parts.push_back(std::format("{} : index", s));
    }

    constexpr std::string_view dispatch_schema =
        R"(    {0} = hal.dispatch.extern "{1}"[%{2}]({3})
        : ({4}) -> {5}
      count(%device: !hal.device, %workload: index) -> (index, index, index) {{
        %count_y = arith.constant {6} : index
        %count_z = arith.constant {7} : index
        hal.return %workload, %count_y, %count_z : index, index, index
      }}
      layout(#hal.pipeline.layout<constants = {8}, bindings = [
{9}      ]>)
      objects({{
        #executable_target ordinal(0) = [
          #hal.executable.object<{{path = "{10}"}}>
        ]
      }})
      attributes {{workgroup_size = [{11}]}}
)";
    out_ << std::format(dispatch_schema, Join(dispatch_outs, ", "),  // {0}
                        kernel_name,                                 // {1}
                        workload_x_name,                             // {2}
                        Join(dispatch_args, ", "),                   // {3}
                        Join(dispatch_arg_types, ", "),              // {4}
                        ret_types_str,                               // {5}
                        workgroup_count[1],                          // {6}
                        workgroup_count[2],                          // {7}
                        pc_names.size(),                             // {8}
                        Join(bindings, ",\n") + "\n",                // {9}
                        kernel_object,                               // {10}
                        Join(wg_size_parts, ", "));                  // {11}

    // Step 6: Bridge outputs back to torch types.
    // When the output tensor type was resolved to static (no '?'), update
    // the vtensor type to match so from_builtin_tensor types are consistent.
    for (size_t i = 0; i < out_raw_names.size(); ++i) {
      std::string raw_type = annotated_out_types.size() > i
          ? annotated_out_types[i] : out_raw_types[i];
      // Strip any {%dim} annotation for the bridge type
      auto brace = raw_type.find('{');
      if (brace != std::string::npos) raw_type = raw_type.substr(0, brace);
      std::string vt = out_vtensor_types[i];
      // If raw_type has no '?' but vtensor has '?', replace '?' in vtensor
      // with the corresponding static dim from raw_type.
      if (raw_type.find('?') == std::string::npos &&
          vt.find('?') != std::string::npos) {
        // Parse dims from raw_type: tensor<AxBxCxdtype>
        std::vector<std::string> raw_dims;
        size_t s = raw_type.find('<') + 1;
        while (s < raw_type.size()) {
          size_t e = raw_type.find('x', s);
          if (e == std::string::npos) break;
          std::string d = raw_type.substr(s, e - s);
          if (d.empty() || !std::isdigit(d[0])) break;
          raw_dims.push_back(d);
          s = e + 1;
        }
        // Replace each '?' in vtensor with corresponding raw dim
        size_t di = 0;
        for (size_t p = 0; p < vt.size() && di < raw_dims.size(); ++p) {
          if (vt[p] == '?') {
            vt.replace(p, 1, raw_dims[di++]);
          }
        }
      }
      out_ << std::format(
          "    %{} = torch_c.from_builtin_tensor %{} : {} -> {}\n",
          out_ssa_names[i], out_raw_names[i], raw_type, vt);
      // Store the resolved vtensor type so downstream ExternDispatch nodes
      // consuming this output use the correct (potentially static) type.
      if (i < outputs.size() && outputs[i]) {
        refined_types_[outputs[i].GetName()] = vt;
      }
    }
    return nullptr;
  }

  void EmitReturn() {
    std::ostringstream ret_values;
    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_values << ", ";
        ret_types << ", ";
      }
      ret_values << "%" << SanitizeName(graph_outputs_[i].GetName());
      std::string out_name = graph_outputs_[i].GetName();
      ret_types << GetVtensorType(out_name, graph_outputs_[i].TypeInfo());
    }

    out_ << std::format("    return {0} : {1}\n", ret_values.str(),
                        ret_types.str());
  }

  void EmitModuleFooter() {
    out_ << "  }\n";
    out_ << "}\n";
  }

  // Shape refinement: try to resolve static dimensions for Reshape outputs.
  // ORT's shape inference can't trace constant values through Concat→Reshape
  // chains, so it marks all output dims as dynamic. We recover static dims by
  // tracking known constant values from initializers and Constant nodes.

  // Returns a refined output type string for Reshape if we can recover any
  // static dims, or empty string if we can't improve on ORT's inference.
  // Parse a vtensor type string like "!torch.vtensor<[?,?,16,128],f16>" into
  // shape dims and element type. Returns false if parsing fails.
  static bool ParseVtensorType(const std::string& type,
                                std::vector<int64_t>& dims,
                                std::string& elem_type) {
    // Expected format: !torch.vtensor<[d0,d1,...],dtype>
    auto bracket_start = type.find('[');
    auto bracket_end = type.find(']');
    auto last_comma = type.rfind(',');
    auto angle_end = type.rfind('>');
    if (bracket_start == std::string::npos || bracket_end == std::string::npos ||
        last_comma == std::string::npos || angle_end == std::string::npos)
      return false;

    // Parse dims between [ and ]
    dims.clear();
    std::string dims_str = type.substr(bracket_start + 1, bracket_end - bracket_start - 1);
    if (dims_str.empty()) return false;
    std::istringstream iss(dims_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
      if (token == "?") {
        dims.push_back(-1);
      } else {
        dims.push_back(std::stoll(token));
      }
    }

    // Element type is after the last comma before >
    elem_type = type.substr(bracket_end + 2, angle_end - bracket_end - 2);
    return !dims.empty();
  }

  // Build a vtensor type string from dims and element type.
  static std::string BuildVtensorType(const std::vector<int64_t>& dims,
                                       const std::string& elem_type) {
    std::ostringstream ss;
    ss << "!torch.vtensor<[";
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i > 0) ss << ",";
      if (dims[i] < 0) ss << "?"; else ss << dims[i];
    }
    ss << "]," << elem_type << ">";
    return ss.str();
  }

  // Propagate refined types through ops with predictable shape behavior.
  // Updates output_type_strs in-place and stores results in refined_types_.
  void PropagateRefinedTypes(
      const std::string& op_type,
      const std::vector<Ort::ConstValueInfo>& inputs,
      const std::vector<Ort::ConstValueInfo>& outputs,
      const std::vector<Ort::ConstOpAttr>& attrs,
      const std::vector<std::string>& output_names,
      std::vector<std::string>& output_type_strs) {

    // Get the effective type of input[0] (refined or ORT original).
    if (inputs.empty() || !inputs[0]) return;
    std::string input0_name = inputs[0].GetName();
    if (input0_name.empty()) return;
    std::string input0_type = GetVtensorType(input0_name, inputs[0].TypeInfo());

    std::vector<int64_t> in_dims;
    std::string elem_type;
    if (!ParseVtensorType(input0_type, in_dims, elem_type)) return;

    // Helper to store a refined output type (declared early for ConstantOfShape).
    auto StoreRefined = [&](size_t idx, const std::string& refined) {
      if (idx < output_names.size() && !output_names[idx].empty()) {
        output_type_strs[idx] = refined;
        refined_types_[output_names[idx]] = refined;
      }
    };

    // ConstantOfShape: output shape comes from the VALUES in input[0],
    // not from its tensor shape. Use ResolveShapeVector to get known dims.
    if (op_type == "ConstantOfShape") {
      std::vector<int64_t> shape_vals;
      if (ResolveShapeVector(input0_name, shape_vals)) {
        auto out_info = outputs[0].TypeInfo().GetTensorTypeAndShapeInfo();
        auto ort_shape = out_info.GetShape();
        auto dtype = out_info.GetElementType();
        if (ort_shape.size() == shape_vals.size()) {
          std::vector<int64_t> merged(ort_shape.begin(), ort_shape.end());
          bool improved = false;
          for (size_t i = 0; i < merged.size(); ++i) {
            if (merged[i] < 0 && shape_vals[i] >= 0) {
              merged[i] = shape_vals[i];
              improved = true;
            }
          }
          if (improved) {
            StoreRefined(0, BuildVtensorType(merged, GetElementType(dtype)));
          }
        }
      }
      return;
    }

    // Check if input[0] has any static dims that ORT's output might lack.
    // Skip propagation if input[0] is fully dynamic.
    bool has_static_dim = false;
    for (int64_t d : in_dims) {
      if (d >= 0) { has_static_dim = true; break; }
    }
    if (!has_static_dim) return;

    // Helper to read an int64 attribute by name.
    auto GetIntAttr = [&](const std::string& attr_name,
                          int64_t default_val) -> int64_t {
      for (const auto& attr : attrs) {
        if (attr.GetName() == attr_name) {
          int64_t val = default_val;
          attr.GetValue(val);
          return val;
        }
      }
      return default_val;
    };

    auto GetIntsAttr = [&](const std::string& attr_name,
                           std::vector<int64_t>& out) -> bool {
      for (const auto& attr : attrs) {
        if (attr.GetName() == attr_name) {
          attr.GetValueArray(out);
          return !out.empty();
        }
      }
      return false;
    };

    // Helper to get ORT's output shape and element type.
    auto GetOrtOutputInfo = [&](size_t idx)
        -> std::pair<std::vector<int64_t>, ONNXTensorElementDataType> {
      auto info = outputs[idx].TypeInfo().GetTensorTypeAndShapeInfo();
      return {info.GetShape(), info.GetElementType()};
    };

    // Transpose: permute dimensions.
    if (op_type == "Transpose") {
      std::vector<int64_t> perm;
      GetIntsAttr("perm", perm);
      if (perm.size() == in_dims.size()) {
        std::vector<int64_t> out_dims(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
          out_dims[i] = in_dims[perm[i]];
        }
        StoreRefined(0, BuildVtensorType(out_dims, elem_type));
      }
    }
    // Element-wise ops: output shape matches broadcast of inputs.
    else if (op_type == "Add" || op_type == "Sub" || op_type == "Mul" ||
             op_type == "Div" || op_type == "Pow" || op_type == "Relu" ||
             op_type == "Sigmoid" || op_type == "Tanh" || op_type == "Neg" ||
             op_type == "Sqrt" || op_type == "Erf" || op_type == "Cast" ||
             op_type == "Silu" || op_type == "Sin" || op_type == "Cos" ||
             op_type == "Where" ||
             op_type == "ScatterElements" || op_type == "GatherElements") {
      auto [ort_shape, dtype] = GetOrtOutputInfo(0);
      // Collect refined dims from ALL inputs to compute broadcast shape.
      // For each dim position, pick the largest known value (>1 wins over 1,
      // known wins over unknown). This handles cases like:
      //   Mul([?,1], [?,2048]) → [?,2048]
      //   Mul([2048], [1,?,2048]) → [1,?,2048]
      std::vector<std::vector<int64_t>> all_dims;
      all_dims.push_back(in_dims);
      for (size_t i = 1; i < inputs.size(); ++i) {
        if (!inputs[i]) continue;
        std::string inp_name = inputs[i].GetName();
        if (inp_name.empty()) continue;
        std::string inp_type = GetVtensorType(inp_name, inputs[i].TypeInfo());
        std::vector<int64_t> inp_dims;
        std::string inp_elem;
        if (ParseVtensorType(inp_type, inp_dims, inp_elem)) {
          all_dims.push_back(inp_dims);
        }
      }
      // Find the max rank across all inputs.
      size_t max_rank = 0;
      for (const auto& dims : all_dims) {
        max_rank = std::max(max_rank, dims.size());
      }
      // Broadcast: right-align dims and pick the best for each position.
      std::vector<int64_t> best_dims(max_rank, -1);
      for (const auto& dims : all_dims) {
        size_t offset = max_rank - dims.size();
        for (size_t j = 0; j < dims.size(); ++j) {
          int64_t d = dims[j];
          int64_t& bd = best_dims[offset + j];
          // Prefer known > 1 dims; known=1 is a broadcast dim.
          if (d > 1) {
            bd = d;
          } else if (d == 1 && bd < 0) {
            bd = 1;  // Only set 1 if nothing better known.
          }
          // d <= 0 means dynamic — keep existing bd if better.
        }
      }
      if (ort_shape.size() == best_dims.size()) {
        std::vector<int64_t> merged(best_dims.size());
        for (size_t i = 0; i < best_dims.size(); ++i) {
          merged[i] = (ort_shape[i] >= 0) ? ort_shape[i] : best_dims[i];
        }
        StoreRefined(0, BuildVtensorType(merged, GetElementType(dtype)));
      }
    }
    // MatMul: propagate batch dims from both inputs.
    else if (op_type == "MatMul") {
      auto [ort_shape, dtype] = GetOrtOutputInfo(0);
      if (ort_shape.size() >= 2) {
        std::vector<int64_t> merged(ort_shape.begin(), ort_shape.end());
        // Gather batch dim info from all inputs.
        for (size_t inp = 0; inp < inputs.size(); ++inp) {
          if (!inputs[inp]) continue;
          std::string inp_name = inputs[inp].GetName();
          if (inp_name.empty()) continue;
          std::string inp_type = GetVtensorType(inp_name, inputs[inp].TypeInfo());
          std::vector<int64_t> inp_dims;
          std::string inp_elem;
          if (!ParseVtensorType(inp_type, inp_dims, inp_elem)) continue;
          if (inp_dims.size() != ort_shape.size()) continue;
          for (size_t i = 0; i + 2 < merged.size(); ++i) {
            if (merged[i] < 0 && inp_dims[i] >= 0) merged[i] = inp_dims[i];
          }
        }
        StoreRefined(0, BuildVtensorType(merged, GetElementType(dtype)));
      }
    }
    // ReduceMean (keepdims): non-reduced dims match input.
    else if (op_type == "ReduceMean" || op_type == "ReduceSum" ||
             op_type == "ReduceMax" || op_type == "ReduceMin") {
      auto [ort_shape, dtype] = GetOrtOutputInfo(0);
      if (ort_shape.size() == in_dims.size()) {
        // keepdims case: non-reduced dims should match input.
        std::vector<int64_t> merged(ort_shape.begin(), ort_shape.end());
        for (size_t i = 0; i < merged.size(); ++i) {
          if (merged[i] < 0 && in_dims[i] >= 0) merged[i] = in_dims[i];
        }
        StoreRefined(0, BuildVtensorType(merged, GetElementType(dtype)));
      }
    }
    // Softmax/LayerNorm: same shape as input.
    else if (op_type == "Softmax" || op_type == "LayerNormalization") {
      auto [_, dtype] = GetOrtOutputInfo(0);
      StoreRefined(0, BuildVtensorType(in_dims, GetElementType(dtype)));
    }
    // Split: same as input except split dim is divided.
    else if (op_type == "Split") {
      int64_t axis = GetIntAttr("axis", 0);
      if (axis < 0) axis += in_dims.size();
      // Try to resolve split sizes from input[1] (known constant).
      std::vector<int64_t> split_sizes;
      if (inputs.size() >= 2 && inputs[1]) {
        std::string splits_name = inputs[1].GetName();
        if (!splits_name.empty()) {
          auto it = known_int64_vectors_.find(splits_name);
          if (it != known_int64_vectors_.end()) {
            split_sizes = it->second;
          }
        }
      }
      for (size_t o = 0; o < outputs.size(); ++o) {
        if (output_names[o].empty() || !outputs[o]) continue;
        auto [ort_shape, dtype] = GetOrtOutputInfo(o);
        if (ort_shape.size() != in_dims.size()) continue;
        std::vector<int64_t> merged(ort_shape.begin(), ort_shape.end());
        for (size_t i = 0; i < merged.size(); ++i) {
          if ((int64_t)i == axis) {
            // Use split size if known.
            if (merged[i] < 0 && o < split_sizes.size() &&
                split_sizes[o] >= 0) {
              merged[i] = split_sizes[o];
            }
          } else if (merged[i] < 0 && in_dims[i] >= 0) {
            merged[i] = in_dims[i];
          }
        }
        StoreRefined(o, BuildVtensorType(merged, GetElementType(dtype)));
      }
    }
    // Concat: merge dims from all refined inputs.
    // For non-axis dims: use any input that has a known value.
    // For axis dim: sum all inputs' axis dims if all are known.
    else if (op_type == "Concat") {
      auto [ort_shape, dtype] = GetOrtOutputInfo(0);
      if (ort_shape.size() == in_dims.size()) {
        int64_t axis = GetIntAttr("axis", 0);
        if (axis < 0) axis += in_dims.size();
        std::vector<int64_t> merged(ort_shape.begin(), ort_shape.end());
        // Collect refined dims from all inputs.
        for (size_t inp_idx = 0; inp_idx < inputs.size(); ++inp_idx) {
          if (!inputs[inp_idx]) continue;
          std::string iname = inputs[inp_idx].GetName();
          if (iname.empty()) continue;
          std::string itype = GetVtensorType(iname, inputs[inp_idx].TypeInfo());
          std::vector<int64_t> idims;
          std::string ielem;
          if (!ParseVtensorType(itype, idims, ielem)) continue;
          if (idims.size() != merged.size()) continue;
          for (size_t i = 0; i < merged.size(); ++i) {
            if ((int64_t)i != axis && merged[i] < 0 && idims[i] >= 0) {
              merged[i] = idims[i];
            }
          }
        }
        // For the axis dim: sum all inputs if all have known axis sizes.
        if (merged[axis] < 0) {
          int64_t total = 0;
          bool all_known = true;
          for (size_t inp_idx = 0; inp_idx < inputs.size(); ++inp_idx) {
            if (!inputs[inp_idx]) { all_known = false; break; }
            std::string iname = inputs[inp_idx].GetName();
            if (iname.empty()) { all_known = false; break; }
            std::string itype = GetVtensorType(iname, inputs[inp_idx].TypeInfo());
            std::vector<int64_t> idims;
            std::string ielem;
            if (!ParseVtensorType(itype, idims, ielem) ||
                idims.size() != merged.size() || idims[axis] < 0) {
              all_known = false;
              break;
            }
            total += idims[axis];
          }
          if (all_known) merged[axis] = total;
        }
        StoreRefined(0, BuildVtensorType(merged, GetElementType(dtype)));
      }
    }
    // TopK: preserves all dims except the axis dim. Has 2 outputs.
    else if (op_type == "TopK") {
      int64_t axis = GetIntAttr("axis", -1);
      if (axis < 0) axis += in_dims.size();
      for (size_t o = 0; o < outputs.size(); ++o) {
        if (o >= output_names.size() || output_names[o].empty() || !outputs[o])
          continue;
        auto out_info = outputs[o].TypeInfo().GetTensorTypeAndShapeInfo();
        auto ort_shape = out_info.GetShape();
        auto dtype = out_info.GetElementType();
        if (ort_shape.size() == in_dims.size()) {
          std::vector<int64_t> merged(ort_shape.begin(), ort_shape.end());
          for (size_t i = 0; i < merged.size(); ++i) {
            if ((int64_t)i != axis && merged[i] < 0 && in_dims[i] >= 0) {
              merged[i] = in_dims[i];
            }
          }
          StoreRefined(o, BuildVtensorType(merged, GetElementType(dtype)));
        }
      }
    }
    // Expand: output shape is determined by the shape tensor (input[1]),
    // broadcasting with the input shape. Resolve shape tensor to get
    // known dims.
    else if (op_type == "Expand") {
      auto [ort_shape, dtype] = GetOrtOutputInfo(0);
      // Try to resolve the shape tensor (input[1]).
      if (inputs.size() >= 2 && inputs[1]) {
        std::string shape_name = inputs[1].GetName();
        if (!shape_name.empty()) {
          std::vector<int64_t> shape_vals;
          if (ResolveShapeVector(shape_name, shape_vals) &&
              shape_vals.size() == in_dims.size()) {
            // Compute broadcast output: for each dim, if shape_val > 0
            // use it; if shape_val is 1 use input dim; otherwise dynamic.
            std::vector<int64_t> merged(shape_vals.size());
            for (size_t i = 0; i < merged.size(); ++i) {
              if (shape_vals[i] > 1) {
                merged[i] = shape_vals[i];
              } else if (shape_vals[i] == 1) {
                // Expand with 1 keeps input dim.
                merged[i] = in_dims[i];
              } else {
                // Dynamic shape dim — use input dim if known and > 1,
                // otherwise stay dynamic.
                merged[i] = (in_dims[i] > 1) ? in_dims[i] : -1;
              }
            }
            StoreRefined(0, BuildVtensorType(merged, GetElementType(dtype)));
          } else if (ort_shape.size() == in_dims.size()) {
            // Fallback: propagate input dims where ORT output is unknown.
            std::vector<int64_t> merged(ort_shape.begin(), ort_shape.end());
            for (size_t i = 0; i < merged.size(); ++i) {
              if (merged[i] < 0 && in_dims[i] >= 0) merged[i] = in_dims[i];
            }
            StoreRefined(0, BuildVtensorType(merged, GetElementType(dtype)));
          }
        }
      }
    }
    // Unsqueeze/Squeeze: propagate matching dims.
    else if (op_type == "Unsqueeze" || op_type == "Squeeze") {
      auto [ort_shape, dtype] = GetOrtOutputInfo(0);
      if (ort_shape.size() == in_dims.size()) {
        std::vector<int64_t> merged(ort_shape.begin(), ort_shape.end());
        for (size_t i = 0; i < merged.size(); ++i) {
          if (merged[i] < 0 && in_dims[i] >= 0) merged[i] = in_dims[i];
        }
        StoreRefined(0, BuildVtensorType(merged, GetElementType(dtype)));
      }
    }
    // Gather: output shape = data shape with axis dim replaced by indices shape.
    // For axis=0: data[D0,D1,...] + indices[I0,...] → [I0,...,D1,...]
    // Propagate known static dims from data to output.
    else if (op_type == "Gather") {
      auto [ort_shape, dtype] = GetOrtOutputInfo(0);
      int64_t axis = 0;
      for (const auto& attr : attrs) {
        if (attr.GetName() == "axis") {
          int64_t val = 0;
          attr.GetValue(val);
          axis = val;
        }
      }
      if (axis < 0) axis += static_cast<int64_t>(in_dims.size());
      // Get indices shape.
      std::vector<int64_t> idx_dims;
      std::string idx_elem;
      if (inputs.size() >= 2 && inputs[1]) {
        std::string idx_type = GetVtensorType(inputs[1].GetName(),
                                              inputs[1].TypeInfo());
        ParseVtensorType(idx_type, idx_dims, idx_elem);
      }
      // Build expected output: indices_dims + data_dims (skipping axis).
      std::vector<int64_t> expected;
      for (int64_t i = 0; i < axis; ++i) expected.push_back(in_dims[i]);
      for (auto d : idx_dims) expected.push_back(d);
      for (size_t i = axis + 1; i < in_dims.size(); ++i)
        expected.push_back(in_dims[i]);
      if (expected.size() == ort_shape.size()) {
        std::vector<int64_t> merged(ort_shape.begin(), ort_shape.end());
        bool improved = false;
        for (size_t i = 0; i < merged.size(); ++i) {
          if (merged[i] < 0 && expected[i] >= 0) {
            merged[i] = expected[i];
            improved = true;
          }
        }
        if (improved) {
          StoreRefined(0, BuildVtensorType(merged, GetElementType(dtype)));
        }
      }
    }
    // DequantizeLinear: output has same shape as input, different dtype.
    else if (op_type == "DequantizeLinear") {
      auto [ort_shape, dtype] = GetOrtOutputInfo(0);
      if (ort_shape.size() == in_dims.size()) {
        std::vector<int64_t> merged(ort_shape.begin(), ort_shape.end());
        for (size_t i = 0; i < merged.size(); ++i) {
          if (merged[i] < 0 && in_dims[i] >= 0) merged[i] = in_dims[i];
        }
        StoreRefined(0, BuildVtensorType(merged, GetElementType(dtype)));
      }
    }
  }

  // Get the vtensor type string for a named value. Uses refined type if
  // available, otherwise falls back to ORT's type info.
  std::string GetVtensorType(const std::string& name,
                             const Ort::ConstTypeInfo& type_info) {
    auto it = refined_types_.find(name);
    if (it != refined_types_.end()) return it->second;
    return FormatTensorType(type_info);
  }

  std::string RefineReshapeOutputType(const Ort::ConstNode& node) {
    auto inputs = node.GetInputs();
    if (inputs.size() < 2 || !inputs[1]) return "";

    auto outputs = node.GetOutputs();
    if (outputs.empty() || !outputs[0]) return "";

    // Get the shape tensor name.
    std::string shape_name = inputs[1].GetName();

    // Try to resolve the shape vector to concrete values.
    std::vector<int64_t> shape_values;
    if (!ResolveShapeVector(shape_name, shape_values)) return "";

    // Check if we actually improved anything.
    auto output_info = outputs[0].TypeInfo().GetTensorTypeAndShapeInfo();
    auto ort_shape = output_info.GetShape();
    if (ort_shape.size() != shape_values.size()) return "";

    bool improved = false;
    for (size_t i = 0; i < ort_shape.size(); ++i) {
      if (ort_shape[i] < 0 && shape_values[i] >= 0) {
        improved = true;
        break;
      }
    }
    if (!improved) return "";

    // Build refined type string.
    auto dtype = output_info.GetElementType();
    std::ostringstream ss;
    ss << "!torch.vtensor<[";
    for (size_t i = 0; i < shape_values.size(); ++i) {
      if (i > 0) ss << ",";
      if (shape_values[i] < 0) {
        // Use ORT's value if it knew it, otherwise dynamic.
        if (ort_shape[i] >= 0) {
          ss << ort_shape[i];
        } else {
          ss << "?";
        }
      } else {
        ss << shape_values[i];
      }
    }
    ss << "]," << GetElementType(dtype) << ">";
    std::string result = ss.str();

    // Store the refined type so downstream ops use the correct type.
    std::string output_name = outputs[0].GetName();
    if (!output_name.empty()) {
      refined_types_[output_name] = result;
    }
    return result;
  }

  // Try to resolve a tensor name to a vector of int64 values.
  // Returns true if we could determine at least some values.
  // Values of -1 mean "dynamic/unknown".
  // If unknown_count is provided, it is set to the number of truly
  // unresolvable values (vs -1 from known constants like ONNX infer).
  bool ResolveShapeVector(const std::string& name,
                          std::vector<int64_t>& values,
                          int* unknown_count = nullptr) {
    if (unknown_count) *unknown_count = 0;

    // Case 1: Direct constant initializer.
    auto it = known_int64_vectors_.find(name);
    if (it != known_int64_vectors_.end()) {
      values = it->second;
      return true;
    }

    // Case 2: Produced by a known op — resolve through producer chain.
    auto prod_it = producer_info_.find(name);
    if (prod_it != producer_info_.end()) {
      const auto& info = prod_it->second;
      if (info.op_type == "Concat") {
        values.clear();
        for (const auto& input_name : info.input_names) {
          auto inp_it = known_int64_vectors_.find(input_name);
          if (inp_it != known_int64_vectors_.end()) {
            // Known constant — append all values.
            values.insert(values.end(), inp_it->second.begin(),
                          inp_it->second.end());
          } else {
            // Unknown — mark as dynamic. Assume it contributes 1 element
            // (most shape concat inputs are scalar-as-1d).
            values.push_back(-1);
            if (unknown_count) (*unknown_count)++;
          }
        }
        return !values.empty();
      }
      // Reshape with [-1] is identity for 1-D shape tensors — trace input.
      if (info.op_type == "Reshape" && !info.input_names.empty()) {
        return ResolveShapeVector(info.input_names[0], values, unknown_count);
      }
      // Where(cond, true_val, false_val): trace the false branch.
      // This handles the ONNX Expand shape pattern:
      //   Where(Equal(shape, -1), ones, shape) → shape (when no -1 dims).
      if (info.op_type == "Where" && info.input_names.size() >= 3) {
        return ResolveShapeVector(info.input_names[2], values, unknown_count);
      }
    }

    return false;
  }

  // Build maps for shape refinement. Called after CollectMetadata().
  void BuildShapeRefinementMaps() {
    // Read constant values from small int64 initializers.
    for (const auto& init : initializers_) {
      auto type_info = init.TypeInfo();
      if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) continue;

      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        continue;

      size_t count = tensor_info.GetElementCount();
      if (count > 16) continue;  // Only track small tensors.

      Ort::ConstValue tensor_value{nullptr};
      auto status = init.GetInitializer(tensor_value);
      if (!status.IsOK()) continue;

      const auto* data =
          static_cast<const int64_t*>(tensor_value.GetTensorRawData());
      std::vector<int64_t> vals(data, data + count);
      known_int64_vectors_[init.GetName()] = std::move(vals);
    }

    // Build producer map and value-info lookup from graph nodes.
    auto nodes = graph_.GetNodes();
    // Map output name → ConstValueInfo for ORT type info lookups.
    std::unordered_map<std::string, Ort::ConstValueInfo> value_info_map;

    for (const auto& node : nodes) {
      std::string op_type = node.GetOperatorType();
      auto inputs = node.GetInputs();
      auto outputs = node.GetOutputs();

      // Record producer info for all outputs.
      ProducerInfo info;
      info.op_type = op_type;
      for (const auto& inp : inputs) {
        if (inp) info.input_names.push_back(inp.GetName());
      }
      for (const auto& out : outputs) {
        if (out) {
          std::string out_name = out.GetName();
          if (!out_name.empty()) {
            producer_info_[out_name] = info;
            value_info_map.emplace(out_name, out);
          }
        }
      }
    }

    // Also add graph inputs and initializers to value_info_map.
    for (const auto& gi : graph_inputs_) {
      if (gi) value_info_map.emplace(gi.GetName(), gi);
    }
    for (const auto& init : initializers_) {
      if (init) value_info_map.emplace(init.GetName(), init);
    }

    // Multi-pass: propagate known values through the graph until convergence.
    // Handles chains like: Constant → Gather(Shape(tensor), idx) → Unsqueeze → Concat
    for (int pass = 0; pass < 3; ++pass) {
      size_t prev_size = known_int64_vectors_.size();

      for (const auto& node : nodes) {
        std::string op_type = node.GetOperatorType();
        auto inputs = node.GetInputs();
        auto outputs = node.GetOutputs();

        // Unsqueeze: propagate known scalar.
        if (op_type == "Unsqueeze" && inputs.size() >= 1 && inputs[0]) {
          auto it = known_int64_vectors_.find(inputs[0].GetName());
          if (it != known_int64_vectors_.end() && it->second.size() == 1) {
            for (const auto& out : outputs) {
              if (out && !out.GetName().empty()) {
                known_int64_vectors_.emplace(out.GetName(), it->second);
              }
            }
          }
        }

        // Gather(Shape(tensor), constant_index) → extract specific dim.
        if (op_type == "Gather" && inputs.size() >= 2 &&
            inputs[0] && inputs[1]) {
          std::string shape_out = inputs[0].GetName();
          std::string index_name = inputs[1].GetName();

          auto idx_it = known_int64_vectors_.find(index_name);
          if (idx_it != known_int64_vectors_.end() &&
              idx_it->second.size() == 1) {
            int64_t gather_idx = idx_it->second[0];

            // Check if input[0] was produced by a Shape op.
            auto shape_prod = producer_info_.find(shape_out);
            if (shape_prod != producer_info_.end() &&
                shape_prod->second.op_type == "Shape" &&
                !shape_prod->second.input_names.empty()) {
              std::string shape_src = shape_prod->second.input_names[0];
              auto vi_it = value_info_map.find(shape_src);
              if (vi_it != value_info_map.end()) {
                auto src_shape = vi_it->second.TypeInfo()
                                     .GetTensorTypeAndShapeInfo()
                                     .GetShape();
                if (gather_idx < 0)
                  gather_idx += static_cast<int64_t>(src_shape.size());
                if (gather_idx >= 0 &&
                    gather_idx < static_cast<int64_t>(src_shape.size())) {
                  int64_t dim_val = src_shape[gather_idx];
                  if (dim_val >= 0) {
                    for (const auto& out : outputs) {
                      if (out && !out.GetName().empty()) {
                        known_int64_vectors_.emplace(out.GetName(),
                                                     std::vector<int64_t>{dim_val});
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Stop early if no new values were discovered.
      if (known_int64_vectors_.size() == prev_size) break;
    }

    // Pre-compute refined types for all nodes so they're available when
    // emitting the function signature (which needs output types before the
    // function body is emitted).
    for (const auto& node : nodes) {
      std::string op_type = node.GetOperatorType();
      auto inputs = node.GetInputs();
      auto outputs = node.GetOutputs();
      auto attrs = node.GetAttributes();

      // First try Reshape refinement.
      if (op_type == "Reshape") {
        RefineReshapeOutputType(node);
      }

      // Build output names and type strings.
      std::vector<std::string> output_names;
      std::vector<std::string> output_type_strs;
      for (const auto& out : outputs) {
        if (out) {
          std::string name = out.GetName();
          output_names.push_back(name);
          output_type_strs.push_back(
              GetVtensorType(name, out.TypeInfo()));
        } else {
          output_names.push_back("");
          output_type_strs.push_back("");
        }
      }

      // Convert inputs/outputs to ConstValueInfo vectors.
      std::vector<Ort::ConstValueInfo> inp_infos(inputs.begin(), inputs.end());
      std::vector<Ort::ConstValueInfo> out_infos(outputs.begin(), outputs.end());

      PropagateRefinedTypes(op_type, inp_infos, out_infos, attrs,
                            output_names, output_type_strs);
    }
  }

  // Emits a Reduce op with axes as an attribute instead of a tensor input.
  // Converts opset 18+ format (axes as input[1]) to opset 17 format (axes as
  // attribute) for compatibility with torch-mlir's ONNX conversion patterns.
  OrtStatus* EmitReduceWithAxesAttr(const Ort::ConstNode& node,
                                    const std::string& op_type,
                                    const std::vector<int64_t>& axes) {
    auto inputs = node.GetInputs();
    auto outputs = node.GetOutputs();
    auto attrs = node.GetAttributes();

    // Build output name and type.
    std::string out_name = SanitizeName(outputs[0].GetName());
    std::string out_type = GetVtensorType(outputs[0].GetName(),
                                          outputs[0].TypeInfo());

    // Build data input reference (input[0] only, skip axes tensor).
    std::string data_name = SanitizeName(inputs[0].GetName());
    std::string data_type = GetVtensorType(inputs[0].GetName(),
                                           inputs[0].TypeInfo());

    // Build attributes: keep existing attrs (keepdims, etc.) but add axes
    // and skip noop_with_empty_axes (opset 18+ only).
    std::ostringstream attr_ss;
    bool first = true;
    for (const auto& attr : attrs) {
      std::string name = attr.GetName();
      if (name == "noop_with_empty_axes") continue;  // opset 18+ only
      if (!first) attr_ss << ", ";
      first = false;
      attr_ss << FormatAttribute(attr);
    }
    // Add axes as attribute.
    if (!first) attr_ss << ", ";
    std::ostringstream axes_ss;
    axes_ss << "torch.onnx.axes = [";
    for (size_t i = 0; i < axes.size(); ++i) {
      if (i > 0) axes_ss << ", ";
      axes_ss << axes[i] << " : si64";
    }
    axes_ss << "]";
    attr_ss << axes_ss.str();

    out_ << std::format(
        "    %{0} = torch.operator \"onnx.{1}\"(%{2}) {{{3}}} : ({4}) -> {5}\n",
        out_name, op_type, data_name, attr_ss.str(), data_type, out_type);

    return nullptr;
  }

  // Emits a reshape operation using torch.aten.reshape.
  // Emits torch.constant.int for each dimension (using -1 for dynamic dims),
  // torch.prim.ListConstruct for the shape list, and torch.aten.reshape.
  void EmitReshapeFromDims(const std::string& result_name,
                           const std::string& input_name,
                           const std::string& input_type,
                           const std::vector<int64_t>& target_dims,
                           const std::string& result_type) {
    std::vector<std::string> dim_names;
    for (size_t i = 0; i < target_dims.size(); ++i) {
      std::string dname = result_name + "_d" + std::to_string(i);
      int64_t val = target_dims[i] < 0 ? -1 : target_dims[i];
      out_ << std::format("    %{} = torch.constant.int {}\n", dname, val);
      dim_names.push_back(dname);
    }
    std::ostringstream refs, types;
    for (size_t i = 0; i < dim_names.size(); ++i) {
      if (i > 0) { refs << ", "; types << ", "; }
      refs << "%" << dim_names[i];
      types << "!torch.int";
    }
    std::string list_name = result_name + "_shape";
    out_ << std::format(
        "    %{0} = torch.prim.ListConstruct {1} : ({2}) -> !torch.list<int>\n",
        list_name, refs.str(), types.str());
    out_ << std::format(
        "    %{0} = torch.aten.reshape %{1}, %{2} : {3}, !torch.list<int> -> {4}\n",
        result_name, input_name, list_name, input_type, result_type);
  }

  // Decomposes DequantizeLinear with block_size into arithmetic ops.
  //
  // DequantizeLinear(x, scale, zero_point, axis=A, block_size=B):
  //   x:     [..., K, ...] uint8 (one INT4 value per byte)
  //   scale: [..., G, ...] f16   where G = K / B
  //   zero:  [..., G, ...] uint8
  //   out:   [..., K, ...] f16
  //
  // Decomposition:
  //   1. Reshape x:      [..., K, ...]     → [..., G, B, ...]
  //   2. Cast x:         uint8             → f16
  //   3. Unsqueeze zero: [..., G, ...]     → [..., G, 1, ...]
  //   4. Cast zero:      uint8             → f16
  //   5. Sub:            x_f16 - zero_f16  (broadcasts along block dim)
  //   6. Unsqueeze scale:[..., G, ...]     → [..., G, 1, ...]
  //   7. Mul:            diff * scale      (broadcasts along block dim)
  //   8. Reshape result: [..., G, B, ...]  → [..., K, ...]
  OrtStatus* EmitBlockedDequantizeLinear(const Ort::ConstNode& node) {
    auto inputs = node.GetInputs();
    auto outputs = node.GetOutputs();
    auto attrs = node.GetAttributes();

    // Parse attributes.
    int64_t axis = 1;  // ONNX DequantizeLinear default
    int64_t block_size = 0;
    for (const auto& attr : attrs) {
      std::string name = attr.GetName();
      if (name == "axis") attr.GetValue(axis);
      else if (name == "block_size") attr.GetValue(block_size);
    }

    // Get input/output names.
    std::string x_ssa = SanitizeName(inputs[0].GetName());
    std::string s_ssa = SanitizeName(inputs[1].GetName());
    bool has_zp = inputs.size() > 2 && inputs[2] &&
                  !inputs[2].GetName().empty();
    std::string z_ssa = has_zp ? SanitizeName(inputs[2].GetName()) : "";
    std::string out_ssa = SanitizeName(outputs[0].GetName());

    // Get types (using refined types if available).
    std::string x_type = GetVtensorType(inputs[0].GetName(),
                                        inputs[0].TypeInfo());
    std::string s_type = GetVtensorType(inputs[1].GetName(),
                                        inputs[1].TypeInfo());
    std::string z_type = has_zp
        ? GetVtensorType(inputs[2].GetName(), inputs[2].TypeInfo())
        : "";

    // Parse shapes.
    std::vector<int64_t> x_dims, s_dims, z_dims;
    std::string x_elem, s_elem, z_elem;
    if (!ParseVtensorType(x_type, x_dims, x_elem))
      return MakeError("DequantizeLinear: failed to parse input type: {}",
                       x_type);
    if (!ParseVtensorType(s_type, s_dims, s_elem))
      return MakeError("DequantizeLinear: failed to parse scale type: {}",
                       s_type);
    if (has_zp && !ParseVtensorType(z_type, z_dims, z_elem))
      return MakeError("DequantizeLinear: failed to parse zero_point type: {}",
                       z_type);

    // Get output element type.
    auto out_info = outputs[0].TypeInfo().GetTensorTypeAndShapeInfo();
    auto out_dtype_enum = out_info.GetElementType();
    std::string out_elem = GetElementType(out_dtype_enum);
    // ORT's ONNXTensorElementDataType matches ONNX's TensorProto.DataType.
    int64_t onnx_dtype_int = static_cast<int64_t>(out_dtype_enum);

    // Normalize axis.
    int64_t rank = static_cast<int64_t>(x_dims.size());
    if (axis < 0) axis += rank;

    // Get K (size along quantized axis) and compute G = K / block_size.
    int64_t K = x_dims[axis];
    if (K <= 0)
      return MakeError(
          "DequantizeLinear: quantized axis {} has dynamic size", axis);
    if (K % block_size != 0)
      return MakeError(
          "DequantizeLinear: axis size {} not divisible by block_size {}",
          K, block_size);
    int64_t G = K / block_size;

    // Unique prefix for intermediate SSA names.
    std::string p = out_ssa + "_bdq_";

    // --- Compute intermediate shapes ---

    // x_4d: split quantized axis into [G, block_size].
    std::vector<int64_t> x_4d_dims(x_dims);
    x_4d_dims[axis] = G;
    x_4d_dims.insert(x_4d_dims.begin() + axis + 1, block_size);

    // s_4d: unsqueeze scale at axis+1.
    std::vector<int64_t> s_4d_dims(s_dims);
    s_4d_dims.insert(s_4d_dims.begin() + axis + 1, 1);

    // --- Build intermediate type strings ---
    std::string x_4d_type = BuildVtensorType(x_4d_dims, x_elem);
    std::string x_4d_cast_type = BuildVtensorType(x_4d_dims, out_elem);
    std::string s_4d_type = BuildVtensorType(s_4d_dims, s_elem);

    // --- Step 1: Reshape x to [..., G, block_size, ...] ---
    EmitReshapeFromDims(p + "x4d", x_ssa, x_type, x_4d_dims, x_4d_type);

    // --- Step 2: Cast x from uint8 to output dtype (e.g., f16) ---
    out_ << std::format(
        "    %{0} = torch.operator \"onnx.Cast\"(%{1}) "
        "{{torch.onnx.to = {2} : si64}} : ({3}) -> {4}\n",
        p + "xf", p + "x4d", onnx_dtype_int, x_4d_type, x_4d_cast_type);

    std::string arith_input = p + "xf";

    // --- Steps 3-5: Handle zero point (if present) ---
    if (has_zp) {
      // Unsqueeze zero point at axis+1.
      std::vector<int64_t> z_4d_dims(z_dims);
      z_4d_dims.insert(z_4d_dims.begin() + axis + 1, 1);
      std::string z_4d_type = BuildVtensorType(z_4d_dims, z_elem);
      std::string z_4d_cast_type = BuildVtensorType(z_4d_dims, out_elem);

      out_ << std::format("    %{0} = torch.constant.int {1}\n",
                          p + "z_dim", axis + 1);
      out_ << std::format(
          "    %{0} = torch.aten.unsqueeze %{1}, %{2} : {3}, "
          "!torch.int -> {4}\n",
          p + "z4d", z_ssa, p + "z_dim", z_type, z_4d_type);

      // Cast zero from uint8 to output dtype.
      out_ << std::format(
          "    %{0} = torch.operator \"onnx.Cast\"(%{1}) "
          "{{torch.onnx.to = {2} : si64}} : ({3}) -> {4}\n",
          p + "zf", p + "z4d", onnx_dtype_int, z_4d_type, z_4d_cast_type);

      // Subtract: x - zero (broadcasts along block dim).
      out_ << std::format(
          "    %{0} = torch.operator \"onnx.Sub\"(%{1}, %{2}) {{}} : "
          "({3}, {4}) -> {5}\n",
          p + "diff", p + "xf", p + "zf",
          x_4d_cast_type, z_4d_cast_type, x_4d_cast_type);

      arith_input = p + "diff";
    }

    // --- Step 6: Unsqueeze scale at axis+1 ---
    out_ << std::format("    %{0} = torch.constant.int {1}\n",
                        p + "s_dim", axis + 1);
    out_ << std::format(
        "    %{0} = torch.aten.unsqueeze %{1}, %{2} : {3}, "
        "!torch.int -> {4}\n",
        p + "s4d", s_ssa, p + "s_dim", s_type, s_4d_type);

    // --- Step 7: Multiply diff * scale (broadcasts along block dim) ---
    out_ << std::format(
        "    %{0} = torch.operator \"onnx.Mul\"(%{1}, %{2}) {{}} : "
        "({3}, {4}) -> {5}\n",
        p + "result", arith_input, p + "s4d",
        x_4d_cast_type, s_4d_type, x_4d_cast_type);

    // --- Step 8: Reshape back to original shape ---
    auto ref_it = refined_types_.find(outputs[0].GetName());
    std::string final_type = ref_it != refined_types_.end()
        ? ref_it->second
        : FormatTensorType(outputs[0].TypeInfo());
    std::vector<int64_t> out_dims;
    std::string out_e;
    ParseVtensorType(final_type, out_dims, out_e);

    EmitReshapeFromDims(out_ssa, p + "result", x_4d_cast_type,
                        out_dims, final_type);

    return nullptr;
  }

  struct ProducerInfo {
    std::string op_type;
    std::vector<std::string> input_names;
  };

  // Member variables.
  const Ort::ConstGraph& graph_;
  std::ostream& out_;
  std::string irpa_path_;
  TargetConfig target_config_;

  std::string graph_name_;
  int64_t ir_version_ = 8;
  int64_t opset_version_ = 17;

  std::vector<Ort::ConstValueInfo> graph_inputs_;
  std::vector<Ort::ConstValueInfo> graph_outputs_;
  std::vector<Ort::ConstValueInfo> initializers_;
  std::vector<ParameterInitializer> parameter_initializers_;

  // Shape refinement maps.
  std::unordered_map<std::string, std::vector<int64_t>> known_int64_vectors_;
  std::unordered_map<std::string, ProducerInfo> producer_info_;
  // Refined vtensor type strings for outputs whose shapes we've improved.
  std::unordered_map<std::string, std::string> refined_types_;

  // Extern dispatch state.
  int extern_id_ = 0;

  // Declared util.global names and their tensor types.
  std::unordered_map<std::string, std::string> declared_globals_;

  // Counter for unique constant shape tensor names.
  int const_shape_counter_ = 0;

  // Maps Shape op output name → {source tensor name, source tensor type}.
  // Used to rewrite Gather(Shape(tensor), idx) → torch.aten.size.int.
  std::unordered_map<std::string, std::pair<std::string, std::string>>
      shape_source_map_;

  // Maps Gather output name → size.int SSA name (the !torch.int result).
  // Populated when Gather(Shape(tensor), idx) is rewritten to size.int.
  std::unordered_map<std::string, std::string> size_int_ssa_map_;
};

// Builds an IRPA parameter archive for large initializers.
//
// Large inline initializers are copied into an IRPA (IREE Parameter Archive)
// file on disk. We write to an IRPA file rather than keeping data in memory
// because ORT does not guarantee that initializer tensor data remains valid
// beyond the Compile() call. By persisting to disk via IRPA, the data is
// accessed at runtime through IREE's parameter index with file-backed entries.
//
// External initializers (already backed by external files) are added to the
// parameter index directly, pointing to their original files without copying.
//
// The resulting parameter provider is registered with the IREE session so that
// the compiled module can resolve #flow.parameter.named references at runtime.
OrtStatus* MlirGenerator::BuildParameterArchive(
    ParameterIndexPtr& out_index, ParameterProviderPtr& out_provider) {
  if (parameter_initializers_.empty()) {
    return nullptr;
  }

  iree_allocator_t allocator = iree_allocator_system();

  // Build source index from ORT tensor data wrapped as file handles.
  // The tensor data is valid for the duration of this call (we are inside
  // CompileImpl). iree_io_build_parameter_archive copies it to the IRPA file.
  ParameterIndexPtr source_index;
  IREE_ORT_RETURN_IF_ERROR(
      iree_io_parameter_index_create(allocator, source_index.ForOutput()));

  for (const auto& param : parameter_initializers_) {
    const auto& init = initializers_[param.initializer_index];

    // Skip external initializers — added to target index later.
    // Note: GetExternalInitializerInfo returns OK with null output for
    // non-external initializers, so we must check both status and pointer.
    Ort::ExternalInitializerInfo ext_info(nullptr);
    ORT_RETURN_IF_ERROR(init.GetExternalInitializerInfo(ext_info).release());
    if (ext_info) {
      continue;
    }

    Ort::ConstValue tensor(nullptr);
    auto status = init.GetInitializer(tensor);
    if (!status.IsOK()) {
      return MakeError("Failed to get initializer: {}", init.GetName());
    }

    auto* data = const_cast<uint8_t*>(
        static_cast<const uint8_t*>(tensor.GetTensorRawData()));
    size_t size = tensor.GetTensorSizeInBytes();

    FileHandlePtr handle;
    iree_byte_span_t span = {data, static_cast<iree_host_size_t>(size)};
    IREE_ORT_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ, span,
        iree_io_file_handle_release_callback_null(), allocator,
        handle.ForOutput()));

    iree_io_parameter_index_entry_t entry = {};
    entry.key = iree_make_string_view(param.sanitized_name.data(),
                                      param.sanitized_name.size());
    entry.length = size;
    entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
    entry.storage.file.handle = handle.Get();
    entry.storage.file.offset = 0;
    IREE_ORT_RETURN_IF_ERROR(
        iree_io_parameter_index_add(source_index.Get(), &entry));
  }

  // Build IRPA archive from source index.
  ParameterIndexPtr target_index;
  IREE_ORT_RETURN_IF_ERROR(
      iree_io_parameter_index_create(allocator, target_index.ForOutput()));

  if (iree_io_parameter_index_count(source_index.Get()) > 0) {
    iree_io_parameter_archive_file_open_callback_t file_open = {
        IrpaFileOpenCallback,
        const_cast<std::string*>(&irpa_path_),
    };
    IREE_ORT_RETURN_IF_ERROR(iree_io_build_parameter_archive(
        source_index.Get(), target_index.Get(), file_open, 0, allocator));
  }

  // Add external initializer entries directly to target index.
  for (const auto& param : parameter_initializers_) {
    const auto& init = initializers_[param.initializer_index];

    Ort::ExternalInitializerInfo ext_info(nullptr);
    ORT_RETURN_IF_ERROR(init.GetExternalInitializerInfo(ext_info).release());
    if (!ext_info) {
      continue;
    }

    FileHandlePtr ext_handle;
    // External data paths are relative to the model directory.
    std::filesystem::path model_dir =
        std::filesystem::path(graph_.GetModelPath()).parent_path();
    std::string filepath = (model_dir / ext_info.GetFilePath()).string();
    IREE_ORT_RETURN_IF_ERROR(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ,
        iree_make_string_view(filepath.data(), filepath.size()), allocator,
        ext_handle.ForOutput()));

    // Compute byte size from tensor shape if external data lacks length field.
    size_t byte_size = ext_info.GetByteSize();
    if (byte_size == 0) {
      auto tensor_info = init.TypeInfo().GetTensorTypeAndShapeInfo();
      byte_size = tensor_info.GetElementCount() *
                  OnnxElementTypeSize(tensor_info.GetElementType());
    }

    iree_io_parameter_index_entry_t entry = {};
    entry.key = iree_make_string_view(param.sanitized_name.data(),
                                      param.sanitized_name.size());
    entry.length = byte_size;
    entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
    entry.storage.file.handle = ext_handle.Get();
    entry.storage.file.offset = static_cast<uint64_t>(ext_info.GetFileOffset());
    IREE_ORT_RETURN_IF_ERROR(
        iree_io_parameter_index_add(target_index.Get(), &entry));
  }

  ParameterProviderPtr provider;
  IREE_ORT_RETURN_IF_ERROR(iree_io_parameter_index_provider_create(
      iree_make_cstring_view("model"), target_index.Get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      allocator, provider.ForOutput()));

  out_index = std::move(target_index);
  out_provider = std::move(provider);
  return nullptr;
}

}  // namespace

/*static*/
TargetConfig TargetConfig::Create(const std::string& target_arch,
                                  const std::string& backend) {
  TargetConfig config;
  config.target_arch = target_arch;
  config.backend = backend;
  if (backend == "hip") {
    config.hal_backend = "rocm";
    config.hal_format = "rocm-hsaco-fb";
  } else if (backend == "cuda") {
    config.hal_backend = "cuda";
    config.hal_format = "cuda-nvptx-fb";
  } else if (backend == "vulkan") {
    config.hal_backend = "vulkan-spirv";
    config.hal_format = "vulkan-spirv-fb";
  }
  return config;
}

OrtStatus* GenerateMlir(const Ort::ConstGraph& graph, const OrtApi& /*ort_api*/,
                        const std::string& mlir_path,
                        const std::string& irpa_path,
                        ParameterIndexPtr& out_index,
                        ParameterProviderPtr& out_provider,
                        TargetConfig target_config) {
  std::ofstream file(mlir_path);
  if (!file.is_open()) {
    return MakeError("Failed to open output file: {}", mlir_path);
  }

  MlirGenerator gen(graph, file, irpa_path, std::move(target_config));
  OrtStatus* gen_status = gen.Generate();
  if (gen_status) return gen_status;

  file.close();
  if (file.fail()) {
    return MakeError("Failed to write to file: {}", mlir_path);
  }

  return gen.BuildParameterArchive(out_index, out_provider);
}

}  // namespace onnxruntime::iree

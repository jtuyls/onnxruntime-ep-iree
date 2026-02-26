//===- iree_ep.cc ---------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements IREE-based compilation and execution for ONNX models.
// Uses IREE runtime API for loading VMFB modules and executing functions.
//
//===----------------------------------------------------------------------===//

#include "iree_ep.h"

#include <filesystem>
#include <fstream>
#include <vector>

#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/modules/io/parameters/module.h"
#include "iree/runtime/api.h"
#include "iree_compile.h"
#include "iree_ep_factory.h"
#include "iree_ort_utils.h"
#include "mlir_gen.h"
#include "temp_file.h"

namespace onnxruntime::iree {

static std::vector<std::string> GenerateCompileFlags(
    const IreeEp::Config& config) {
  std::vector<std::string> flags;

  if (config.backend == "llvm-cpu") {
    flags.push_back("--iree-hal-target-device=local");
    flags.push_back("--iree-hal-local-target-device-backends=llvm-cpu");
    flags.push_back("--iree-llvmcpu-target-cpu=host");
    if (config.opt_level == "O2" || config.opt_level == "O3") {
      flags.push_back("--iree-opt-data-tiling");
    }
  } else if (config.backend == "vulkan") {
    flags.push_back("--iree-hal-target-device=vulkan");
  } else if (config.backend == "cuda") {
    flags.push_back("--iree-hal-target-device=cuda");
    if (!config.target_arch.empty()) {
      flags.push_back("--iree-cuda-target=" + config.target_arch);
    }
  } else if (config.backend == "hip") {
    flags.push_back("--iree-hal-target-device=hip");
    if (!config.target_arch.empty()) {
      flags.push_back("--iree-hip-target=" + config.target_arch);
    }
  } else {
    assert(false && "Unsupported backend, should have been caught earlier");
  }

  flags.push_back("--iree-opt-level=" + config.opt_level);
  return flags;
}

// ============================================================================
// IreeEp Implementation
// ============================================================================

IreeEp::IreeEp(IreeEpFactory& factory, const std::string& name,
               const Config& config, const OrtLogger& logger,
               uint32_t device_id)
    : OrtEp{},
      ApiPtrs(static_cast<const ApiPtrs&>(factory)),
      factory_(factory),
      name_(name),
      config_(config),
      logger_(&logger),
      device_id_(device_id) {
  // Set ORT version we support.
  ort_version_supported = ORT_API_VERSION;

  // Set function pointers for EP interface.
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
}

IreeEp::~IreeEp() {
  // Note: Avoid using logger during cleanup - ORT logging infrastructure may
  // be torn down before EP destructors run during Python interpreter shutdown.
}

iree_hal_device_t* IreeEp::IreeDevice() const {
  return factory_.GetDeviceForId(device_id_);
}

iree_runtime_instance_t* IreeEp::IreeInstance() const {
  return factory_.IreeInstance();
}

/*static*/
const char* ORT_API_CALL IreeEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const IreeEp*>(this_ptr);
  return ep->name_.c_str();
}

// Returns true if the node is an EPContext node created by this EP.
static bool IsIreeEpContextNode(const Ort::ConstNode& node) {
  return node.GetOperatorType() == kEpContextOpType &&
         node.GetDomain() == kEpContextDomain && [&]() {
           Ort::ConstOpAttr source_attr;
           auto status = node.GetAttributeByName("source", source_attr);
           if (!status.IsOK() || !source_attr) return false;
           std::string source;
           status = source_attr.GetValue<std::string>(source);
           return status.IsOK() && source == kEpContextSource;
         }();
}

/*static*/
OrtStatus* ORT_API_CALL
IreeEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                          OrtEpGraphSupportInfo* graph_support_info) noexcept {
  auto* ep = static_cast<IreeEp*>(this_ptr);

  // Use the C++ wrapper for easier API access.
  Ort::ConstGraph graph{ort_graph};

  // Get all nodes in the graph.
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.empty()) {
    return nullptr;  // Empty graph, nothing to claim.
  }

  // Check if this is a pre-compiled EPContext model (loading path).
  // An EPContext model has a single EPContext node created by this EP.
  if (nodes.size() == 1 && IsIreeEpContextNode(nodes[0])) {
    ORT_CXX_LOG_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                         "IREE EP: Detected EPContext model (loading path)");

    std::vector<const OrtNode*> nodes_to_fuse = {nodes[0]};
    OrtNodeFusionOptions node_fusion_options = {};
    node_fusion_options.ort_version_supported = ORT_API_VERSION;
    // EPContext model has no initializers to drop.
    node_fusion_options.drop_constant_initializers = false;

    return ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
        graph_support_info, nodes_to_fuse.data(), nodes_to_fuse.size(),
        &node_fusion_options);
  }

  // Normal path: claim the entire graph for compilation.
  std::vector<const OrtNode*> nodes_to_fuse;
  nodes_to_fuse.reserve(nodes.size());
  for (const auto& node : nodes) {
    nodes_to_fuse.push_back(node);
  }

  // Create fusion options for compiling EP.
  OrtNodeFusionOptions node_fusion_options = {};
  node_fusion_options.ort_version_supported = ORT_API_VERSION;

  // Drop constant initializers - EP will save them during Compile().
  // This reduces memory usage and allows weight preprocessing.
  node_fusion_options.drop_constant_initializers = true;

  // Register all nodes as a single fused subgraph.
  return ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
      graph_support_info, nodes_to_fuse.data(), nodes_to_fuse.size(),
      &node_fusion_options);
}

// Reads a binary file into a byte vector.
static OrtStatus* ReadFileToBytes(const std::string& path,
                                  std::vector<uint8_t>& out) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    return Ort::Status(("IREE EP: Failed to open file: " + path).c_str(),
                       ORT_FAIL)
        .release();
  }
  std::streampos pos = stream.tellg();
  if (pos == std::streampos(-1) || pos <= 0) {
    return Ort::Status(
               ("IREE EP: Failed to determine file size: " + path).c_str(),
               ORT_FAIL)
        .release();
  }
  auto size = static_cast<size_t>(pos);
  stream.seekg(0, std::ios::beg);
  out.resize(size);
  if (!stream.read(reinterpret_cast<char*>(out.data()),
                   static_cast<std::streamsize>(size))) {
    return Ort::Status(("IREE EP: Failed to read file: " + path).c_str(),
                       ORT_FAIL)
        .release();
  }
  return nullptr;
}

// Copies a file from src to dst.
static OrtStatus* CopyFile(const std::string& src, const std::string& dst) {
  std::error_code ec;
  std::filesystem::copy_file(
      src, dst, std::filesystem::copy_options::overwrite_existing, ec);
  if (ec) {
    return Ort::Status(("IREE EP: Failed to copy " + src + " to " + dst + ": " +
                        ec.message())
                           .c_str(),
                       ORT_FAIL)
        .release();
  }
  return nullptr;
}

// Reads a string attribute from an EPContext node.
static OrtStatus* ReadEpContextStringAttr(const Ort::ConstNode& node,
                                          const char* attr_name,
                                          std::string& out) {
  Ort::ConstOpAttr attr;
  auto status = node.GetAttributeByName(attr_name, attr);
  if (!status.IsOK()) {
    return Ort::Status(
               ("IREE EP: Failed to get attribute '" + std::string(attr_name) +
                "': " + status.GetErrorMessage())
                   .c_str(),
               ORT_FAIL)
        .release();
  }
  if (!attr) {
    return Ort::Status(("IREE EP: Missing required attribute '" +
                        std::string(attr_name) + "'")
                           .c_str(),
                       ORT_FAIL)
        .release();
  }
  status = attr.GetValue<std::string>(out);
  if (!status.IsOK()) {
    return Ort::Status(
               ("IREE EP: Failed to read attribute '" + std::string(attr_name) +
                "': " + status.GetErrorMessage())
                   .c_str(),
               ORT_FAIL)
        .release();
  }
  return nullptr;
}

// Loads IRPA from file and creates parameter index + provider.
static OrtStatus* LoadIrpaFromFile(const std::string& irpa_path,
                                   ParameterIndexPtr& out_index,
                                   ParameterProviderPtr& out_provider) {
  iree_allocator_t allocator = iree_allocator_system();

  // Open the IRPA file.
  FileHandlePtr file_handle;
  IREE_ORT_RETURN_IF_ERROR(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ,
      iree_make_string_view(irpa_path.data(), irpa_path.size()), allocator,
      file_handle.ForOutput()));

  // Create parameter index and parse IRPA into it.
  ParameterIndexPtr index;
  IREE_ORT_RETURN_IF_ERROR(
      iree_io_parameter_index_create(allocator, index.ForOutput()));
  IREE_ORT_RETURN_IF_ERROR(
      iree_io_parse_irpa_index(file_handle.Get(), index.Get(), allocator));

  // Create provider from index.
  ParameterProviderPtr provider;
  IREE_ORT_RETURN_IF_ERROR(iree_io_parameter_index_provider_create(
      iree_make_cstring_view("model"), index.Get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      allocator, provider.ForOutput()));

  out_index = std::move(index);
  out_provider = std::move(provider);
  return nullptr;
}

// Determines the base path for EPContext external files.
// If ep_context_file_path is set, uses its stem. Otherwise derives from the
// original model path by appending "_ctx".
static std::filesystem::path GetEpContextBasePath(
    const Ort::ConstGraph& graph, const IreeEp::Config& config) {
  if (!config.ep_context_file_path.empty()) {
    std::filesystem::path ctx_path(config.ep_context_file_path);
    return ctx_path.parent_path() / ctx_path.stem();
  }
  // Derive from original model path: model.onnx -> model_ctx
  std::filesystem::path model_path(graph.GetModelPath());
  return model_path.parent_path() / (model_path.stem().string() + "_ctx");
}

// Creates an EPContext node for the generation path.
// Uses Ort::Node C++ wrapper which handles attribute ownership transfer.
static OrtStatus* CreateEpContextNode(const Ort::ConstNode& fused_node,
                                      const std::string& vmfb_filename,
                                      const std::string& irpa_filename,
                                      const std::string& function_name,
                                      OrtNode** out_node) {
  // Get fused node input/output names.
  std::vector<Ort::ConstValueInfo> inputs = fused_node.GetInputs();
  std::vector<Ort::ConstValueInfo> outputs = fused_node.GetOutputs();

  std::vector<std::string> input_names, output_names;
  input_names.reserve(inputs.size());
  output_names.reserve(outputs.size());
  for (const auto& vi : inputs) input_names.push_back(vi.GetName());
  for (const auto& vi : outputs) output_names.push_back(vi.GetName());

  // Create attributes.
  std::string source(kEpContextSource);
  int64_t embed_mode = 0;  // external files
  int64_t main_context = 1;

  std::vector<Ort::OpAttr> attrs;
  attrs.push_back(Ort::OpAttr("source", source.data(),
                              static_cast<int>(source.size()),
                              ORT_OP_ATTR_STRING));
  attrs.push_back(Ort::OpAttr("embed_mode", &embed_mode, 1, ORT_OP_ATTR_INT));
  attrs.push_back(
      Ort::OpAttr("main_context", &main_context, 1, ORT_OP_ATTR_INT));
  attrs.push_back(Ort::OpAttr("ep_cache_context", vmfb_filename.data(),
                              static_cast<int>(vmfb_filename.size()),
                              ORT_OP_ATTR_STRING));
  // Only set iree_parameters_path if an IRPA file exists (small models may
  // inline all weights and not produce an IRPA).
  if (!irpa_filename.empty()) {
    attrs.push_back(Ort::OpAttr("iree_parameters_path", irpa_filename.data(),
                                static_cast<int>(irpa_filename.size()),
                                ORT_OP_ATTR_STRING));
  }
  // Store the IREE function name for lookup at load time.
  attrs.push_back(Ort::OpAttr("iree_function_name", function_name.data(),
                              static_cast<int>(function_name.size()),
                              ORT_OP_ATTR_STRING));

  std::string fused_name = fused_node.GetName();

  // Ort::Node constructor calls CreateNode and takes ownership of attributes.
  // Wrap in try-catch since Ort::Node constructor throws on error and we need
  // to return OrtStatus* from the noexcept CompileImpl call chain.
  try {
    Ort::Node node(kEpContextOpType, kEpContextDomain, fused_name, input_names,
                   output_names, attrs);
    // Transfer ownership to the output parameter (ORT takes ownership).
    *out_node = node.release();
  } catch (const Ort::Exception& e) {
    return Ort::Status(e.what(), ORT_FAIL).release();
  }
  return nullptr;
}

// CompileImpl: EPContext loading path.
// Loads cached VMFB and IRPA from external files referenced by the EPContext
// node, skipping MLIR generation and iree-compile entirely.
static OrtStatus* CompileEpContextPath(
    IreeEp* ep, const Ort::ConstGraph& graph,
    OrtNodeComputeInfo** node_compute_infos) {
  // Get the single EPContext node.
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.size() != 1) {
    return Ort::Status("IREE EP: Expected single EPContext node",
                       ORT_INVALID_ARGUMENT)
        .release();
  }
  const Ort::ConstNode& node = nodes[0];

  // Read VMFB path (required) and IRPA path (optional).
  std::string vmfb_filename;
  ORT_RETURN_IF_ERROR(
      ReadEpContextStringAttr(node, "ep_cache_context", vmfb_filename));

  // IRPA is optional — small models may inline all weights.
  std::string irpa_filename;
  {
    Ort::ConstOpAttr irpa_attr;
    auto status = node.GetAttributeByName("iree_parameters_path", irpa_attr);
    if (status.IsOK() && irpa_attr) {
      irpa_attr.GetValue<std::string>(irpa_filename);
    }
  }

  // Resolve paths relative to the context model directory.
  std::filesystem::path model_dir =
      std::filesystem::path(graph.GetModelPath()).parent_path();
  std::string vmfb_path = (model_dir / vmfb_filename).string();

  ORT_CXX_LOGF_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Loading cached VMFB from: %s",
                        vmfb_path.c_str());

  // Load VMFB bytes.
  std::vector<uint8_t> vmfb_data;
  ORT_RETURN_IF_ERROR(ReadFileToBytes(vmfb_path, vmfb_data));

  // Load IRPA into parameter index + provider (if present).
  ParameterIndexPtr parameter_index;
  ParameterProviderPtr parameter_provider;
  if (!irpa_filename.empty()) {
    std::string irpa_path = (model_dir / irpa_filename).string();
    ORT_CXX_LOGF_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Loading cached IRPA from: %s",
                          irpa_path.c_str());
    ORT_RETURN_IF_ERROR(
        LoadIrpaFromFile(irpa_path, parameter_index, parameter_provider));
  }

  // Read function name from EPContext attribute, or default to "module.main".
  std::string function_name = "module.main";
  {
    Ort::ConstOpAttr fn_attr;
    auto status = node.GetAttributeByName("iree_function_name", fn_attr);
    if (status.IsOK() && fn_attr) {
      fn_attr.GetValue<std::string>(function_name);
    }
  }

  // Create NodeComputeInfo with loaded artifacts.
  auto* info = new IreeNodeComputeInfo(
      *ep, std::move(vmfb_data), std::move(parameter_index),
      std::move(parameter_provider), std::move(function_name));
  node_compute_infos[0] = info;

  ORT_CXX_LOG_NOEXCEPT(
      ep->Logger(), ORT_LOGGING_LEVEL_INFO,
      "IREE EP: EPContext loading complete (skipped compilation)");
  return nullptr;
}

// CompileImpl: Normal compilation path.
// Generates MLIR, compiles to VMFB, optionally creates EPContext node.
static OrtStatus* CompileNormalPath(IreeEp* ep, const Ort::ConstGraph& graph,
                                    const OrtNode* fused_node,
                                    OrtNodeComputeInfo** node_compute_infos,
                                    OrtNode** ep_context_nodes) {
  const auto& config = ep->GetConfig();

  // Create temp files for intermediate artifacts.
  TempFile mlir_file(".mlir");
  TempFile vmfb_file(".vmfb");
  TempFile irpa_file(".irpa");

  // save_intermediates: keep all artifacts on disk for debugging/inspection.
  if (config.save_intermediates) {
    mlir_file.Keep();
    vmfb_file.Keep();
    irpa_file.Keep();
    ORT_CXX_LOGF_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Saving MLIR to: %s",
                          mlir_file.Path().c_str());
    ORT_CXX_LOGF_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Saving VMFB to: %s",
                          vmfb_file.Path().c_str());
    ORT_CXX_LOGF_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Saving IRPA to: %s",
                          irpa_file.Path().c_str());
  }

  // Phase 1: Generate MLIR from the graph.
  // Also builds an IRPA parameter archive for large initializers.
  ParameterIndexPtr parameter_index;
  ParameterProviderPtr parameter_provider;
  ORT_CXX_LOG_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                       "IREE EP: Generating MLIR");
  ORT_RETURN_IF_ERROR(GenerateMlir(
      graph, ep->ort_api, mlir_file.Path(), irpa_file.Path(), parameter_index,
      parameter_provider, config.ep_context_enable));
  ORT_CXX_LOG_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                       "IREE EP: MLIR Generated Successfully");

  // Phase 2: Compile MLIR to VMFB.
  std::vector<std::string> flags = GenerateCompileFlags(config);
  ORT_CXX_LOG_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                       "IREE EP: Generating VMFB");
  ORT_RETURN_IF_ERROR(
      CompileToVmfb(mlir_file.Path(), vmfb_file.Path(), flags, ep->ort_api));
  ORT_CXX_LOG_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                       "IREE EP: VMFB Generated Successfully");

  // Phase 3: Read VMFB into memory.
  std::vector<uint8_t> vmfb_data;
  ORT_RETURN_IF_ERROR(ReadFileToBytes(vmfb_file.Path(), vmfb_data));

  // Build function name for later lookup.
  // Format: "module.{graph_name}" (defaults to "main" if empty).
  std::string graph_name = graph.GetName();
  std::string function_name =
      "module." + (graph_name.empty() ? std::string("main") : graph_name);

  // Phase 4: If EPContext generation is enabled, save external files and create
  // the EPContext node for ORT to serialize into the context model.
  if (config.ep_context_enable && ep_context_nodes != nullptr) {
    std::filesystem::path base_path = GetEpContextBasePath(graph, config);
    std::string vmfb_dest = base_path.string() + ".vmfb";

    ORT_CXX_LOGF_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Saving EPContext VMFB to: %s",
                          vmfb_dest.c_str());

    // Copy compiled artifacts to their permanent locations.
    ORT_RETURN_IF_ERROR(CopyFile(vmfb_file.Path(), vmfb_dest));

    // Copy IRPA only if parameters were generated (large models with
    // externalized weights). Small models inline all weights in MLIR.
    std::string irpa_filename;
    if (parameter_index) {
      std::string irpa_dest = base_path.string() + ".irpa";
      ORT_CXX_LOGF_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                            "IREE EP: Saving EPContext IRPA to: %s",
                            irpa_dest.c_str());
      ORT_RETURN_IF_ERROR(CopyFile(irpa_file.Path(), irpa_dest));
      irpa_filename = std::filesystem::path(irpa_dest).filename().string();
    }

    // Filenames (not paths) stored in EPContext attributes. These are resolved
    // relative to the context model's directory at load time.
    std::string vmfb_filename =
        std::filesystem::path(vmfb_dest).filename().string();

    Ort::ConstNode fused_node_wrapper{fused_node};
    ORT_RETURN_IF_ERROR(CreateEpContextNode(fused_node_wrapper, vmfb_filename,
                                            irpa_filename, function_name,
                                            &ep_context_nodes[0]));

    ORT_CXX_LOG_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                         "IREE EP: EPContext node created");
  }

  // Create NodeComputeInfo with compiled artifacts. Session creation and VMFB
  // loading are deferred to first ComputeImpl call (lazy initialization).
  auto* info = new IreeNodeComputeInfo(
      *ep, std::move(vmfb_data), std::move(parameter_index),
      std::move(parameter_provider), std::move(function_name));
  node_compute_infos[0] = info;

  ORT_CXX_LOG_NOEXCEPT(ep->Logger(), ORT_LOGGING_LEVEL_INFO,
                       "IREE EP: Compilation complete");
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEp::CompileImpl(
    OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** fused_nodes,
    size_t count, OrtNodeComputeInfo** node_compute_infos,
    OrtNode** ep_context_nodes) noexcept {
  auto* ep = static_cast<IreeEp*>(this_ptr);

  if (count == 0 || graphs == nullptr) {
    return Ort::Status("IREE EP: No graphs provided to compile.",
                       ORT_INVALID_ARGUMENT)
        .release();
  }

  // TODO: Handle multiple graphs (count > 1).
  Ort::ConstGraph graph{graphs[0]};

  // Check if this is an EPContext loading path.
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.size() == 1 && IsIreeEpContextNode(nodes[0])) {
    return CompileEpContextPath(ep, graph, node_compute_infos);
  }

  // Normal compilation path.
  return CompileNormalPath(ep, graph, fused_nodes[0], node_compute_infos,
                           ep_context_nodes);
}

/*static*/
void ORT_API_CALL IreeEp::ReleaseNodeComputeInfosImpl(
    OrtEp* /*this_ptr*/, OrtNodeComputeInfo** node_compute_infos,
    size_t num_node_compute_infos) noexcept {
  // Delete all node compute infos we created
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    if (node_compute_infos[i] != nullptr) {
      delete static_cast<IreeNodeComputeInfo*>(node_compute_infos[i]);
    }
  }
}

// ============================================================================
// IreeNodeComputeInfo Implementation
// ============================================================================

IreeNodeComputeInfo::IreeNodeComputeInfo(
    IreeEp& ep_ref, std::vector<uint8_t> vmfb_data,
    ParameterIndexPtr parameter_index, ParameterProviderPtr parameter_provider,
    std::string function_name)
    : ep(ep_ref),
      vmfb_data_(std::move(vmfb_data)),
      parameter_index_(std::move(parameter_index)),
      parameter_provider_(std::move(parameter_provider)),
      function_name_(std::move(function_name)) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

IreeNodeComputeInfo::~IreeNodeComputeInfo() {
  // Note: Avoid using logger during cleanup - ORT logging infrastructure may
  // be torn down before our destructors run during Python interpreter shutdown.
  // Release session before vmfb_data_ is destroyed, since the session
  // references VMFB bytes directly (loaded with iree_allocator_null).
  // Redundant with current member declaration order but defensive against
  // future reordering.
  session_.Reset();
}

OrtStatus* IreeNodeComputeInfo::InitializeSession() {
  // Phase 1: Create IREE runtime session with device.
  iree_runtime_session_options_t session_opts;
  iree_runtime_session_options_initialize(&session_opts);
  IREE_ORT_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      ep.IreeInstance(), &session_opts, ep.IreeDevice(),
      iree_runtime_instance_host_allocator(ep.IreeInstance()),
      session_.ForOutput()));

  // Phase 2: Register io_parameters module if we have parameters.
  if (parameter_provider_) {
    VmModulePtr parameters_module;
    iree_io_parameter_provider_t* provider_raw = parameter_provider_.Get();
    IREE_ORT_RETURN_IF_ERROR(iree_io_parameters_module_create(
        iree_runtime_instance_vm_instance(ep.IreeInstance()), 1, &provider_raw,
        iree_runtime_instance_host_allocator(ep.IreeInstance()),
        parameters_module.ForOutput()));
    IREE_ORT_RETURN_IF_ERROR(iree_runtime_session_append_module(
        session_.Get(), parameters_module.Get()));
  }

  // Phase 3: Load VMFB from memory (avoids disk round-trip).
  iree_const_byte_span_t flatbuffer_data = {
      vmfb_data_.data(), static_cast<iree_host_size_t>(vmfb_data_.size())};
  IREE_ORT_RETURN_IF_ERROR(
      iree_runtime_session_append_bytecode_module_from_memory(
          session_.Get(), flatbuffer_data, iree_allocator_null()));

  // Phase 4: Lookup function.
  IREE_ORT_RETURN_IF_ERROR(iree_runtime_session_lookup_function(
      session_.Get(), iree_make_cstring_view(function_name_.c_str()),
      &function_));

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL IreeNodeComputeInfo::CreateStateImpl(
    OrtNodeComputeInfo* /*this_ptr*/,
    OrtNodeComputeContext* /*compute_context*/, void** compute_state) noexcept {
  // No per-invocation state needed - session/function stored in
  // NodeComputeInfo.
  *compute_state = nullptr;
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL IreeNodeComputeInfo::ComputeImpl(
    OrtNodeComputeInfo* this_ptr, void* /*compute_state*/,
    OrtKernelContext* kernel_context) noexcept {
  auto* info = static_cast<IreeNodeComputeInfo*>(this_ptr);

  // Lazy-init: create session on first invocation (thread-safe).
  std::call_once(info->init_flag_, [info]() {
    OrtStatus* status = info->InitializeSession();
    if (status) {
      info->init_error_ = Ort::Status(status).GetErrorMessage();
    }
  });
  if (!info->init_error_.empty()) {
    return Ort::Status(info->init_error_.c_str(), ORT_FAIL).release();
  }

  Ort::KernelContext ctx(kernel_context);

  iree_hal_device_t* device = info->ep.IreeDevice();
  iree_hal_allocator_t* allocator =
      iree_runtime_session_device_allocator(info->session_.Get());

  // Convert ORT inputs to IREE buffer views.
  std::vector<HalBufferViewPtr> input_views;
  size_t input_count = ctx.GetInputCount();
  input_views.reserve(input_count);

  for (size_t i = 0; i < input_count; ++i) {
    Ort::ConstValue input = ctx.GetInput(i);
    iree_hal_buffer_view_t* view = nullptr;
    ORT_RETURN_IF_ERROR(OrtTensorToIreeBufferView(
        input, device, allocator, iree_allocator_system(), &view,
        info->ep.ep_api, info->ep.Logger()));
    input_views.emplace_back(view);
  }

  // Initialize the call.
  RuntimeCall call;
  IREE_ORT_RETURN_IF_ERROR(iree_runtime_call_initialize(
      info->session_.Get(), info->function_, call.Get()));
  call.MarkInitialized();

  // Push input buffer views.
  for (auto& view : input_views) {
    IREE_ORT_RETURN_IF_ERROR(
        iree_runtime_call_inputs_push_back_buffer_view(call.Get(), view.Get()));
  }

  // Invoke the function.
  IREE_ORT_RETURN_IF_ERROR(
      iree_runtime_call_invoke(call.Get(), IREE_RUNTIME_CALL_FLAG_RESERVED));

  // Pop outputs and copy to ORT tensors.
  //
  // TODO(perf): Currently IREE allocates its own output buffers, then we copy
  // to ORT's pre-allocated device buffers (D2D copy). The way to properly
  // eliminate this is by passing mutable dps buffers as part of the iree input
  // signature and writing to them. The problem is that ORT doesn't give us a
  // good way to infer the output shape. I'm not sure what the right fix is.
  // Maybe we could have a custom iree allocator that does the job for us?
  // I'm just not sure how to do this properly.
  iree_vm_list_t* output_list = iree_runtime_call_outputs(call.Get());
  iree_host_size_t output_count = iree_vm_list_size(output_list);

  for (size_t i = 0; i < output_count; ++i) {
    // Pop output buffer view.
    iree_hal_buffer_view_t* output_view = nullptr;
    IREE_ORT_RETURN_IF_ERROR(iree_runtime_call_outputs_pop_front_buffer_view(
        call.Get(), &output_view));
    HalBufferViewPtr output_view_ptr(output_view);

    // Get shape and element type from IREE buffer view.
    std::vector<int64_t> shape = GetBufferViewShape(output_view);
    iree_hal_element_type_t iree_dtype =
        iree_hal_buffer_view_element_type(output_view);
    ONNXTensorElementDataType onnx_dtype = IreeToOnnxElementType(iree_dtype);

    if (onnx_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return Ort::Status("IREE EP: Unsupported output element type",
                         ORT_NOT_IMPLEMENTED)
          .release();
    }

    // Allocate ORT output tensor and copy data from IREE buffer.
    Ort::UnownedValue output = ctx.GetOutput(i, shape.data(), shape.size());
    ORT_RETURN_IF_ERROR(IreeBufferViewToOrtTensor(
        output_view, output, device, info->ep.ep_api, info->ep.Logger()));
  }
  return nullptr;
}

/*static*/
void ORT_API_CALL IreeNodeComputeInfo::ReleaseStateImpl(
    OrtNodeComputeInfo* /*this_ptr*/, void* /*compute_state*/) noexcept {
  // No per-invocation state to release.
}

}  // namespace onnxruntime::iree

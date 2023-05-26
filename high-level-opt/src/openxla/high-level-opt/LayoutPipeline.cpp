// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/high-level-opt/LayoutPipeline.h"

#include "mhlo/transforms/passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/hlo_constant_splitter.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/gpu/gpu_layout_assignment.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/reshape_decomposer.h"
#include "xla/service/reduce_decomposer.h"
#include "xla/service/transpose_folding.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

using namespace mlir;

namespace xla {

StatusOr<std::optional<HloInstruction*>> NormalizeLayoutForGpuCustomCalls(
    HloCustomCallInstruction* hlo) {
  return {std::nullopt};
}

Status RunLayoutPipeline(HloModule *hlo_module) {
  // Run layout assignment in a separate pipeline from
  // "post-layout-assignment" because we want everything after layout
  // assignment to have a layout-sensitive invariant-checker, but
  // HloPassPipeline also runs its invariant checker before any passes are
  // run, meaning, the pipeline that contains layout assignment cannot contain
  // a layout-sensitive verifier!
  HloPassPipeline pipeline("layout assignment");
  // Layout assignment uses alias analysis, which requires the call graph to
  // be flattened.
  pipeline.AddPass<FlattenCallGraph>();
  ChannelLayoutConstraints layout_constraints;
  pipeline.AddPass<gpu::GpuLayoutAssignment>(
      hlo_module->mutable_entry_computation_layout(), /* se= */ nullptr,
      &layout_constraints);


  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  AlgebraicSimplifierOptions options;
  options.set_enable_dot_strength_reduction(false);
  options.set_supports_non_canonical_dots(false);
  options.set_is_layout_sensitive(true);
  options.set_enable_conv_operand_swap(false);
  options.set_minmax_propagate_nan(false);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);

  // GemmRewriter assumes that all transposes are folded into gemms, but,
  // since commit 7d529df, this is not always true at this point.
  // Therefore, rerun transpose folding.
  // pipeline.AddPass<TransposeFolding>(CanFoldTransposeOperandIntoDot,
  //                                    TransposeFolding::NeverFoldTranspose);

  pipeline.AddPass<ReshapeDecomposer>();
  // pipeline.AddPass<ReduceDecomposer>([&](const HloInstruction* r) {
  //   return IsReductionFromOrToContiguousDimensions(*r);
  // });
  pipeline.AddPass<LayoutNormalization>(&NormalizeLayoutForGpuCustomCalls);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);

  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return OkStatus();
}

}  // namespace xla

namespace openxla::hlopt {

class RunLayoutPipelinePass
    : public PassWrapper<RunLayoutPipelinePass, OperationPass<ModuleOp>> {
 public:
  RunLayoutPipelinePass() {}

 private:
  StringRef getArgument() const override { return "openxla-high-level-opts-layout"; }
  void runOnOperation() override {
    // Convert to HLO.
    xla::HloProto hlo_proto;
    auto status = ConvertMlirHloToHlo(getOperation(), &hlo_proto,
                                      /*use_tuple_args=*/false,
                                      /*return_tuple=*/false);
    if (!status.ok()) {
      getOperation()->emitError()
          << "failed to convert mhlo to hlo: " << status.ToString();
      return signalPassFailure();
    }

    // Run the optimizer.
    auto moduleOr = runHLOOptimizer(hlo_proto);
    if (!moduleOr.ok()) {
      getOperation()->emitError()
          << "failed to run hlo optimizer: " << status.ToString();
      return signalPassFailure();
    }
    auto hloModule = std::move(moduleOr).value();

    // When converting back, the HLO is inlined into the MLIR module. So
    // first, we remove all children from the MLIR module first.
    while (!getOperation().getBody()->empty()) {
      getOperation().getBody()->front().erase();
    }
    status = xla::ConvertHloToMlirHlo(getOperation(), hloModule.get());
    if (!status.ok()) {
      getOperation()->emitError()
          << "failed to convert hlo to mhlo: " << status.ToString();
      return signalPassFailure();
    }
  }

  xla::StatusOr<std::unique_ptr<xla::HloModule>> runHLOOptimizer(
      xla::HloProto hlo_proto) {
    xla::DebugOptions debugOptions;
    TF_ASSIGN_OR_RETURN(xla::HloModuleConfig hlo_module_config,
                        xla::HloModule::CreateModuleConfigFromProto(
                            hlo_proto.hlo_module(), debugOptions));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloModule> hlo_module,
                        xla::HloModule::CreateFromProto(hlo_proto.hlo_module(),
                                                        hlo_module_config));
    TF_RETURN_IF_ERROR(RunLayoutPipeline(hlo_module.get()));

    return hlo_module;
  }
};

// Builds a pipeline which runs the XLA layout passes.
void buildLayoutPipeline(mlir::OpPassManager &passManager) {
  // To MHLO.
  passManager.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());

  passManager.addPass(std::make_unique<RunLayoutPipelinePass>());

  // And back to stablehlo.
  passManager.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
}

}  // namespace openxla::hlopt


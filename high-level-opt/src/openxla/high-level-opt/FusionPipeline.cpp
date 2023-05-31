// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/high-level-opt/FusionPipeline.h"

#include "mhlo/transforms/passes.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/gpu/instruction_fusion.h"
#include "xla/service/gpu/multi_output_fusion.h"
#include "xla/service/gpu/fusion_merger.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

using namespace mlir;

namespace xla {

HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() {
    return [](const Shape& shape) {
    constexpr int pointer_size = 8;
    return gpu::GetSizeOfShape(shape, pointer_size);
  };
}

Status RunFusionPipeline(HloModule *hlo_module) {
    HloPassFix<HloPassPipeline> fusion("fusion");

    gpu::GpuDeviceInfo gpu_device_info; 
    
    fusion.AddPass<gpu::GpuInstructionFusion>(/*may_duplicate=*/false,
                                         gpu_device_info);
    fusion.AddPass<gpu::GpuInstructionFusion>(/*may_duplicate=*/true,
                                         gpu_device_info);
    fusion.AddPass<gpu::FusionMerger>(gpu_device_info, ShapeSizeBytesFunction());
    fusion.AddPass<gpu::GpuMultiOutputFusion>(gpu_device_info,
                                         ShapeSizeBytesFunction());
    fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                           /*only_fusion_computations=*/true);
    fusion.AddPass<HloDCE>();
    
    return fusion.Run(hlo_module).status();
}

}  // namespace xla

namespace {

class RunFusionPipelinePass
    : public PassWrapper<RunFusionPipelinePass, OperationPass<ModuleOp>> {
 public:
  RunFusionPipelinePass() = default;

 private:
  StringRef getArgument() const override {
    return "openxla-high-level-opts-fusion";
  }
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
    TF_RETURN_IF_ERROR(RunFusionPipeline(hlo_module.get()));

    return hlo_module;
  }

};

static LogicalResult convertFusionToDispatch(mhlo::FusionOp op,
                                             PatternRewriter& rewriter) {
  
  return failure();
}

class ConvertFusionIntoDispatchPass
    : public PassWrapper<ConvertFusionIntoDispatchPass, OperationPass<ModuleOp>> {
 public:
  ConvertFusionIntoDispatchPass() = default;

 private:
  StringRef getArgument() const override {
    return "openxla-high-level-opts-fusion-to-dispatch";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(&convertFusionToDispatch);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // anonymous namespace

// Builds a pipeline which runs the XLA fusion passes.
void openxla::hlopt::buildFusionPipeline(mlir::OpPassManager & passManager) {
  // To MHLO.
  passManager.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());

  passManager.addPass(std::make_unique<RunFusionPipelinePass>());
  passManager.addPass(std::make_unique<ConvertFusionIntoDispatchPass>());

  // And back to stablehlo.
  passManager.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
}

// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "openxla/high-level-opt/FusionPipeline.h"
#include "openxla/high-level-opt/LayoutPipeline.h"
#include "stablehlo/dialect/Register.h"

using namespace mlir;

int main(int argc, char** argv) {
  // mlir::registerAllPasses();
  // mhlo::registerAllMhloPasses();

  PassPipelineRegistration<> layout_pipeline(
      "openxla-layout-pipeline",
      "Runs XLA's layout assignment on top of a stable-hlo input",
      [](OpPassManager& pm) { openxla::hlopt::buildLayoutPipeline(pm); });

  PassPipelineRegistration<> fusion_pipeline(
      "openxla-fusion-pipeline",
      "Runs XLA's fusion passes on top of a stable-hlo input",
      [](OpPassManager& pm) { openxla::hlopt::buildFusionPipeline(pm); });

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mhlo::registerAllMhloDialects(registry);
  stablehlo::registerAllDialects(registry);

  return mlir::failed(MlirOptMain(
      argc, argv, "openxla high-level-opt pass driver\n", registry));
}

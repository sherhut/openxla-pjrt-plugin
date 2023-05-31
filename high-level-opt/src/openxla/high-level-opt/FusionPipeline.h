// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_HIGH_LEVEL_OPT_FUSION_PIPELINE_H
#define OPENXLA_HIGH_LEVEL_OPT_FUSION_PIPELINE_H

#include "mlir/Pass/PassManager.h"

namespace openxla::hlopt {

// Builds a pipeline which runs the XLA fusion pipeline
void buildFusionPipeline(mlir::OpPassManager &passManager);

}  // namespace openxla::hlopt

#endif  // OPENXLA_HIGH_LEVEL_OPT_FUSION_PIPELINE_H

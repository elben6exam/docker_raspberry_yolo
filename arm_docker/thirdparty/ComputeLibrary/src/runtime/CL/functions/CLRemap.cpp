/*
 * Copyright (c) 2017-2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "arm_compute/runtime/CL/functions/CLRemap.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLRemapKernel.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
void CLRemap::configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output, InterpolationPolicy policy,
                        BorderMode border_mode, uint8_t constant_border_value)
{
    configure(compile_context, input, map_x, map_y, output, policy, border_mode, PixelValue{ constant_border_value });
}

void CLRemap::configure(ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, map_x, map_y, output, policy, border_mode, PixelValue{ constant_border_value });
}

void CLRemap::configure(ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, map_x, map_y, output, policy, border_mode, constant_border_value);
}

void CLRemap::configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *map_x, const ICLTensor *map_y, ICLTensor *output, InterpolationPolicy policy,
                        BorderMode border_mode, PixelValue constant_border_value)
{
    ARM_COMPUTE_LOG_PARAMS(input, map_x, map_y, output, policy, border_mode, constant_border_value);
    auto k = std::make_unique<CLRemapKernel>();
    k->configure(compile_context, input, map_x, map_y, output, RemapInfo{ policy, border_mode, constant_border_value });
    _kernel = std::move(k);
    _border_handler->configure(compile_context, input, _kernel->border_size(), border_mode, constant_border_value);
}

Status CLRemap::validate(const ITensorInfo *input, const ITensorInfo *map_x, const ITensorInfo *map_y, const ITensorInfo *output, const InterpolationPolicy policy, const BorderMode border_mode,
                         PixelValue constant_border_value)
{
    return CLRemapKernel::validate(input, map_x, map_y, output, RemapInfo{ policy, border_mode, constant_border_value });
}
} // namespace arm_compute

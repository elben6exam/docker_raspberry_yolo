/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEREMAPKERNEL_H
#define ARM_COMPUTE_NEREMAPKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Kernel to perform a remap on a tensor */
class NERemapKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NERemapKernel";
    }
    /** Default constructor */
    NERemapKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NERemapKernel(const NERemapKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NERemapKernel &operator=(const NERemapKernel &) = delete;
    /** Allow instances of this class to be moved */
    NERemapKernel(NERemapKernel &&) = default;
    /** Allow instances of this class to be moved */
    NERemapKernel &operator=(NERemapKernel &&) = default;
    /** Default destructor */
    ~NERemapKernel() = default;

    /** Initialize the kernel's input, output and border mode.
     *
     * @param[in]  input                 Source tensor. Data type supported: U8.
     * @param[in]  map_x                 Map for X coordinates. Data type supported: F32.
     * @param[in]  map_y                 Map for Y coordinates. Data type supported: F32.
     * @param[out] output                Destination tensor. Data types supported: U8. All but the lowest two dimensions must be the same size as in the input tensor, i.e. remapping is only performed within the XY-plane.
     * @param[in]  policy                The interpolation type.
     * @param[in]  border_mode           Border mode to use on the input tensor.
     * @param[in]  constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT. Defaults to 0.
     */
    void configure(const ITensor *input, const ITensor *map_x, const ITensor *map_y, ITensor *output, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** function to perform nearest interpolation on the given window */
    void remap_nearest(const Window &window);
    /** function to perform bilinear interpolation on the given window */
    void remap_bilinear(const Window &window);
    /** Remap function to use for the particular interpolation type passed to configure() */
    void (NERemapKernel::*_func)(const Window &window);

    const ITensor *_input;                 /**< Input image */
    ITensor       *_output;                /**< Output image */
    const ITensor *_map_x;                 /**< Input remap x coordinates */
    const ITensor *_map_y;                 /**< Input remap y coordinates */
    BorderMode     _border_mode;           /**< Border mode */
    uint8_t        _constant_border_value; /**< Border value to use */
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEREMAPKERNEL_H */
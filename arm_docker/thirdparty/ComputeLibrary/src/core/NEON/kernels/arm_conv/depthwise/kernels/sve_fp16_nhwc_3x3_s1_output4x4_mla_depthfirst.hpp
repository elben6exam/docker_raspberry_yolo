/*
 * Copyright (c) 2021 Arm Limited.
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

#include "src/core/NEON/kernels/arm_gemm/utils.hpp"

#include <cstdint>

#pragma once

#if __aarch64__ && defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)

namespace arm_conv {
namespace depthwise {

void sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst_indirect_impl(const __fp16 *const *const, __fp16 *const *const, const void *, unsigned int, const __fp16, const __fp16);
void sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst_direct_impl(const unsigned int, const unsigned int, const __fp16 *, int64_t, int64_t, __fp16 *, int64_t, int64_t, const void *, unsigned int, const __fp16, const __fp16);

class sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst : public IDepthwiseDepthfirstStrategy
{
  private:
  typedef void (*indirect_kern_type)(const __fp16 *const *const, __fp16 *const *const, const void *, unsigned int, const __fp16, const __fp16);
  indirect_kern_type m_indirect_kernel = sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst_indirect_impl;

  typedef void (*direct_kern_type)(const unsigned int, const unsigned int, const __fp16 *, int64_t, int64_t, __fp16 *, int64_t, int64_t, const void *, unsigned int, const __fp16, const __fp16);
  direct_kern_type m_direct_kernel = sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst_direct_impl;

  public:
  typedef __fp16 return_type;

  constexpr static arm_gemm::VLType vl_type = arm_gemm::VLType::SVE;

  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  constexpr static unsigned int output_rows = 4;
  constexpr static unsigned int output_cols = 4;

  constexpr static unsigned int input_rows = 6;
  constexpr static unsigned int input_cols = 6;

  sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst(const CPUInfo *) {}

  arm_gemm::VLType get_vl_type(void) const override { return vl_type; }

  unsigned int get_kernel_rows(void) const override { return kernel_rows; }
  unsigned int get_kernel_cols(void) const override { return kernel_cols; }

  unsigned int get_stride_rows(void) const override { return stride_rows; }
  unsigned int get_stride_cols(void) const override { return stride_cols; }

  unsigned int get_output_rows(void) const override { return output_rows; }
  unsigned int get_output_cols(void) const override { return output_cols; }

  unsigned int get_input_rows(void) const override { return input_rows; }
  unsigned int get_input_cols(void) const override { return input_cols; }

  void indirect_kernel(
    const void *const *const input_ptrs,
    void *const *const outptrs,
    const void *params,
    unsigned int n_channels,
    const void *activation_min,
    const void *activation_max
  ) const override
  {
    m_indirect_kernel(
      reinterpret_cast<const __fp16 *const *>(input_ptrs),
      reinterpret_cast<__fp16 *const *>(outptrs),
      params, n_channels,
      *static_cast<const __fp16 *>(activation_min),
      *static_cast<const __fp16 *>(activation_max)
    );
  }

  void direct_kernel(
    const unsigned int n_tile_rows, const unsigned int n_tile_cols,
    const void *inptr, int64_t ld_input_row, int64_t ld_input_col,
    void *outptr, int64_t ld_output_row, int64_t ld_output_col,
    const void *params, unsigned int n_channels,
    const void *activation_min, const void *activation_max
  ) const override
  {
    m_direct_kernel(
      n_tile_rows, n_tile_cols,
      static_cast<const __fp16 *>(inptr), ld_input_row, ld_input_col,
      static_cast<__fp16 *>(outptr), ld_output_row, ld_output_col,
      params, n_channels,
      *static_cast<const __fp16 *>(activation_min),
      *static_cast<const __fp16 *>(activation_max)
    );
  }
};

}  // namespace depthwise
}  // namespace arm_conv

#endif  // __aarch64__ && defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)

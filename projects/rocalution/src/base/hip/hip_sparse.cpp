/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "hip_sparse.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "hip_allocate_free.hpp"
#include "hip_utils.hpp"

#include <cassert>
#include <complex>
#include <rocsparse/rocsparse.h>

namespace rocalution
{
    template <typename ValueType>
    HIPSpMV<ValueType>::HIPSpMV(void)
        : descr_(NULL)
        , buffer_size_(0)
        , buffer_(NULL)
    {
    }

    template <typename ValueType>
    HIPSpMV<ValueType>::~HIPSpMV(void)
    {
        this->Clear();
    }

    template <typename ValueType>
    bool HIPSpMV<ValueType>::IsAnalysed(void) const
    {
        return this->descr_ != NULL;
    }

    template <typename ValueType>
    void HIPSpMV<ValueType>::Analyse(rocsparse_handle            handle,
                                     rocsparse_spmv_alg          alg,
                                     rocsparse_const_spmat_descr mat,
                                     ValueType                   alpha,
                                     rocsparse_const_dnvec_descr x,
                                     ValueType                   beta,
                                     rocsparse_dnvec_descr       y)
    {
        assert(this->descr_ == NULL);
        assert(mat != NULL);

        const rocsparse_operation operation = rocsparse_operation_none;
        const rocsparse_datatype  datatype  = rocalution_datatype_traits<ValueType>::value;

        rocsparse_status status = rocsparse_create_spmv_descr(&this->descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmv_set_input(
            handle, this->descr_, rocsparse_spmv_input_alg, &alg, sizeof(alg), NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmv_set_input(handle,
                                          this->descr_,
                                          rocsparse_spmv_input_operation,
                                          &operation,
                                          sizeof(operation),
                                          NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmv_set_input(handle,
                                          this->descr_,
                                          rocsparse_spmv_input_scalar_datatype,
                                          &datatype,
                                          sizeof(datatype),
                                          NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmv_set_input(handle,
                                          this->descr_,
                                          rocsparse_spmv_input_compute_datatype,
                                          &datatype,
                                          sizeof(datatype),
                                          NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t analysis_buffer_size = 0;
        status                      = rocsparse_v2_spmv_buffer_size(handle,
                                               this->descr_,
                                               mat,
                                               x,
                                               y,
                                               rocsparse_v2_spmv_stage_analysis,
                                               &analysis_buffer_size,
                                               NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        char* analysis_buffer = NULL;
        allocate_hip(analysis_buffer_size, &analysis_buffer);

        status = rocsparse_v2_spmv(handle,
                                   this->descr_,
                                   &alpha,
                                   mat,
                                   x,
                                   &beta,
                                   y,
                                   rocsparse_v2_spmv_stage_analysis,
                                   analysis_buffer_size,
                                   analysis_buffer,
                                   NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&analysis_buffer);

        status = rocsparse_v2_spmv_buffer_size(handle,
                                               this->descr_,
                                               mat,
                                               x,
                                               y,
                                               rocsparse_v2_spmv_stage_compute,
                                               &this->buffer_size_,
                                               NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(this->buffer_size_, &this->buffer_);
    }

    template <typename ValueType>
    void HIPSpMV<ValueType>::Analyse(rocsparse_handle            handle,
                                     rocsparse_spmv_alg          alg,
                                     rocsparse_const_spmat_descr mat)
    {
        // The analysis only depends on the matrix, empty vectors are sufficient
        rocsparse_dnvec_descr x;
        rocsparse_dnvec_descr y;

        rocsparse_status status = rocsparse_create_dnvec_descr(
            &x, 0, NULL, rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_create_dnvec_descr(
            &y, 0, NULL, rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        this->Analyse(handle, alg, mat, static_cast<ValueType>(1), x, static_cast<ValueType>(0), y);

        status = rocsparse_destroy_dnvec_descr(x);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_dnvec_descr(y);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPSpMV<ValueType>::Compute(rocsparse_handle            handle,
                                     ValueType                   alpha,
                                     rocsparse_const_spmat_descr mat,
                                     rocsparse_const_dnvec_descr x,
                                     ValueType                   beta,
                                     rocsparse_dnvec_descr       y) const
    {
        assert(this->descr_ != NULL);

        rocsparse_status status = rocsparse_v2_spmv(handle,
                                                    this->descr_,
                                                    &alpha,
                                                    mat,
                                                    x,
                                                    &beta,
                                                    y,
                                                    rocsparse_v2_spmv_stage_compute,
                                                    this->buffer_size_,
                                                    this->buffer_,
                                                    NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPSpMV<ValueType>::Clear(void)
    {
        if(this->descr_ != NULL)
        {
            rocsparse_status status = rocsparse_destroy_spmv_descr(this->descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            this->descr_ = NULL;
        }

        free_hip(&this->buffer_);
        this->buffer_size_ = 0;
    }

    template class HIPSpMV<float>;
    template class HIPSpMV<double>;
#ifdef SUPPORT_COMPLEX
    template class HIPSpMV<std::complex<float>>;
    template class HIPSpMV<std::complex<double>>;
#endif
    // rocsparse csrsv buffer size
    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 int                       m,
                                                 int                       nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_scsrsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 int                       m,
                                                 int                       nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_dcsrsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle           handle,
                                                 rocsparse_operation        trans,
                                                 int                        m,
                                                 int                        nnz,
                                                 const rocsparse_mat_descr  descr,
                                                 const std::complex<float>* csr_val,
                                                 const int*                 csr_row_ptr,
                                                 const int*                 csr_col_ind,
                                                 rocsparse_mat_info         info,
                                                 size_t*                    buffer_size)
    {
        return rocsparse_ccsrsv_buffer_size(handle,
                                            trans,
                                            m,
                                            nnz,
                                            descr,
                                            (const rocsparse_float_complex*)csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            info,
                                            buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle            handle,
                                                 rocsparse_operation         trans,
                                                 int                         m,
                                                 int                         nnz,
                                                 const rocsparse_mat_descr   descr,
                                                 const std::complex<double>* csr_val,
                                                 const int*                  csr_row_ptr,
                                                 const int*                  csr_col_ind,
                                                 rocsparse_mat_info          info,
                                                 size_t*                     buffer_size)
    {
        return rocsparse_zcsrsv_buffer_size(handle,
                                            trans,
                                            m,
                                            nnz,
                                            descr,
                                            (const rocsparse_double_complex*)csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            info,
                                            buffer_size);
    }

    // rocsparse csrsv analysis
    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const float*              csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_scsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const double*             csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_dcsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle           handle,
                                              rocsparse_operation        trans,
                                              int                        m,
                                              int                        nnz,
                                              const rocsparse_mat_descr  descr,
                                              const std::complex<float>* csr_val,
                                              const int*                 csr_row_ptr,
                                              const int*                 csr_col_ind,
                                              rocsparse_mat_info         info,
                                              rocsparse_analysis_policy  analysis,
                                              rocsparse_solve_policy     solve,
                                              void*                      temp_buffer)
    {
        return rocsparse_ccsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         (const rocsparse_float_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle            handle,
                                              rocsparse_operation         trans,
                                              int                         m,
                                              int                         nnz,
                                              const rocsparse_mat_descr   descr,
                                              const std::complex<double>* csr_val,
                                              const int*                  csr_row_ptr,
                                              const int*                  csr_col_ind,
                                              rocsparse_mat_info          info,
                                              rocsparse_analysis_policy   analysis,
                                              rocsparse_solve_policy      solve,
                                              void*                       temp_buffer)
    {
        return rocsparse_zcsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         (const rocsparse_double_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    // rocsparse csrsv
    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       nnz,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const float*              x,
                                     float*                    y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        return rocsparse_scsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      alpha,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       nnz,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const double*             x,
                                     double*                   y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        return rocsparse_dcsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      alpha,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle           handle,
                                     rocsparse_operation        trans,
                                     int                        m,
                                     int                        nnz,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* csr_val,
                                     const int*                 csr_row_ptr,
                                     const int*                 csr_col_ind,
                                     rocsparse_mat_info         info,
                                     const std::complex<float>* x,
                                     std::complex<float>*       y,
                                     rocsparse_solve_policy     policy,
                                     void*                      temp_buffer)
    {
        return rocsparse_ccsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      (const rocsparse_float_complex*)alpha,
                                      descr,
                                      (const rocsparse_float_complex*)csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      (const rocsparse_float_complex*)x,
                                      (rocsparse_float_complex*)y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle            handle,
                                     rocsparse_operation         trans,
                                     int                         m,
                                     int                         nnz,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* csr_val,
                                     const int*                  csr_row_ptr,
                                     const int*                  csr_col_ind,
                                     rocsparse_mat_info          info,
                                     const std::complex<double>* x,
                                     std::complex<double>*       y,
                                     rocsparse_solve_policy      policy,
                                     void*                       temp_buffer)
    {
        return rocsparse_zcsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      (const rocsparse_double_complex*)alpha,
                                      descr,
                                      (const rocsparse_double_complex*)csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      (const rocsparse_double_complex*)x,
                                      (rocsparse_double_complex*)y,
                                      policy,
                                      temp_buffer);
    }

    // rocsprarse csritsv buffer size
    template <>
    rocsparse_status rocsparseTcsritsv_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   rocsparse_int             m,
                                                   rocsparse_int             nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const float*              csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_scsritsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   rocsparse_int             m,
                                                   rocsparse_int             nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const double*             csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dcsritsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_buffer_size(rocsparse_handle           handle,
                                                   rocsparse_operation        trans,
                                                   rocsparse_int              m,
                                                   rocsparse_int              nnz,
                                                   const rocsparse_mat_descr  descr,
                                                   const std::complex<float>* csr_val,
                                                   const rocsparse_int*       csr_row_ptr,
                                                   const rocsparse_int*       csr_col_ind,
                                                   rocsparse_mat_info         info,
                                                   size_t*                    buffer_size)
    {
        return rocsparse_ccsritsv_buffer_size(handle,
                                              trans,
                                              m,
                                              nnz,
                                              descr,
                                              (const rocsparse_float_complex*)csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_buffer_size(rocsparse_handle            handle,
                                                   rocsparse_operation         trans,
                                                   rocsparse_int               m,
                                                   rocsparse_int               nnz,
                                                   const rocsparse_mat_descr   descr,
                                                   const std::complex<double>* csr_val,
                                                   const rocsparse_int*        csr_row_ptr,
                                                   const rocsparse_int*        csr_col_ind,
                                                   rocsparse_mat_info          info,
                                                   size_t*                     buffer_size)
    {
        return rocsparse_zcsritsv_buffer_size(handle,
                                              trans,
                                              m,
                                              nnz,
                                              descr,
                                              (const rocsparse_double_complex*)csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              buffer_size);
    }

    // rocsprarse csritsv analysis
    template <>
    rocsparse_status rocsparseTcsritsv_analysis(rocsparse_handle          handle,
                                                rocsparse_operation       trans,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const float*              csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_scsritsv_analysis(handle,
                                           trans,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_analysis(rocsparse_handle          handle,
                                                rocsparse_operation       trans,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const double*             csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_dcsritsv_analysis(handle,
                                           trans,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_analysis(rocsparse_handle           handle,
                                                rocsparse_operation        trans,
                                                rocsparse_int              m,
                                                rocsparse_int              nnz,
                                                const rocsparse_mat_descr  descr,
                                                const std::complex<float>* csr_val,
                                                const rocsparse_int*       csr_row_ptr,
                                                const rocsparse_int*       csr_col_ind,
                                                rocsparse_mat_info         info,
                                                rocsparse_analysis_policy  analysis,
                                                rocsparse_solve_policy     solve,
                                                void*                      temp_buffer)
    {
        return rocsparse_ccsritsv_analysis(handle,
                                           trans,
                                           m,
                                           nnz,
                                           descr,
                                           (const rocsparse_float_complex*)csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_analysis(rocsparse_handle            handle,
                                                rocsparse_operation         trans,
                                                rocsparse_int               m,
                                                rocsparse_int               nnz,
                                                const rocsparse_mat_descr   descr,
                                                const std::complex<double>* csr_val,
                                                const rocsparse_int*        csr_row_ptr,
                                                const rocsparse_int*        csr_col_ind,
                                                rocsparse_mat_info          info,
                                                rocsparse_analysis_policy   analysis,
                                                rocsparse_solve_policy      solve,
                                                void*                       temp_buffer)
    {
        return rocsparse_zcsritsv_analysis(handle,
                                           trans,
                                           m,
                                           nnz,
                                           descr,
                                           (const rocsparse_double_complex*)csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    // rocsprarse csritsv analysis
    template <>
    rocsparse_status rocsparseTcsritsv_solve(rocsparse_handle          handle,
                                             rocsparse_int*            host_nmaxiter,
                                             const float*              host_tol,
                                             float*                    host_history,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const float*              alpha,
                                             const rocsparse_mat_descr descr,
                                             const float*              csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             const float*              x,
                                             float*                    y,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer)
    {
        return rocsparse_scsritsv_solve(handle,
                                        host_nmaxiter,
                                        host_tol,
                                        host_history,
                                        trans,
                                        m,
                                        nnz,
                                        alpha,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        x,
                                        y,
                                        policy,
                                        temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_solve(rocsparse_handle          handle,
                                             rocsparse_int*            host_nmaxiter,
                                             const double*             host_tol,
                                             double*                   host_history,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const double*             alpha,
                                             const rocsparse_mat_descr descr,
                                             const double*             csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             const double*             x,
                                             double*                   y,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer)
    {
        return rocsparse_dcsritsv_solve(handle,
                                        host_nmaxiter,
                                        host_tol,
                                        host_history,
                                        trans,
                                        m,
                                        nnz,
                                        alpha,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        x,
                                        y,
                                        policy,
                                        temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_solve(rocsparse_handle           handle,
                                             rocsparse_int*             host_nmaxiter,
                                             const float*               host_tol,
                                             float*                     host_history,
                                             rocsparse_operation        trans,
                                             rocsparse_int              m,
                                             rocsparse_int              nnz,
                                             const std::complex<float>* alpha,
                                             const rocsparse_mat_descr  descr,
                                             const std::complex<float>* csr_val,
                                             const rocsparse_int*       csr_row_ptr,
                                             const rocsparse_int*       csr_col_ind,
                                             rocsparse_mat_info         info,
                                             const std::complex<float>* x,
                                             std::complex<float>*       y,
                                             rocsparse_solve_policy     policy,
                                             void*                      temp_buffer)
    {
        return rocsparse_ccsritsv_solve(handle,
                                        host_nmaxiter,
                                        host_tol,
                                        host_history,
                                        trans,
                                        m,
                                        nnz,
                                        (const rocsparse_float_complex*)alpha,
                                        descr,
                                        (const rocsparse_float_complex*)csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        (const rocsparse_float_complex*)x,
                                        (rocsparse_float_complex*)y,
                                        policy,
                                        temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_solve(rocsparse_handle            handle,
                                             rocsparse_int*              host_nmaxiter,
                                             const double*               host_tol,
                                             double*                     host_history,
                                             rocsparse_operation         trans,
                                             rocsparse_int               m,
                                             rocsparse_int               nnz,
                                             const std::complex<double>* alpha,
                                             const rocsparse_mat_descr   descr,
                                             const std::complex<double>* csr_val,
                                             const rocsparse_int*        csr_row_ptr,
                                             const rocsparse_int*        csr_col_ind,
                                             rocsparse_mat_info          info,
                                             const std::complex<double>* x,
                                             std::complex<double>*       y,
                                             rocsparse_solve_policy      policy,
                                             void*                       temp_buffer)
    {
        return rocsparse_zcsritsv_solve(handle,
                                        host_nmaxiter,
                                        host_tol,
                                        host_history,
                                        trans,
                                        m,
                                        nnz,
                                        (const rocsparse_double_complex*)alpha,
                                        descr,
                                        (const rocsparse_double_complex*)csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        (const rocsparse_double_complex*)x,
                                        (rocsparse_double_complex*)y,
                                        policy,
                                        temp_buffer);
    }

    // rocsparse bsrsv buffer size
    template <>
    rocsparse_status rocsparseTbsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_direction       dir,
                                                 rocsparse_operation       trans,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              bsr_val,
                                                 const int*                bsr_row_ptr,
                                                 const int*                bsr_col_ind,
                                                 int                       bsr_dim,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_sbsrsv_buffer_size(handle,
                                            dir,
                                            trans,
                                            mb,
                                            nnzb,
                                            descr,
                                            bsr_val,
                                            bsr_row_ptr,
                                            bsr_col_ind,
                                            bsr_dim,
                                            info,
                                            buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_direction       dir,
                                                 rocsparse_operation       trans,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             bsr_val,
                                                 const int*                bsr_row_ptr,
                                                 const int*                bsr_col_ind,
                                                 int                       bsr_dim,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_dbsrsv_buffer_size(handle,
                                            dir,
                                            trans,
                                            mb,
                                            nnzb,
                                            descr,
                                            bsr_val,
                                            bsr_row_ptr,
                                            bsr_col_ind,
                                            bsr_dim,
                                            info,
                                            buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_buffer_size(rocsparse_handle           handle,
                                                 rocsparse_direction        dir,
                                                 rocsparse_operation        trans,
                                                 int                        mb,
                                                 int                        nnzb,
                                                 const rocsparse_mat_descr  descr,
                                                 const std::complex<float>* bsr_val,
                                                 const int*                 bsr_row_ptr,
                                                 const int*                 bsr_col_ind,
                                                 int                        bsr_dim,
                                                 rocsparse_mat_info         info,
                                                 size_t*                    buffer_size)
    {
        return rocsparse_cbsrsv_buffer_size(handle,
                                            dir,
                                            trans,
                                            mb,
                                            nnzb,
                                            descr,
                                            (const rocsparse_float_complex*)bsr_val,
                                            bsr_row_ptr,
                                            bsr_col_ind,
                                            bsr_dim,
                                            info,
                                            buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_buffer_size(rocsparse_handle            handle,
                                                 rocsparse_direction         dir,
                                                 rocsparse_operation         trans,
                                                 int                         mb,
                                                 int                         nnzb,
                                                 const rocsparse_mat_descr   descr,
                                                 const std::complex<double>* bsr_val,
                                                 const int*                  bsr_row_ptr,
                                                 const int*                  bsr_col_ind,
                                                 int                         bsr_dim,
                                                 rocsparse_mat_info          info,
                                                 size_t*                     buffer_size)
    {
        return rocsparse_zbsrsv_buffer_size(handle,
                                            dir,
                                            trans,
                                            mb,
                                            nnzb,
                                            descr,
                                            (const rocsparse_double_complex*)bsr_val,
                                            bsr_row_ptr,
                                            bsr_col_ind,
                                            bsr_dim,
                                            info,
                                            buffer_size);
    }

    // rocsparse bsrsv analysis
    template <>
    rocsparse_status rocsparseTbsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans,
                                              int                       mb,
                                              int                       nnzb,
                                              const rocsparse_mat_descr descr,
                                              const float*              bsr_val,
                                              const int*                bsr_row_ptr,
                                              const int*                bsr_col_ind,
                                              int                       bsr_dim,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_sbsrsv_analysis(handle,
                                         dir,
                                         trans,
                                         mb,
                                         nnzb,
                                         descr,
                                         bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         bsr_dim,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans,
                                              int                       mb,
                                              int                       nnzb,
                                              const rocsparse_mat_descr descr,
                                              const double*             bsr_val,
                                              const int*                bsr_row_ptr,
                                              const int*                bsr_col_ind,
                                              int                       bsr_dim,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_dbsrsv_analysis(handle,
                                         dir,
                                         trans,
                                         mb,
                                         nnzb,
                                         descr,
                                         bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         bsr_dim,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_analysis(rocsparse_handle           handle,
                                              rocsparse_direction        dir,
                                              rocsparse_operation        trans,
                                              int                        mb,
                                              int                        nnzb,
                                              const rocsparse_mat_descr  descr,
                                              const std::complex<float>* bsr_val,
                                              const int*                 bsr_row_ptr,
                                              const int*                 bsr_col_ind,
                                              int                        bsr_dim,
                                              rocsparse_mat_info         info,
                                              rocsparse_analysis_policy  analysis,
                                              rocsparse_solve_policy     solve,
                                              void*                      temp_buffer)
    {
        return rocsparse_cbsrsv_analysis(handle,
                                         dir,
                                         trans,
                                         mb,
                                         nnzb,
                                         descr,
                                         (const rocsparse_float_complex*)bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         bsr_dim,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_analysis(rocsparse_handle            handle,
                                              rocsparse_direction         dir,
                                              rocsparse_operation         trans,
                                              int                         mb,
                                              int                         nnzb,
                                              const rocsparse_mat_descr   descr,
                                              const std::complex<double>* bsr_val,
                                              const int*                  bsr_row_ptr,
                                              const int*                  bsr_col_ind,
                                              int                         bsr_dim,
                                              rocsparse_mat_info          info,
                                              rocsparse_analysis_policy   analysis,
                                              rocsparse_solve_policy      solve,
                                              void*                       temp_buffer)
    {
        return rocsparse_zbsrsv_analysis(handle,
                                         dir,
                                         trans,
                                         mb,
                                         nnzb,
                                         descr,
                                         (const rocsparse_double_complex*)bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         bsr_dim,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    // rocsparse bsrsv
    template <>
    rocsparse_status rocsparseTbsrsv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nnzb,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     rocsparse_mat_info        info,
                                     const float*              x,
                                     float*                    y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        return rocsparse_sbsrsv_solve(handle,
                                      dir,
                                      trans,
                                      mb,
                                      nnzb,
                                      alpha,
                                      descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      bsr_dim,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nnzb,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     rocsparse_mat_info        info,
                                     const double*             x,
                                     double*                   y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        return rocsparse_dbsrsv_solve(handle,
                                      dir,
                                      trans,
                                      mb,
                                      nnzb,
                                      alpha,
                                      descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      bsr_dim,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv(rocsparse_handle           handle,
                                     rocsparse_direction        dir,
                                     rocsparse_operation        trans,
                                     int                        mb,
                                     int                        nnzb,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* bsr_val,
                                     const int*                 bsr_row_ptr,
                                     const int*                 bsr_col_ind,
                                     int                        bsr_dim,
                                     rocsparse_mat_info         info,
                                     const std::complex<float>* x,
                                     std::complex<float>*       y,
                                     rocsparse_solve_policy     policy,
                                     void*                      temp_buffer)
    {
        return rocsparse_cbsrsv_solve(handle,
                                      dir,
                                      trans,
                                      mb,
                                      nnzb,
                                      (const rocsparse_float_complex*)alpha,
                                      descr,
                                      (const rocsparse_float_complex*)bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      bsr_dim,
                                      info,
                                      (const rocsparse_float_complex*)x,
                                      (rocsparse_float_complex*)y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv(rocsparse_handle            handle,
                                     rocsparse_direction         dir,
                                     rocsparse_operation         trans,
                                     int                         mb,
                                     int                         nnzb,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* bsr_val,
                                     const int*                  bsr_row_ptr,
                                     const int*                  bsr_col_ind,
                                     int                         bsr_dim,
                                     rocsparse_mat_info          info,
                                     const std::complex<double>* x,
                                     std::complex<double>*       y,
                                     rocsparse_solve_policy      policy,
                                     void*                       temp_buffer)
    {
        return rocsparse_zbsrsv_solve(handle,
                                      dir,
                                      trans,
                                      mb,
                                      nnzb,
                                      (const rocsparse_double_complex*)alpha,
                                      descr,
                                      (const rocsparse_double_complex*)bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      bsr_dim,
                                      info,
                                      (const rocsparse_double_complex*)x,
                                      (rocsparse_double_complex*)y,
                                      policy,
                                      temp_buffer);
    }

    // rocsparse csric0 buffer size
    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  float*                    csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_scsric0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  double*                   csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_dcsric0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  std::complex<float>*      csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_ccsric0_buffer_size(handle,
                                             m,
                                             nnz,
                                             descr,
                                             (rocsparse_float_complex*)csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             info,
                                             buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  std::complex<double>*     csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_zcsric0_buffer_size(handle,
                                             m,
                                             nnz,
                                             descr,
                                             (rocsparse_double_complex*)csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             info,
                                             buffer_size);
    }

    // rocsparse csric0 analysis
    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               float*                    csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_scsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               double*                   csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_dcsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               std::complex<float>*      csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_ccsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          (rocsparse_float_complex*)csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               std::complex<double>*     csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_zcsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          (rocsparse_double_complex*)csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    // rocsparse csric0
    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      float*                    csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_scsric0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      double*                   csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_dcsric0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      std::complex<float>*      csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_ccsric0(handle,
                                 m,
                                 nnz,
                                 descr,
                                 (rocsparse_float_complex*)csr_val,
                                 csr_row_ptr,
                                 csr_col_ind,
                                 info,
                                 policy,
                                 temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      std::complex<double>*     csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_zcsric0(handle,
                                 m,
                                 nnz,
                                 descr,
                                 (rocsparse_double_complex*)csr_val,
                                 csr_row_ptr,
                                 csr_col_ind,
                                 info,
                                 policy,
                                 temp_buffer);
    }

    // rocsparse csrilu0 buffer size
    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   float*                    csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_scsrilu0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   double*                   csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dcsrilu0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   std::complex<float>*      csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_ccsrilu0_buffer_size(handle,
                                              m,
                                              nnz,
                                              descr,
                                              (rocsparse_float_complex*)csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   std::complex<double>*     csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_zcsrilu0_buffer_size(handle,
                                              m,
                                              nnz,
                                              descr,
                                              (rocsparse_double_complex*)csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              buffer_size);
    }

    // rocsparse csrilu0 analysis
    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                float*                    csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_scsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                double*                   csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_dcsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                std::complex<float>*      csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_ccsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           (rocsparse_float_complex*)csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                std::complex<double>*     csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_zcsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           (rocsparse_double_complex*)csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    // rocsparse csrilu0
    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       float*                    csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_scsrilu0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       double*                   csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_dcsrilu0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       std::complex<float>*      csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_ccsrilu0(handle,
                                  m,
                                  nnz,
                                  descr,
                                  (rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       std::complex<double>*     csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_zcsrilu0(handle,
                                  m,
                                  nnz,
                                  descr,
                                  (rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    // rocsparse bsrilu0 buffer size
    template <>
    rocsparse_status rocsparseTbsrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   float*                    bsr_val,
                                                   const int*                bsr_row_ptr,
                                                   const int*                bsr_col_ind,
                                                   int                       bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_sbsrilu0_buffer_size(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              bsr_dim,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   double*                   bsr_val,
                                                   const int*                bsr_row_ptr,
                                                   const int*                bsr_col_ind,
                                                   int                       bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dbsrilu0_buffer_size(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              bsr_dim,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   std::complex<float>*      bsr_val,
                                                   const int*                bsr_row_ptr,
                                                   const int*                bsr_col_ind,
                                                   int                       bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_cbsrilu0_buffer_size(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              (rocsparse_float_complex*)bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              bsr_dim,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   std::complex<double>*     bsr_val,
                                                   const int*                bsr_row_ptr,
                                                   const int*                bsr_col_ind,
                                                   int                       bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_zbsrilu0_buffer_size(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              (rocsparse_double_complex*)bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              bsr_dim,
                                              info,
                                              buffer_size);
    }

    // rocsparse bsrilu0 analysis
    template <>
    rocsparse_status rocsparseTbsrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                int                       mb,
                                                int                       nnzb,
                                                const rocsparse_mat_descr descr,
                                                float*                    bsr_val,
                                                const int*                bsr_row_ptr,
                                                const int*                bsr_col_ind,
                                                int                       bsr_dim,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_sbsrilu0_analysis(handle,
                                           dir,
                                           mb,
                                           nnzb,
                                           descr,
                                           bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_dim,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                int                       mb,
                                                int                       nnzb,
                                                const rocsparse_mat_descr descr,
                                                double*                   bsr_val,
                                                const int*                bsr_row_ptr,
                                                const int*                bsr_col_ind,
                                                int                       bsr_dim,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_dbsrilu0_analysis(handle,
                                           dir,
                                           mb,
                                           nnzb,
                                           descr,
                                           bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_dim,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                int                       mb,
                                                int                       nnzb,
                                                const rocsparse_mat_descr descr,
                                                std::complex<float>*      bsr_val,
                                                const int*                bsr_row_ptr,
                                                const int*                bsr_col_ind,
                                                int                       bsr_dim,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_cbsrilu0_analysis(handle,
                                           dir,
                                           mb,
                                           nnzb,
                                           descr,
                                           (rocsparse_float_complex*)bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_dim,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                int                       mb,
                                                int                       nnzb,
                                                const rocsparse_mat_descr descr,
                                                std::complex<double>*     bsr_val,
                                                const int*                bsr_row_ptr,
                                                const int*                bsr_col_ind,
                                                int                       bsr_dim,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_zbsrilu0_analysis(handle,
                                           dir,
                                           mb,
                                           nnzb,
                                           descr,
                                           (rocsparse_double_complex*)bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_dim,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    // rocsparse bsrilu0
    template <>
    rocsparse_status rocsparseTbsrilu0(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nnzb,
                                       const rocsparse_mat_descr descr,
                                       float*                    bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       bsr_dim,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_sbsrilu0(handle,
                                  dir,
                                  mb,
                                  nnzb,
                                  descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nnzb,
                                       const rocsparse_mat_descr descr,
                                       double*                   bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       bsr_dim,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_dbsrilu0(handle,
                                  dir,
                                  mb,
                                  nnzb,
                                  descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nnzb,
                                       const rocsparse_mat_descr descr,
                                       std::complex<float>*      bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       bsr_dim,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_cbsrilu0(handle,
                                  dir,
                                  mb,
                                  nnzb,
                                  descr,
                                  (rocsparse_float_complex*)bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nnzb,
                                       const rocsparse_mat_descr descr,
                                       std::complex<double>*     bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       bsr_dim,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_zbsrilu0(handle,
                                  dir,
                                  mb,
                                  nnzb,
                                  descr,
                                  (rocsparse_double_complex*)bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTgthr(rocsparse_handle     handle,
                                    int                  nnz,
                                    float*               y,
                                    float*               x_val,
                                    int*                 x_ind,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_sgthr(handle, nnz, y, x_val, x_ind, idx_base);
    }

    template <>
    rocsparse_status rocsparseTgthr(rocsparse_handle     handle,
                                    int                  nnz,
                                    double*              y,
                                    double*              x_val,
                                    int*                 x_ind,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_dgthr(handle, nnz, y, x_val, x_ind, idx_base);
    }

    template <>
    rocsparse_status rocsparseTgthr(rocsparse_handle     handle,
                                    int                  nnz,
                                    std::complex<float>* y,
                                    std::complex<float>* x_val,
                                    int*                 x_ind,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_cgthr(handle,
                               nnz,
                               (rocsparse_float_complex*)y,
                               (rocsparse_float_complex*)x_val,
                               x_ind,
                               idx_base);
    }

    template <>
    rocsparse_status rocsparseTgthr(rocsparse_handle      handle,
                                    int                   nnz,
                                    std::complex<double>* y,
                                    std::complex<double>* x_val,
                                    int*                  x_ind,
                                    rocsparse_index_base  idx_base)
    {
        return rocsparse_zgthr(handle,
                               nnz,
                               (rocsparse_double_complex*)y,
                               (rocsparse_double_complex*)x_val,
                               x_ind,
                               idx_base);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_compute(rocsparse_handle     handle,
                                                 rocsparse_itilu0_alg alg,
                                                 rocsparse_int        option,
                                                 rocsparse_int*       nmaxiter,
                                                 float                tol,
                                                 rocsparse_int        m,
                                                 rocsparse_int        nnz,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 const float*         csr_val,
                                                 float*               ilu0,
                                                 rocsparse_index_base idx_base,
                                                 size_t               buffer_size,
                                                 void*                buffer)
    {
        return rocsparse_scsritilu0_compute(handle,
                                            alg,
                                            option,
                                            nmaxiter,
                                            tol,
                                            m,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            ilu0,
                                            idx_base,
                                            buffer_size,
                                            buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_compute(rocsparse_handle     handle,
                                                 rocsparse_itilu0_alg alg,
                                                 rocsparse_int        option,
                                                 rocsparse_int*       nmaxiter,
                                                 double               tol,
                                                 rocsparse_int        m,
                                                 rocsparse_int        nnz,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 const double*        csr_val,
                                                 double*              ilu0,
                                                 rocsparse_index_base idx_base,
                                                 size_t               buffer_size,
                                                 void*                buffer)
    {
        return rocsparse_dcsritilu0_compute(handle,
                                            alg,
                                            option,
                                            nmaxiter,
                                            tol,
                                            m,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            ilu0,
                                            idx_base,
                                            buffer_size,
                                            buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_compute(rocsparse_handle           handle,
                                                 rocsparse_itilu0_alg       alg,
                                                 rocsparse_int              option,
                                                 rocsparse_int*             nmaxiter,
                                                 float                      tol,
                                                 rocsparse_int              m,
                                                 rocsparse_int              nnz,
                                                 const rocsparse_int*       csr_row_ptr,
                                                 const rocsparse_int*       csr_col_ind,
                                                 const std::complex<float>* csr_val,
                                                 std::complex<float>*       ilu0,
                                                 rocsparse_index_base       idx_base,
                                                 size_t                     buffer_size,
                                                 void*                      buffer)
    {
        return rocsparse_ccsritilu0_compute(handle,
                                            alg,
                                            option,
                                            nmaxiter,
                                            tol,
                                            m,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            (const rocsparse_float_complex*)csr_val,
                                            (rocsparse_float_complex*)ilu0,
                                            idx_base,
                                            buffer_size,
                                            buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_compute(rocsparse_handle            handle,
                                                 rocsparse_itilu0_alg        alg,
                                                 rocsparse_int               option,
                                                 rocsparse_int*              nmaxiter,
                                                 double                      tol,
                                                 rocsparse_int               m,
                                                 rocsparse_int               nnz,
                                                 const rocsparse_int*        csr_row_ptr,
                                                 const rocsparse_int*        csr_col_ind,
                                                 const std::complex<double>* csr_val,
                                                 std::complex<double>*       ilu0,
                                                 rocsparse_index_base        idx_base,
                                                 size_t                      buffer_size,
                                                 void*                       buffer)
    {
        return rocsparse_zcsritilu0_compute(handle,
                                            alg,
                                            option,
                                            nmaxiter,
                                            tol,
                                            m,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            (const rocsparse_double_complex*)csr_val,
                                            (rocsparse_double_complex*)ilu0,
                                            idx_base,
                                            buffer_size,
                                            buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_history<float>(rocsparse_handle     handle,
                                                        rocsparse_itilu0_alg alg,
                                                        rocsparse_int*       niter,
                                                        float*               data,
                                                        size_t               buffer_size,
                                                        void*                buffer)
    {
        return rocsparse_scsritilu0_history(handle, alg, niter, data, buffer_size, buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_history<std::complex<float>>(rocsparse_handle     handle,
                                                                      rocsparse_itilu0_alg alg,
                                                                      rocsparse_int*       niter,
                                                                      float*               data,
                                                                      size_t buffer_size,
                                                                      void*  buffer)
    {
        return rocsparse_ccsritilu0_history(handle, alg, niter, data, buffer_size, buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_history<double>(rocsparse_handle     handle,
                                                         rocsparse_itilu0_alg alg,
                                                         rocsparse_int*       niter,
                                                         double*              data,
                                                         size_t               buffer_size,
                                                         void*                buffer)
    {
        return rocsparse_dcsritilu0_history(handle, alg, niter, data, buffer_size, buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_history<std::complex<double>>(rocsparse_handle     handle,
                                                                       rocsparse_itilu0_alg alg,
                                                                       rocsparse_int*       niter,
                                                                       double*              data,
                                                                       size_t buffer_size,
                                                                       void*  buffer)
    {
        return rocsparse_zcsritilu0_history(handle, alg, niter, data, buffer_size, buffer);
    }

} // namespace rocalution

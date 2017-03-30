/*
    Copyright (C) 2015 Tomas Flouri, Diego Darriba

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact: Tomas Flouri <Tomas.Flouri@h-its.org>,
    Exelixis Lab, Heidelberg Instutute for Theoretical Studies
    Schloss-Wolfsbrunnenweg 35, D-69118 Heidelberg, Germany
*/
#define PLLCUDA
#include "cuda_runtime.h"
#include "pll.h"

void *__gxx_personality_v0;

static unsigned int cuda_check(cudaError_t error_code, const char *msg)
{
  if (cudaSuccess != error_code) 
  {
    fprintf(stderr, "[libpll cuda error] [%s] [%s]\n", msg, cudaGetErrorString(error_code));
    return 0;
  }
  return 1;
}

__global__ void cu_update_partial(
    unsigned int states,
    unsigned int sites,
    unsigned int rate_cats,
    double *parent_clv,
    const double *left_clv,
    const double *right_clv,
    unsigned int *parent_scaler,
    unsigned int *left_scaler,
    unsigned int *right_scaler,
    double *lmat,
    double *rmat
    )

{
  int site = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int k, i, j;
  unsigned int offset = states * rate_cats * site;
  unsigned int scaling = parent_scaler ? 1 : 0;
  parent_clv += offset;
  left_clv += offset;
  right_clv += offset;
  if (site < sites)
  {
    
    if (parent_scaler != 0)
    {
      parent_scaler[site] = 0;
      if (left_scaler)
        parent_scaler[site] += left_scaler[site];
      if (right_scaler)
        parent_scaler[site] += right_scaler[site];
    }
    
    for (k = 0; k < rate_cats; ++k)
    {
      for (i = 0; i < states; ++i)
      {
        double terma = 0;
        double termb = 0;
        for (j = 0; j < states; ++j)
        {
          terma += lmat[j] * left_clv[j];
          termb += rmat[j] * right_clv[j];
        }
        parent_clv[i] = terma*termb;
        lmat += states;
        rmat += states;
        scaling &= parent_clv[i] < PLL_SCALE_THRESHOLD;
      }
      parent_clv += states;
      left_clv   += states;
      right_clv  += states;
    }
    if (scaling)
    {
      unsigned int span = states * rate_cats;
      parent_clv -= span;
      for (i = 0; i < span; ++i)
      {
        parent_clv[i] *= PLL_SCALE_THRESHOLD;
      }
      parent_scaler[site]++;
    }
  }
}


 PLL_EXPORT void pll_update_partial_ii_cuda(pll_partition_t * partition,
                            const pll_operation_t * op)
{
  
  // cpu variables
  const double * left_matrix = partition->pmatrix[op->child1_matrix_index];
  const double * right_matrix = partition->pmatrix[op->child2_matrix_index];
  double * parent_clv = partition->clv[op->parent_clv_index];
  unsigned int sites = partition->asc_bias_alloc ? 
    partition->sites + partition->states : partition->sites;
  unsigned int clv_size = sites * partition->states * partition->rate_cats;
  // gpu varables
  double * cuda_parent_clv = (double*)partition->cuda->clv[op->parent_clv_index];
  const double * cuda_left_clv = (double*)partition->cuda->clv[op->child1_clv_index];
  const double * cuda_right_clv = (double*)partition->cuda->clv[op->child2_clv_index];
  double *cuda_left_matrix = partition->cuda->lmatrix_buffer;
  double *cuda_right_matrix = partition->cuda->rmatrix_buffer;
  unsigned int * cuda_parent_scaler = (op->parent_scaler_index != PLL_SCALE_BUFFER_NONE) ?
    partition->cuda->scale_buffer[op->parent_scaler_index] : 0;
  unsigned int * cuda_left_scaler = (op->child1_scaler_index != PLL_SCALE_BUFFER_NONE) ?
    partition->cuda->scale_buffer[op->child1_scaler_index] : 0;
  unsigned int * cuda_right_scaler = (op->child2_scaler_index != PLL_SCALE_BUFFER_NONE) ?
    partition->cuda->scale_buffer[op->child2_scaler_index] : 0;
  // call kernel
  unsigned int block_size = 512;
  unsigned int grid_size = sites / block_size + 1;
  

  pll_cuda_memcpy_to_gpu(cuda_left_matrix, left_matrix, partition->states * partition->states * partition->rate_cats * sizeof(double));
  pll_cuda_memcpy_to_gpu(cuda_right_matrix, right_matrix, partition->states * partition->states * partition->rate_cats * sizeof(double));

  cu_update_partial << <grid_size, block_size >> > (
      partition->states,
      sites,
      partition->rate_cats,
      cuda_parent_clv,
      cuda_left_clv,
      cuda_right_clv,
      cuda_parent_scaler,
      cuda_left_scaler,
      cuda_right_scaler,
      cuda_left_matrix,
      cuda_right_matrix
      );
  cuda_check(cudaGetLastError(), "pll_update_partial_ii_cuda");
  // copy data to cpu (to rm ?)
  pll_cuda_memcpy_to_cpu(parent_clv, cuda_parent_clv, clv_size * sizeof(double)); 
}

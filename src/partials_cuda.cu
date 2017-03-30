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

__global__ void kernel_normal_layout(
    const double special_layout,
    double *normal_layout,
    unsigned int wrap_size,
    unsigned int sites,
    unsigned int rates_cats,
    unsigned int states)
{
  /*int clv_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (clv_idx >= sites * states * rate_cats)
    return; 
  int rate = (clv_idx / (states * 32)) % rate_cats;
  int state = (clv_idx % (states * 32)) / 32;
  mat_offset =  state * states + states * states * rate;
  */
}
__global__ void kernel_special_layout(
    const double *normal_layout,
    double *special_layout,
    unsigned int wrap_size,
    unsigned int sites,
    unsigned int rates_cats,
    unsigned int states)
{
  
}

__global__ void kernel_update_scalers(double *parent,
    unsigned int *parent_scaler,
    const unsigned int *left_scaler,
    const unsigned int *right_scaler,
    unsigned int sites,
    unsigned int span)
{
  int site = threadIdx.x + blockIdx.x * blockDim.x;
  if (site >= sites) {
    return;
  }
  parent_scaler[site] = 0;
  if (left_scaler)
    parent_scaler[site] += left_scaler[site];
  if (right_scaler)
    parent_scaler[site] += right_scaler[site];
  parent += span * site;
  unsigned int scaling = 1;
  for (unsigned int i = 0; i < span; ++i) {
    scaling = scaling && (parent[i] < PLL_SCALE_THRESHOLD);
  }
  if (scaling) {
    for (unsigned int i = 0; i < span; ++i) {
      parent[i] *= PLL_SCALE_FACTOR;
    }
    parent_scaler[site] += 1;
  }
}

__global__ void kernel_update_classic(double *parent_clv,
    const double *left_clv, 
    const double *right_clv, 
    const double *lmat, 
    const double *rmat,
    unsigned int sites, 
    unsigned int rate_cats, 
    unsigned int states)
{
  int clv_idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  unsigned int mat_offset = 0;
  int rate = (clv_idx / (states * 32)) % rate_cats;
  int state = (clv_idx % (states * 32)) / 32;
  mat_offset =  state * states + states * states * rate;
#ifdef MAT_USE_SHARED
  const int size = RATES * STATES * STATES;
  __shared__ double sh_mat1[size];
  __shared__ double sh_mat2[size];
  unsigned int ind = (clv_idx * 2) % size;
  sh_mat1[ind] = lmat[ind];
  sh_mat2[ind] = rmat[ind];
  ind++;
  sh_mat1[ind] = lmat[ind];
  sh_mat2[ind] = rmat[ind];
  __syncthreads();
  lmat = sh_mat1;
  rmat = sh_mat2;
#endif
  if (clv_idx >= sites * states * rate_cats)
    return;
#ifdef MAT_USE_CONST
  lmat = cuda_left_matrix_const;
  rmat = cuda_right_matrix_const;
#endif
  lmat += mat_offset;
  rmat += mat_offset;
  left_clv += clv_idx - (clv_idx % (states * 32)) + (clv_idx % 32); 
  right_clv += clv_idx - (clv_idx % (states * 32)) + (clv_idx % 32); 
  double lterm = 0;
  double rterm = 0;

  for (unsigned int st2 = 0; st2 < states; ++st2)
  {
#ifndef MAT_USE_TEXTURE 
    lterm += lmat[st2] * left_clv[st2*32];
    rterm += rmat[st2] * right_clv[st2*32];
#else
    int2 mat2int = tex1Dfetch(texture_left_matrix_int2, mat_offset + st2);
    lterm += __hiloint2double(mat2int.y, mat2int.x) * left_clv[st2] ;
    mat2int = tex1Dfetch(texture_right_matrix_int2, mat_offset + st2);
    rterm += __hiloint2double(mat2int.y, mat2int.x) * right_clv[st2] ;
#endif
  }
  parent_clv[clv_idx] = lterm * rterm;
}

__global__ void kernel_update_ll(unsigned int states,
    unsigned int sites,
    unsigned int rate_cats,
    const double *parent_clv,
    const unsigned int *parent_scaler,
    const double *child_clv,
    const unsigned int *child_scaler,
    const double *pmatrix,
    double *frequencies,
    const double *rate_weights,
    double *persite_ll)
{
  int site = threadIdx.x + blockIdx.x * blockDim.x;
  const double * clvp = parent_clv + states * rate_cats * (site - site % 32) + site % 32;
  const double * clvc = child_clv + states * rate_cats * (site - site % 32) + site % 32;
  const double * pmat = pmatrix;
  const double * freqs = frequencies;
  double terma, terma_r, termb;
  double site_lk;
  unsigned int scale_factors;

#ifdef MAT_USE_SHARED
  const int size = RATES * STATES * STATES;
  __shared__ double sh_mat1[size];
  unsigned int ind = (site * 2) % size;
  sh_mat1[ind] = pmat[ind];
  ind++;
  sh_mat1[ind] = pmat[ind];
  __syncthreads();
  pmat = sh_mat1;
#endif
  if (site >= sites) {
    return;
  }
  
  terma = 0;
  for (unsigned int i = 0; i < rate_cats; ++i)
  {
    terma_r = 0;
    for (unsigned int j = 0; j < states; ++j)
    {
      termb = 0;
      for (unsigned int k = 0; k < states; ++k)
      {
        termb += pmat[k] * clvc[k * 32];
      }
#ifdef FREQ_USE_CONST
      terma_r += clvp[j * 32] * cuda_frequencies_const[j] * termb;
#else
      terma_r += clvp[j * 32] * frequencies[j] * termb;
#endif
      pmat += states;
    }
    terma += terma_r * rate_weights[i];
    clvp += states * 32;
    clvc += states * 32;
  }
  scale_factors = (parent_scaler) ? parent_scaler[site] : 0;
  scale_factors += (child_scaler) ? child_scaler[site] : 0;
  site_lk = log(terma);
  if (scale_factors)
    site_lk += scale_factors * log(PLL_SCALE_THRESHOLD);
  persite_ll[site] = site_lk;
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
  

  pll_cuda_memcpy_to_gpu(cuda_left_matrix, left_matrix, partition->states * partition->states * partition->rate_cats * sizeof(double));
  pll_cuda_memcpy_to_gpu(cuda_right_matrix, right_matrix, partition->states * partition->states * partition->rate_cats * sizeof(double));

  unsigned int block_size = 1024;
  unsigned int grid_size = (partition->sites * partition->states_padded * partition->rate_cats) / block_size + 1;
  kernel_update_classic << <grid_size, block_size >> > (cuda_parent_clv,
      cuda_left_clv,
      cuda_right_clv,
      cuda_left_matrix,
      cuda_right_matrix,
      partition->sites,
      partition->rate_cats,
      partition->states);
  cuda_check(cudaGetLastError(), "update_partials gpu failed");
  /*
  if (cuda_parent_scaler) 
  {
    unsigned int grid_size_scalers = sites / block_size + 1;
    kernel_update_scalers << <grid_size_scalers, block_size >> > (cuda_parent_clv,
        cuda_parent_scaler,
        cuda_left_scaler,
        cuda_right_scaler,
        partition->sites,
        partition->rate_cats * partition->states);
    cuda_check(cudaGetLastError(), "update_scalers gpu failed");
  }
  */
    // copy data to cpu (to rm ?)
  pll_cuda_memcpy_to_cpu(parent_clv, cuda_parent_clv, clv_size * sizeof(double)); 
}

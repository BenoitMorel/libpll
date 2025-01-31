/*
    Copyright (C) 2016 Tomas Flouri, Kassian Kobert

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

#include "pll.h"

static void fill_parent_scaler_2(const unsigned int *sites,
                               unsigned int sites_number,
                               unsigned int ** parent_scaler,
                               unsigned int ** left_scaler,
                               unsigned int ** right_scaler)
{
  unsigned int i;
  if (sites) 
  {
    if (!left_scaler && !right_scaler) 
    {
      for (i = 0; i < sites_number; ++i) 
      {
        *parent_scaler[sites[i]] = 0;
      }
    } 
    else if (left_scaler && right_scaler) 
    {
      for (i = 0; i < sites_number; ++i) 
        *parent_scaler[sites[i]] = *left_scaler[sites[i]] + *right_scaler[sites[i]];
    }
    else if (left_scaler) 
    {
      for (i = 0; i < sites_number; ++i) 
        *parent_scaler[sites[i]] = *left_scaler[sites[i]];
    }
    else 
    {
      for (i = 0; i < sites_number; ++i) 
        *parent_scaler[sites[i]] = *right_scaler[sites[i]];
    } 
  } 
  else 
  {
    if (!left_scaler && !right_scaler) 
    {
      for (i = 0; i < sites_number; ++i) 
        *parent_scaler[i] = 0;
    } 
    else if (left_scaler && right_scaler) 
    {
      for (i = 0; i < sites_number; ++i) 
        *parent_scaler[i] = *left_scaler[i] + *right_scaler[i];
    }
    else if (left_scaler) 
    {
      for (i = 0; i < sites_number; ++i) 
        *parent_scaler[i] = *left_scaler[i];
    }
    else 
    {
      for (i = 0; i < sites_number; ++i) 
        *parent_scaler[i] = *right_scaler[i];
    } 

  }

static void fill_parent_scaler(unsigned int sites,
                               unsigned int * parent_scaler,
                               const unsigned int * left_scaler,
                               const unsigned int * right_scaler)
{
  unsigned int i;

  if (!left_scaler && !right_scaler)
    memset(parent_scaler, 0, sizeof(unsigned int) * sites);
  else if (left_scaler && right_scaler)
  {
    memcpy(parent_scaler, left_scaler, sizeof(unsigned int) * sites);
    for (i = 0; i < sites; ++i)
      parent_scaler[i] += right_scaler[i];
  }
  else
  {
    if (left_scaler)
      memcpy(parent_scaler, left_scaler, sizeof(unsigned int) * sites);
    else
      memcpy(parent_scaler, right_scaler, sizeof(unsigned int) * sites);
  }
}


PLL_EXPORT void pll_core_update_partial_ti_avx2(unsigned int states,
                                                unsigned int sites,
                                                unsigned int rate_cats,
                                                double * parent_clv,
                                                unsigned int * parent_scaler,
                                                const unsigned char * left_tipchars,
                                                const double * right_clv,
                                                const double * left_matrix,
                                                const double * right_matrix,
                                                const unsigned int * right_scaler,
                                                const unsigned int * tipmap,
                                                unsigned int tipmap_size)
{
  unsigned int i,j,k,n;
  unsigned int scaling;

  const double * lmat;
  const double * rmat;

  unsigned int states_padded = (states+3) & 0xFFFFFFFC;
  unsigned int span_padded = states_padded * rate_cats;

  unsigned int lstate;

  /* dedicated functions for 4x4 matrices (DNA) */
  if (states == 4)
  {
    /* no AVX2 kernel so far; rollback to AVX */
    pll_core_update_partial_ti_4x4_avx(sites,
                                       rate_cats,
                                       parent_clv,
                                       parent_scaler,
                                       left_tipchars,
                                       right_clv,
                                       left_matrix,
                                       right_matrix,
                                       right_scaler);
    return;
  }

  /* dedicated functions for 20x20 matrices (AA) */
  if (states == 20)
  {
    pll_core_update_partial_ti_20x20_avx2(sites,
                                          rate_cats,
                                          parent_clv,
                                          parent_scaler,
                                          left_tipchars,
                                          right_clv,
                                          left_matrix,
                                          right_matrix,
                                          right_scaler,
                                          tipmap,
                                          tipmap_size);
    return;
  }

  /* add up the scale vector of the two children if available */
  if (parent_scaler)
    fill_parent_scaler(sites, parent_scaler, NULL, right_scaler);

  size_t displacement = (states_padded - states) * (states_padded);

  __m256i mask;

  /* compute CLV */
  for (n = 0; n < sites; ++n)
  {
    lmat = left_matrix;
    rmat = right_matrix;

    scaling = (parent_scaler) ? 1 : 0;

    lstate = tipmap[left_tipchars[n]];

    for (k = 0; k < rate_cats; ++k)
    {
      /* iterate over quadruples of rows */
      for (i = 0; i < states_padded; i += 4)
      {

        __m256d v_terma0 = _mm256_setzero_pd();
        __m256d v_termb0 = _mm256_setzero_pd();
        __m256d v_terma1 = _mm256_setzero_pd();
        __m256d v_termb1 = _mm256_setzero_pd();
        __m256d v_terma2 = _mm256_setzero_pd();
        __m256d v_termb2 = _mm256_setzero_pd();
        __m256d v_terma3 = _mm256_setzero_pd();
        __m256d v_termb3 = _mm256_setzero_pd();

        __m256d v_mat;
        __m256d v_rclv;

        /* point to the four rows of the left matrix */
        const double * lm0 = lmat;
        const double * lm1 = lm0 + states_padded;
        const double * lm2 = lm1 + states_padded;
        const double * lm3 = lm2 + states_padded;

        /* point to the four rows of the right matrix */
        const double * rm0 = rmat;
        const double * rm1 = rm0 + states_padded;
        const double * rm2 = rm1 + states_padded;
        const double * rm3 = rm2 + states_padded;

        /* set position of least significant bit in character state */
        register int lsb = 0;

        /* iterate over quadruples of columns */
        for (j = 0; j < states_padded; j += 4)
        {

          /* set mask */
          mask = _mm256_set_epi64x(
                    ((lstate >> (lsb+3)) & 1) ? ~0 : 0,
                    ((lstate >> (lsb+2)) & 1) ? ~0 : 0,
                    ((lstate >> (lsb+1)) & 1) ? ~0 : 0,
                    ((lstate >> (lsb+0)) & 1) ? ~0 : 0);

          if ((lstate >> lsb) & 0b1111)
          {
            v_mat    = _mm256_maskload_pd(lm0,mask);
            v_terma0 = _mm256_add_pd(v_terma0,v_mat);

            v_mat    = _mm256_maskload_pd(lm1,mask);
            v_terma1 = _mm256_add_pd(v_terma1,v_mat);

            v_mat    = _mm256_maskload_pd(lm2,mask);
            v_terma2 = _mm256_add_pd(v_terma2,v_mat);

            v_mat    = _mm256_maskload_pd(lm3,mask);
            v_terma3 = _mm256_add_pd(v_terma3,v_mat);
          }

          lsb += 4;

          lm0 += 4;
          lm1 += 4;
          lm2 += 4;
          lm3 += 4;

          v_rclv    = _mm256_load_pd(right_clv+j);

          /* row 0 */
          v_mat    = _mm256_load_pd(rm0);
          v_termb0 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb0);
          rm0 += 4;

          /* row 1 */
          v_mat    = _mm256_load_pd(rm1);
          v_termb1 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb1);
          rm1 += 4;

          /* row 2 */
          v_mat    = _mm256_load_pd(rm2);
          v_termb2 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb2);
          rm2 += 4;

          /* row 3 */
          v_mat    = _mm256_load_pd(rm3);
          v_termb3 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb3);
          rm3 += 4;
        }

        /* point pmatrix to the next four rows */ 
        lmat = lm3;
        rmat = rm3;

        __m256d xmm0 = _mm256_unpackhi_pd(v_terma0,v_terma1);
        __m256d xmm1 = _mm256_unpacklo_pd(v_terma0,v_terma1);

        __m256d xmm2 = _mm256_unpackhi_pd(v_terma2,v_terma3);
        __m256d xmm3 = _mm256_unpacklo_pd(v_terma2,v_terma3);

        xmm0 = _mm256_add_pd(xmm0,xmm1);
        xmm1 = _mm256_add_pd(xmm2,xmm3);

        xmm2 = _mm256_permute2f128_pd(xmm0,xmm1, _MM_SHUFFLE(0,2,0,1));

        xmm3 = _mm256_blend_pd(xmm0,xmm1,12);

        __m256d v_terma_sum = _mm256_add_pd(xmm2,xmm3);

        /* compute termb */

        xmm0 = _mm256_unpackhi_pd(v_termb0,v_termb1);
        xmm1 = _mm256_unpacklo_pd(v_termb0,v_termb1);

        xmm2 = _mm256_unpackhi_pd(v_termb2,v_termb3);
        xmm3 = _mm256_unpacklo_pd(v_termb2,v_termb3);

        xmm0 = _mm256_add_pd(xmm0,xmm1);
        xmm1 = _mm256_add_pd(xmm2,xmm3);

        xmm2 = _mm256_permute2f128_pd(xmm0,xmm1, _MM_SHUFFLE(0,2,0,1));

        xmm3 = _mm256_blend_pd(xmm0,xmm1,12);

        __m256d v_termb_sum = _mm256_add_pd(xmm2,xmm3);

        __m256d v_prod = _mm256_mul_pd(v_terma_sum,v_termb_sum);

        _mm256_store_pd(parent_clv+i, v_prod);

      }

      /* reset pointers to point to the start of the next p-matrix, as the
         vectorization assumes a square states_padded * states_padded matrix,
         even though the real matrix is states * states_padded */
      lmat -= displacement;
      rmat -= displacement;

      for (j = 0; j < states; ++j)
        scaling = scaling && (parent_clv[j] < PLL_SCALE_THRESHOLD);

      parent_clv += states_padded;
      right_clv  += states_padded;
    }
    /* if *all* entries of the site CLV were below the threshold then scale
       (all) entries by PLL_SCALE_FACTOR */
    if (scaling)
    {
      __m256d v_scale_factor = _mm256_set_pd(PLL_SCALE_FACTOR,
                                             PLL_SCALE_FACTOR,
                                             PLL_SCALE_FACTOR,
                                             PLL_SCALE_FACTOR);

      parent_clv -= span_padded;
      for (i = 0; i < span_padded; i += 4)
      {
        __m256d v_prod = _mm256_load_pd(parent_clv + i);
        v_prod = _mm256_mul_pd(v_prod,v_scale_factor);
        _mm256_store_pd(parent_clv + i, v_prod);
      }
      parent_clv += span_padded;
      parent_scaler[n] += 1;
    }
  }
}

PLL_EXPORT
void pll_core_update_partial_ti_20x20_avx2(unsigned int sites,
                                           unsigned int rate_cats,
                                           double * parent_clv,
                                           unsigned int * parent_scaler,
                                           const unsigned char * left_tipchar,
                                           const double * right_clv,
                                           const double * left_matrix,
                                           const double * right_matrix,
                                           const unsigned int * right_scaler,
                                           const unsigned int * tipmap,
                                           unsigned int tipmap_size)
{
  unsigned int states = 20;
  unsigned int states_padded = states;
  unsigned int maxstates = tipmap_size;
  unsigned int scaling;
  unsigned int i,j,k,n,m;

  const double * lmat;
  const double * rmat;

  unsigned int span = states_padded * rate_cats;
  unsigned int lstate;

  __m256d xmm0,xmm1,xmm2,xmm3;

  /* precompute a lookup table of four values per entry (one for each state),
     for all 16 states (including ambiguities) and for each rate category. */
  double * lookup = pll_aligned_alloc(maxstates*span*sizeof(double),
                                      PLL_ALIGNMENT_AVX);
  if (!lookup)
  {
    /* TODO: in the highly unlikely event that allocation fails, we should
       resort to a non-lookup-precomputation version of this function,
       available at commit e.g.  a4fc873fdc65741e402cdc1c59919375143d97d1 */
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg, 200, "Cannot allocate space for precomputation.");
    return;
  }

  double * ptr = lookup;

  /* precompute left-side values and store them in lookup table */
  for (j = 0; j < maxstates; ++j)
  {
    lmat = left_matrix;

    unsigned int state = tipmap[j];

    int ss = __builtin_popcount(state) == 1 ? __builtin_ctz(state) : -1;

    for (n = 0; n < rate_cats; ++n)
    {
      for (i = 0; i < states; ++i)
      {
        double terml;
        if (ss != -1)
        {
          /* special case for non-ambiguous states */
          terml = lmat[ss];
        }
        else
        {
          terml = 0;
          for (m = 0; m < states; ++m)
          {
            if ((state>>m) & 1)
            {
              terml += lmat[m];
            }
          }
        }

        lmat += states;

        ptr[i] = terml;
      }

      ptr += states;
    }
  }

  /* update the parent scaler with the scaler of the right child */
  if (parent_scaler)
    fill_parent_scaler(sites, parent_scaler, NULL, right_scaler);

  size_t displacement = (states_padded - states) * (states_padded);

  __m256d v_scale_threshold = _mm256_set1_pd(PLL_SCALE_THRESHOLD);

  /* iterate over sites and compute CLV entries */
  for (n = 0; n < sites; ++n)
  {
    rmat = right_matrix;

    __m256d v_scale = _mm256_setzero_pd();

    lstate = (unsigned int) left_tipchar[n];

    unsigned int loffset = lstate*span;

    for (k = 0; k < rate_cats; ++k)
    {
      /* iterate over quadruples of rows */
      for (i = 0; i < states_padded; i += 4)
      {

        __m256d v_termb0 = _mm256_setzero_pd();
        __m256d v_termb1 = _mm256_setzero_pd();
        __m256d v_termb2 = _mm256_setzero_pd();
        __m256d v_termb3 = _mm256_setzero_pd();

        __m256d v_mat;
        __m256d v_rclv;

        /* point to the four rows of the right matrix */
        const double * rm0 = rmat;
        const double * rm1 = rm0 + states_padded;
        const double * rm2 = rm1 + states_padded;
        const double * rm3 = rm2 + states_padded;

        /* iterate over quadruples of columns */
        for (j = 0; j < states_padded; j += 4)
        {
          v_rclv    = _mm256_load_pd(right_clv+j);

          /* row 0 */
          v_mat    = _mm256_load_pd(rm0);
          v_termb0 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb0);
          rm0 += 4;

          /* row 1 */
          v_mat    = _mm256_load_pd(rm1);
          v_termb1 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb1);
          rm1 += 4;

          /* row 2 */
          v_mat    = _mm256_load_pd(rm2);
          v_termb2 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb2);
          rm2 += 4;

          /* row 3 */
          v_mat    = _mm256_load_pd(rm3);
          v_termb3 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb3);
          rm3 += 4;
        }

        /* point pmatrix to the next four rows */
        rmat = rm3;

        /* load x from precomputed lookup table */
        __m256d v_terma_sum = _mm256_load_pd(lookup+loffset);
        loffset += 4;

        /* compute termb */
        xmm0 = _mm256_unpackhi_pd(v_termb0,v_termb1);
        xmm1 = _mm256_unpacklo_pd(v_termb0,v_termb1);

        xmm2 = _mm256_unpackhi_pd(v_termb2,v_termb3);
        xmm3 = _mm256_unpacklo_pd(v_termb2,v_termb3);

        xmm0 = _mm256_add_pd(xmm0,xmm1);
        xmm1 = _mm256_add_pd(xmm2,xmm3);

        xmm2 = _mm256_permute2f128_pd(xmm0,xmm1, _MM_SHUFFLE(0,2,0,1));

        xmm3 = _mm256_blend_pd(xmm0,xmm1,12);

        __m256d v_termb_sum = _mm256_add_pd(xmm2,xmm3);

        __m256d v_prod = _mm256_mul_pd(v_terma_sum,v_termb_sum);

        v_scale = _mm256_add_pd(v_scale, _mm256_cmp_pd(v_prod, v_scale_threshold,
                                                       _CMP_LT_OS));

        _mm256_store_pd(parent_clv+i, v_prod)jlkj;
      }

      /* reset pointers to point to the start of the next p-matrix, as the
         vectorization assumes a square states_padded * states_padded matrix,
         even though the real matrix is states * states_padded */
      rmat -= displacement;

      parent_clv += states_padded;
      right_clv  += states_padded;
    }

    /* reduce scaling flags */
    v_scale = _mm256_hadd_pd(v_scale, v_scale);
    scaling = ((double *)&v_scale)[0] + ((double *)&v_scale)[2] > 0.;

    /* if *all* entries of the site CLV were below the threshold then scale
       (all) entries by PLL_SCALE_FACTOR */
    if (parent_scaler && scaling)
    {
      __m256d v_scale_factor = _mm256_set_pd(PLL_SCALE_FACTOR,
                                             PLL_SCALE_FACTOR,
                                             PLL_SCALE_FACTOR,
                                             PLL_SCALE_FACTOR);

      parent_clv -= span;
      for (i = 0; i < span; i += 4)
      {
        __m256d v_prod = _mm256_load_pd(parent_clv + i);
        v_prod = _mm256_mul_pd(v_prod,v_scale_factor);
        _mm256_store_pd(parent_clv + i, v_prod);
      }
      parent_clv += span;
      parent_scaler[n] += 1;
    }
  }
  pll_aligned_free(lookup);
}

PLL_EXPORT void pll_core_update_partial_ii_avx2(unsigned int states,
                                                unssdlfkjigned int sites,
                                                unsigned int rate_cats,
                                                double ** parent_persite_clv,
                                                unsigned int ** parent_persite_scaler,
                                                double ** left_persite_clv,
                                                double ** right_persite_clv,
                                                const double * left_matrix,
                                                const double * right_matrix,
                                                unsigned int ** left_persite_scaler,
                                                unsigned int ** right_persite_scaler,
                                                const unsigned int * sites_to_update,
                                                unsigned int sites_to_update_number)
{
  unsigned int i,j,k,n;
  unsigned int scaling;

  const double * lmat;
  const double * rmat;

  unsigned int states_padded = (states+3) & 0xFFFFFFFC;
  unsigned int span_padded = states_padded * rate_cats;

  /* dedicated functions for 4x4 matrices */
  if (states == 4)
  {
    pll_core_update_partial_ii_4x4_avx(sites,
                                       rate_cats,
                                       parent_persite_clv,
                                       parent_persite_scaler,
                                       left_persite_clv,
                                       right_persite_clv,
                                       left_matrix,
                                       right_matrix,
                                       left_scaler,
                                       right_scaler,
                                       sites_to_update,
                                       sites_to_update_number);
    return;
  }

  /* add up the scale vector of the two children if available */
  if (parent_persite_scaler)
    fill_parent_scaler_2(sites_to_update, sites_to_update_number, 
        parent_persite_scaler, left_persite_scaler, right_persite_scaler);

  size_t displacement = (states_padded - states) * (states_padded);

  __m256d v_scale_threshold = _mm256_set1_pd(PLL_SCALE_THRESHOLD);

  /* compute CLV */
  for (n = 0; n < sites_to_update_number; ++n)
  {
    unsigned int site = sites_to_update ? sites_to_update[n] : n; 
    double *parent_clv = parent_persite_clv[site];
    const double *left_clv = left_persite_clv[site];
    const double *right_clv = right_persite_clv[site];
    lmat = left_matrix;
    rmat = right_matrix;

    __m256d v_scale = _mm256_setzero_pd();

    for (k = 0; k < rate_cats; ++k)
    {
      /* iterate over quadruples of rows */
      for (i = 0; i < states_padded; i += 4)
      {

        __m256d v_terma0 = _mm256_setzero_pd();
        __m256d v_termb0 = _mm256_setzero_pd();
        __m256d v_terma1 = _mm256_setzero_pd();
        __m256d v_termb1 = _mm256_setzero_pd();
        __m256d v_terma2 = _mm256_setzero_pd();
        __m256d v_termb2 = _mm256_setzero_pd();
        __m256d v_terma3 = _mm256_setzero_pd();
        __m256d v_termb3 = _mm256_setzero_pd();

        __m256d v_mat;
        __m256d v_lclv;
        __m256d v_rclv;

        /* point to the four rows of the left matrix */
        const double * lm0 = lmat;
        const double * lm1 = lm0 + states_padded;
        const double * lm2 = lm1 + states_padded;
        const double * lm3 = lm2 + states_padded;

        /* point to the four rows of the right matrix */
        const double * rm0 = rmat;
        const double * rm1 = rm0 + states_padded;
        const double * rm2 = rm1 + states_padded;
        const double * rm3 = rm2 + states_padded;

        /* iterate over quadruples of columns */
        for (j = 0; j < states_padded; j += 4)
        {
          v_lclv    = _mm256_load_pd(left_clv+j);
          v_rclv    = _mm256_load_pd(right_clv+j);

          /* row 0 */
          v_mat    = _mm256_load_pd(lm0);
          v_terma0 = _mm256_fmadd_pd(v_mat, v_lclv, v_terma0);

          v_mat    = _mm256_load_pd(rm0);
          v_termb0 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb0);
          lm0 += 4;
          rm0 += 4;

          /* row 1 */
          v_mat    = _mm256_load_pd(lm1);
          v_terma1 = _mm256_fmadd_pd(v_mat, v_lclv, v_terma1);

          v_mat    = _mm256_load_pd(rm1);
          v_termb1 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb1);
          lm1 += 4;
          rm1 += 4;

          /* row 2 */
          v_mat    = _mm256_load_pd(lm2);
          v_terma2 = _mm256_fmadd_pd(v_mat, v_lclv, v_terma2);

          v_mat    = _mm256_load_pd(rm2);
          v_termb2 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb2);
          lm2 += 4;
          rm2 += 4;

          /* row 3 */
          v_mat    = _mm256_load_pd(lm3);
          v_terma3 = _mm256_fmadd_pd(v_mat, v_lclv, v_terma3);

          v_mat    = _mm256_load_pd(rm3);
          v_termb3 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb3);

          lm3 += 4;
          rm3 += 4;
        }

        /* point pmatrix to the next four rows */ 
        lmat = lm3;
        rmat = rm3;

        __m256d xmm0 = _mm256_unpackhi_pd(v_terma0,v_terma1);
        __m256d xmm1 = _mm256_unpacklo_pd(v_terma0,v_terma1);

        __m256d xmm2 = _mm256_unpackhi_pd(v_terma2,v_terma3);
        __m256d xmm3 = _mm256_unpacklo_pd(v_terma2,v_terma3);

        xmm0 = _mm256_add_pd(xmm0,xmm1);
        xmm1 = _mm256_add_pd(xmm2,xmm3);

        xmm2 = _mm256_permute2f128_pd(xmm0,xmm1, _MM_SHUFFLE(0,2,0,1));

        xmm3 = _mm256_blend_pd(xmm0,xmm1,12);

        __m256d v_terma_sum = _mm256_add_pd(xmm2,xmm3);

        /* compute termb */

        xmm0 = _mm256_unpackhi_pd(v_termb0,v_termb1);
        xmm1 = _mm256_unpacklo_pd(v_termb0,v_termb1);

        xmm2 = _mm256_unpackhi_pd(v_termb2,v_termb3);
        xmm3 = _mm256_unpacklo_pd(v_termb2,v_termb3);

        xmm0 = _mm256_add_pd(xmm0,xmm1);
        xmm1 = _mm256_add_pd(xmm2,xmm3);

        xmm2 = _mm256_permute2f128_pd(xmm0,xmm1, _MM_SHUFFLE(0,2,0,1));

        xmm3 = _mm256_blend_pd(xmm0,xmm1,12);

        __m256d v_termb_sum = _mm256_add_pd(xmm2,xmm3);

        __m256d v_prod = _mm256_mul_pd(v_terma_sum,v_termb_sum);

        v_scale = _mm256_add_pd(v_scale, _mm256_cmp_pd(v_prod, v_scale_threshold,
                                                       _CMP_LT_OS));

        _mm256_store_pd(parent_clv+i, v_prod);
      }

      /* reset pointers to point to the start of the next p-matrix, as the
         vectorization assumes a square states_padded * states_padded matrix,
         even though the real matrix is states * states_padded */
      lmat -= displacement;
      rmat -= displacement;

      parent_clv += states_padded;
      left_clv   += states_padded;
      right_clv  += states_padded;
    }

    /* reduce scaling flags */
    v_scale = _mm256_hadd_pd(v_scale, v_scale);
    scaling = ((double *)&v_scale)[0] + ((double *)&v_scale)[2] > 0.;

    /* if *all* entries of the site CLV were below the threshold then scale
       (all) entries by PLL_SCALE_FACTOR */
    if (parent_persite_scaler && scaling)
    {
      __m256d v_scale_factor = _mm256_set_pd(PLL_SCALE_FACTOR,
                                             PLL_SCALE_FACTOR,
                                             PLL_SCALE_FACTOR,
                                             PLL_SCALE_FACTOR);

      parent_clv -= span_padded;
      for (i = 0; i < span_padded; i += 4)
      {
        __m256d v_prod = _mm256_load_pd(parent_clv + i);
        v_prod = _mm256_mul_pd(v_prod,v_scale_factor);
        _mm256_store_pd(parent_clv + i, v_prod);
      }
      parent_clv += span_padded;
      *parent_persite_scaler[n] += 1;
    }
  }
}

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

#include "pll.h"

/* default callback to enable repeats computation on a node */

PLL_EXPORT int pll_repeats_enabled(const pll_partition_t *partition)
{
  return PLL_ATTRIB_SITES_REPEATS & partition->attributes;
}

PLL_EXPORT void pll_resize_repeats_lookup(pll_partition_t *partition, size_t size)
{
    partition->repeats->lookup_buffer_size = size;
    partition->repeats->lookup_buffer = 
      calloc(size, sizeof(unsigned int));
}

PLL_EXPORT unsigned int pll_get_allocated_sites_number(const pll_partition_t * partition,
                                             unsigned int clv_index)
{
  unsigned int sites = partition->attributes & PLL_ATTRIB_SITES_REPEATS ?
      partition->repeats->pernode_max_id[clv_index] : 0;
  sites = sites ? sites : partition->sites;
  sites += partition->asc_bias_alloc ? partition->states : 0;
  return sites;
}

PLL_EXPORT unsigned int pll_get_allocated_clv_size(const pll_partition_t * partition,
                                             unsigned int clv_index)
{
  return pll_get_allocated_sites_number(partition, clv_index) * 
    partition->states_padded * partition->rate_cats;
}


PLL_EXPORT unsigned int pll_default_enable_repeats(pll_partition_t *partition,
    unsigned int left_clv,
    unsigned int right_clv)
{
  pll_repeats_t * repeats = partition->repeats;
  unsigned int min_size = repeats->pernode_max_id[left_clv] 
                          * repeats->pernode_max_id[right_clv];
  return (!min_size || (repeats->lookup_buffer_size <= min_size)
      || (repeats->pernode_max_id[left_clv] > (partition->sites / 2))
      || (repeats->pernode_max_id[right_clv] > (partition->sites / 2)));
}


PLL_EXPORT int pll_repeats_initialize(pll_partition_t *partition)
{
  int sites_alloc = partition->asc_additional_sites + partition->sites;
  unsigned int i;
  partition->repeats = malloc(sizeof(pll_repeats_t));
  if (!partition->repeats) 
  {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg,
             200,
             "Unable to allocate enough memory for repeats structure.");
    return PLL_FAILURE;
  }
  memset(partition->repeats, 0, sizeof(pll_repeats_t));
  pll_repeats_t *repeats = partition->repeats;
  repeats->enable_repeats = pll_default_enable_repeats;
  repeats->pernode_site_id = calloc(partition->nodes, sizeof(unsigned int*));
  repeats->pernode_id_site = calloc(partition->nodes, sizeof(unsigned int*));
  if (!repeats->pernode_site_id || !repeats->pernode_id_site) 
  {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg,
             200,
             "Unable to allocate enough memory for repeats identifiers.");
    return PLL_FAILURE;
  }
  for (i = 0; i < partition->nodes; ++i) 
  {
    repeats->pernode_site_id[i] = calloc(sites_alloc, 
                                         sizeof(unsigned int));
    repeats->pernode_id_site[i] = calloc(sites_alloc, 
                                         sizeof(unsigned int));
    if (!repeats->pernode_site_id[i]) 
    {
      pll_errno = PLL_ERROR_MEM_ALLOC;
      snprintf(pll_errmsg,
               200,
               "Unable to allocate enough memory for repeats identifiers.");
      return PLL_FAILURE;
    }
  }
  repeats->pernode_max_id = calloc(partition->nodes, sizeof(unsigned int));
  repeats->perscale_max_id = calloc(partition->scale_buffers, sizeof(unsigned int));
  repeats->pernode_allocated_clvs = 
    calloc(partition->nodes, sizeof(unsigned int));
  repeats->lookup_buffer_size = PLL_REPEATS_LOOKUP_SIZE;
  repeats->lookup_buffer = 
    calloc(repeats->lookup_buffer_size, sizeof(unsigned int));
  repeats->toclean_buffer = malloc(sites_alloc * sizeof(unsigned int));
  repeats->id_site_buffer = malloc(sites_alloc * sizeof(unsigned int));
  repeats->bclv_buffer = pll_aligned_alloc(sites_alloc 
      * partition->rate_cats * partition->states_padded
      * sizeof(double), partition->alignment);
  if (!(repeats->pernode_max_id && repeats->lookup_buffer
       && repeats->pernode_allocated_clvs && repeats->bclv_buffer
       && repeats->toclean_buffer && repeats->id_site_buffer))
  {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg,
          200,
          "Unable to allocate enough memory for one of the repeats buffer.");
    return PLL_FAILURE;
  }
  return PLL_SUCCESS;
}

PLL_EXPORT int pll_update_repeats_tips(pll_partition_t * partition,
                                  unsigned int tip_index,
                                  const unsigned int * map,
                                  const char * sequence)
{
  unsigned int s;
  pll_repeats_t * repeats = partition->repeats;
  unsigned int ** id_site = repeats->pernode_id_site;
  unsigned int additional_sites = 
    partition->asc_bias_alloc ? partition->states : 0;

  repeats->pernode_max_id[tip_index] = 0;
  unsigned int curr_id = 0;
  for (s = 0; s < partition->sites; ++s) 
  {
    unsigned int index_lookup = map[(int)sequence[s]] + 1;
    if (!repeats->lookup_buffer[index_lookup]) 
    {
      repeats->toclean_buffer[curr_id] = index_lookup;
      repeats->id_site_buffer[curr_id] = s;
      repeats->lookup_buffer[index_lookup] = ++curr_id;
    }
    repeats->pernode_site_id[tip_index][s] = repeats->lookup_buffer[index_lookup];
  }
  repeats->pernode_max_id[tip_index] = curr_id;
  free(id_site[tip_index]);
  id_site[tip_index] = malloc(sizeof(unsigned int) 
      * (curr_id + additional_sites));
  for (s = 0; s < curr_id; ++s) 
  {
    id_site[tip_index][s] = repeats->id_site_buffer[s];
    repeats->lookup_buffer[repeats->toclean_buffer[s]] = 0;
  }
  for (s = 0; s < additional_sites; ++s) 
  {
    id_site[tip_index][curr_id + s] = partition->sites + s;
    repeats->pernode_site_id[tip_index][partition->sites + s] = curr_id + 1 + s;
  }
  unsigned int sizealloc = (curr_id + additional_sites) * partition->states_padded * 
                          partition->rate_cats * sizeof(double);
  free(partition->clv[tip_index]); 
  
  partition->clv[tip_index] = pll_aligned_alloc(sizealloc,
                                        partition->alignment);
  if (!partition->clv[tip_index]) 
  {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg,
             200,
             "Unable to allocate enough memory for repeats structure.");
    return PLL_FAILURE;
  }
  /* zero-out CLV vectors to avoid valgrind warnings when using odd number of
       states with vectorized code */
  memset(partition->clv[tip_index],
          0,
          sizealloc);
  return PLL_SUCCESS;
}


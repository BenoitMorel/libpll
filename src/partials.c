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

static void case_tiptip(pll_partition_t * partition,
                        const pll_operation_t * op)
{
  const double * left_matrix = partition->pmatrix[op->child1_matrix_index];
  const double * right_matrix = partition->pmatrix[op->child2_matrix_index];
  double * parent_clv = partition->clv[op->parent_clv_index];
  unsigned int * parent_scaler;
  unsigned int sites = partition->sites;

  /* ascertaiment bias correction */
  if (partition->asc_bias_alloc)
    sites += partition->states;

  /* get parent scaler */
  if (op->parent_scaler_index == PLL_SCALE_BUFFER_NONE)
    parent_scaler = NULL;
  else
    parent_scaler = partition->scale_buffer[op->parent_scaler_index];

  /* precompute lookup table */
  pll_core_create_lookup(partition->states,
                         partition->rate_cats,
                         partition->ttlookup,
                         left_matrix,
                         right_matrix,
                         partition->tipmap,
                         partition->maxstates,
                         partition->attributes);


  /* and update CLV at inner node */
  pll_core_update_partial_tt(partition->states,
                             sites,
                             partition->rate_cats,
                             parent_clv,
                             parent_scaler,
                             partition->tipchars[op->child1_clv_index],
                             partition->tipchars[op->child2_clv_index],
                             partition->tipmap,
                             partition->maxstates,
                             partition->ttlookup,
                             partition->attributes);
}

static void case_tipinner(pll_partition_t * partition,
                          const pll_operation_t * op)
{
  double * parent_clv = partition->clv[op->parent_clv_index];
  unsigned int tip_clv_index;
  unsigned int inner_clv_index;
  unsigned int tip_matrix_index;
  unsigned int inner_matrix_index;
  unsigned int * right_scaler;
  unsigned int * parent_scaler;
  unsigned int sites = partition->sites;

  /* ascertaiment bias correction */
  if (partition->asc_bias_alloc)
    sites += partition->states;

  /* get parent scaler */
  if (op->parent_scaler_index == PLL_SCALE_BUFFER_NONE)
    parent_scaler = NULL;
  else
    parent_scaler = partition->scale_buffer[op->parent_scaler_index];

  /* find which of the two child nodes is the tip */
  if (op->child1_clv_index < partition->tips)
  {
    tip_clv_index = op->child1_clv_index;
    tip_matrix_index = op->child1_matrix_index;
    inner_clv_index = op->child2_clv_index;
    inner_matrix_index = op->child2_matrix_index;
    if (op->child2_scaler_index == PLL_SCALE_BUFFER_NONE)
      right_scaler = NULL;
    else
      right_scaler = partition->scale_buffer[op->child2_scaler_index];
  }
  else
  {
    tip_clv_index = op->child2_clv_index;
    tip_matrix_index = op->child2_matrix_index;
    inner_clv_index = op->child1_clv_index;
    inner_matrix_index = op->child1_matrix_index;
    if (op->child1_scaler_index == PLL_SCALE_BUFFER_NONE)
      right_scaler = NULL;
    else
      right_scaler = partition->scale_buffer[op->child1_scaler_index];
  }

  pll_core_update_partial_ti(partition->states,
                             sites,
                             partition->rate_cats,
                             parent_clv,
                             parent_scaler,
                             partition->tipchars[tip_clv_index],
                             partition->clv[inner_clv_index],
                             partition->pmatrix[tip_matrix_index],
                             partition->pmatrix[inner_matrix_index],
                             right_scaler,
                             partition->tipmap,
                             partition->maxstates,
                             partition->attributes);
}

static void case_innerinner(pll_partition_t * partition,
                            const pll_operation_t * op)
{
  const double * left_matrix = partition->pmatrix[op->child1_matrix_index];
  const double * right_matrix = partition->pmatrix[op->child2_matrix_index];
  double ** parent_clv = partition->persite_clv[op->parent_clv_index];
  double ** left_clv = partition->persite_clv[op->child1_clv_index];
  double ** right_clv = partition->persite_clv[op->child2_clv_index];
  unsigned int ** parent_scaler;
  unsigned int ** left_scaler;
  unsigned int ** right_scaler;
  unsigned int additional_sites = partition->asc_bias_alloc ? partition->states : 0;
  unsigned int sites = partition->sites + additional_sites;
  unsigned int * sites_to_update = NULL; 
  unsigned int sites_to_update_number = sites; 

  if (partition->attributes & PLL_ATTRIB_SITES_REPEATS
      && partition->repeats && partition->repeats->pernode_max_id[op->parent_clv_index])
  {
    sites_to_update = partition->repeats->pernode_id_site[op->parent_clv_index];
    sites_to_update_number = partition->repeats->pernode_max_id[op->parent_clv_index] + additional_sites;
  }

  /* get parent scaler */
  if (op->parent_scaler_index == PLL_SCALE_BUFFER_NONE)
    parent_scaler = NULL;
  else
    parent_scaler = partition->persite_scales[op->parent_scaler_index];

  if (op->child1_scaler_index != PLL_SCALE_BUFFER_NONE)
    left_scaler = partition->persite_scales[op->child1_scaler_index];
  else
    left_scaler = NULL;

  /* if child2 has a scaler add its values to the parent scaler */
  if (op->child2_scaler_index != PLL_SCALE_BUFFER_NONE)
    right_scaler = partition->persite_scales[op->child2_scaler_index];
  else
    right_scaler = NULL;

  pll_core_update_partial_ii(partition->states,
                             sites,
                             partition->rate_cats,
                             parent_clv,
                             parent_scaler,
                             left_clv,
                             right_clv,
                             left_matrix,
                             right_matrix,
                             left_scaler,
                             right_scaler,
                             sites_to_update,
                             sites_to_update_number,
                             partition->attributes);
}


/* Fill the repeat structure in partition for the parent node of op */
static void update_repeats(pll_partition_t * partition,
                    const pll_operation_t * op) 
{
  pll_repeats_t * repeats = partition->repeats;
  unsigned int left = op->child1_clv_index;
  unsigned int right = op->child2_clv_index;
  unsigned int parent = op->parent_clv_index;
  unsigned int ** site_ids = repeats->pernode_site_id;
  unsigned int ** id_site = repeats->pernode_id_site;
  unsigned int * toclean_buffer = repeats->toclean_buffer;
  unsigned int * id_site_buffer = repeats->id_site_buffer;
  unsigned int curr_id = 0;
  unsigned int clv_size = partition->states_padded * 
                              partition->rate_cats * sizeof(double);
  unsigned int min_size = repeats->pernode_max_id[left] 
                          * repeats->pernode_max_id[right];
  unsigned int additional_sites = partition->asc_bias_alloc ? partition->states : 0;
  unsigned int sites_to_alloc;
  int scaler_index = op->parent_scaler_index;
  // in case site repeats is activated but not used for this node
  unsigned int userepeats = !(!min_size || REPEATS_LOOKUP_SIZE <= min_size);
  if (!userepeats)
  {
    sites_to_alloc = partition->sites + additional_sites;
    repeats->pernode_max_id[parent] = 0;
  } 
  else
  {
    // fill the parent repeats identifiers
    unsigned int s;
    for (s = 0; s < partition->sites; ++s) 
    {
      unsigned int index_lookup = (site_ids[left][s] - 1) 
        + (site_ids[right][s] - 1) * repeats->pernode_max_id[left];
      if (!repeats->lookup_buffer[index_lookup]) 
      {
        toclean_buffer[curr_id] = index_lookup;
        id_site_buffer[curr_id] = s;
        repeats->lookup_buffer[index_lookup] = ++curr_id;
      }
      site_ids[parent][s] = repeats->lookup_buffer[index_lookup];
    }
    repeats->pernode_max_id[parent] = curr_id;
    sites_to_alloc = curr_id + additional_sites;
    if (sites_to_alloc > repeats->pernode_allocated_clvs[parent]) {
      free(id_site[parent]);
      id_site[parent] = malloc((sizeof(unsigned int) * sites_to_alloc * 3)/2);
    }
    for (s = 0; s < curr_id; ++s) 
    {
      id_site[parent][s] = id_site_buffer[s];
      repeats->lookup_buffer[toclean_buffer[s]] = 0;
    }
    for (s = 0; s < additional_sites; ++s) 
    {
      id_site[parent][s + curr_id] = partition->sites + s;
    }
  }
  if (sites_to_alloc > repeats->pernode_allocated_clvs[parent]) 
  {
    // allocate too much to avoid too much allocations
    sites_to_alloc = (sites_to_alloc * 3) / 2;
    // but not more than the max needed
    if (sites_to_alloc > partition->sites + additional_sites) 
      sites_to_alloc = partition->sites + additional_sites;
      
    // reallocate clvs
    free(partition->clv[parent]);  
    partition->clv[parent] = pll_aligned_alloc(sites_to_alloc * clv_size,
                                            partition->alignment);
    // reallocate scales
    if (PLL_SCALE_BUFFER_NONE != scaler_index) 
    { 
      free(partition->scale_buffer[scaler_index]);
      partition->scale_buffer[scaler_index] = calloc(sites_to_alloc, sizeof(unsigned int));
    }
    if (!partition->clv[parent]) 
    {
      pll_errno = PLL_ERROR_MEM_ALLOC;
      snprintf(pll_errmsg,
               200,
               "Unable to allocate enough memory for repeats structure.");
    }
    
    // avoid valgrind errors
    memset(partition->clv[parent], 0, sites_to_alloc);
    unsigned int s;

    // set the lookups
    for (s = 0; s < partition->sites; ++s) 
    {
        partition->persite_clv[parent][s] = (partition->clv[parent]) 
            + (site_ids[parent][s] - 1) 
            * partition->states_padded * partition->rate_cats;
        if (PLL_SCALE_BUFFER_NONE != scaler_index) 
        {
          partition->persite_scales[scaler_index][s] = 
            partition->scale_buffer[scaler_index] + (site_ids[parent][s] - 1);
        }
    }
    for (s = 0; s < additional_sites; ++s) 
    {
        partition->persite_clv[parent][s + partition->sites] = (partition->clv[parent]) 
            + (curr_id + s) 
            * partition->states_padded * partition->rate_cats; 
        if (PLL_SCALE_BUFFER_NONE != scaler_index) 
        {
          partition->persite_scales[scaler_index][s + partition->sites] = 
            partition->scale_buffer[scaler_index] + (curr_id + s);
        }
    }
  }
}


PLL_EXPORT void pll_update_partials(pll_partition_t * partition,
                                    const pll_operation_t * operations,
                                    unsigned int count)
{
  unsigned int i;
  const pll_operation_t * op;



  for (i = 0; i < count; ++i)
  {
    op = &(operations[i]);
    
    
    if (partition->attributes & PLL_ATTRIB_SITES_REPEATS) {
      update_repeats(partition, op);
    }



    if (partition->attributes & PLL_ATTRIB_PATTERN_TIP)
    {
      if ((op->child1_clv_index < partition->tips) &&
          (op->child2_clv_index < partition->tips))
      {
        /* tip-tip case */
        case_tiptip(partition,op);
      }
      else if ((operations[i].child1_clv_index < partition->tips) ||
               (operations[i].child2_clv_index < partition->tips))
      {
        /* tip-inner */
        case_tipinner(partition,op);
      }
      else
      {
        /* inner-inner */
        case_innerinner(partition,op);
      }
    }
    else
    {
      /* inner-inner */
      case_innerinner(partition,op);
    }
  }
}


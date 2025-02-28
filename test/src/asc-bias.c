/*
 Copyright (C) 2015 Diego Darriba

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

 Contact: Diego Darriba <Diego.Darriba@h-its.org>,
 Exelixis Lab, Heidelberg Instutute for Theoretical Studies
 Schloss-Wolfsbrunnenweg 35, D-69118 Heidelberg, Germany
 */

/*
    asc-bias.c

    This test evaluates the ascertainment bias correction for DNA data on a
    large tree, taking scaling also into consideration.
    For each ascertainment bias correction type we evaluate the derivatives on
    6 branch lengths ranging from 1e-4 to 10
 */
#include "common.h"
#include <stdio.h>

#define MSA_FILENAME "testdata/2000.fas"
#define TRE_FILENAME "testdata/2000.tree"

#define STATES 4
#define MAX_RATE_CATS 16
#define NUM_BRANCH_LENGTHS 7

static double frequencies[4]  = { 0.1, 0.2, 0.3, 0.4 };
static double subst_params[6] = { 1, 5, 1, 1, 5, 1 };
static double categories[MAX_RATE_CATS] = {0};
static unsigned int params_indices[MAX_RATE_CATS] = {0};
static unsigned int invar_weights[STATES] = { 50, 40, 60, 20 };
static double test_branch_lengths[NUM_BRANCH_LENGTHS] =
                {0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0};

static unsigned int traversal_size, matrix_count, ops_count;
static pll_utree_t ** travbuffer;
static unsigned int * matrix_indices;
static double * branch_lengths;
static pll_operation_t * operations;

void print_travbuffer(pll_utree_t ** travbuffer, unsigned int len)
{
  unsigned int i;
  for (i=0; i<len; ++i)
    printf("%d(%d) ", travbuffer[i]->clv_index, travbuffer[i]->scaler_index);
  printf("\n");
}

void print_operations(pll_operation_t * operations, unsigned int len)
{
  unsigned int i;
  for (i=0; i<len; ++i)
    printf("%d[%d]+%d[%d]->%d ",
                      operations[i].child1_clv_index,
                      operations[i].child1_matrix_index,
                      operations[i].child2_clv_index,
                      operations[i].child2_matrix_index,
                      operations[i].parent_clv_index);
  printf("\n");
}

static double eval(pll_partition_t * partition,
                   pll_utree_t * tree,
                   double alpha)
{
  unsigned int i;
  double logl;
  double d_f, dd_f;
  double * sumtable;

  pll_set_subst_params(partition, 0, subst_params);
  pll_set_frequencies(partition, 0, frequencies);
  pll_compute_gamma_cats(alpha, partition->rate_cats, categories);
  pll_set_category_rates(partition, categories);

  pll_update_prob_matrices(partition,
                           params_indices,
                           matrix_indices,
                           branch_lengths,
                           matrix_count);
  pll_update_partials(partition, operations, ops_count);
  logl = pll_compute_edge_loglikelihood(partition,
                                        tree->clv_index,
                                        tree->scaler_index,
                                        tree->back->clv_index,
                                        tree->back->scaler_index,
                                        tree->pmatrix_index,
                                        params_indices,
                                        NULL);

  printf("Log-L: %f\n", logl);

  sumtable = pll_aligned_alloc(
    (partition->sites + partition->states) * partition->rate_cats * partition->states_padded *
    sizeof(double), partition->alignment);

  pll_update_sumtable(partition,
                      tree->clv_index,
                      tree->back->clv_index,
                      params_indices,
                      sumtable);

  double max_logl = -(1<<30);
  printf("%8s %18s %15s %15s\n", "Br.Len", "logLikelihood", "1st Deriv", "2nd Deriv");
  for (i=0; i<NUM_BRANCH_LENGTHS; ++i)
  {
    double branch_length = test_branch_lengths[i];
    if (!pll_compute_likelihood_derivatives(partition,
                                            tree->scaler_index,
                                            tree->back->scaler_index,
                                            branch_length,
                                            params_indices,
                                            sumtable,
                                            &d_f,
                                            &dd_f))
   {
     printf("Error computing likelihood derivatives\n");
     exit(1);
   }

   /* update logLikelihood */
   pll_update_prob_matrices(partition,
                            params_indices,
                            &(tree->pmatrix_index),
                            &branch_length,
                            1);
   logl = pll_compute_edge_loglikelihood(partition,
                                         tree->clv_index,
                                         tree->scaler_index,
                                         tree->back->clv_index,
                                         tree->back->scaler_index,
                                         tree->pmatrix_index,
                                         params_indices,
                                         NULL);

    printf("%8.4f %18.6f %15.8e %15.8e  ", branch_length, logl, d_f, dd_f);
    if (logl > max_logl)
    {
      max_logl = logl;
      printf("*");
    }
    printf("\n");
  }
  pll_aligned_free(sumtable);

  return logl;
}

int main(int argc, char * argv[])
{
  unsigned int attributes;
  pll_partition_t * partition;
  pll_utree_t * tree;
  unsigned int taxa_count, nodes_count, inner_nodes_count, branch_count;
  double alpha = 0.5;
  unsigned int rate_cats = 4;

  /* check attributes */
  attributes = get_attributes(argc, argv);

  tree = pll_utree_parse_newick(TRE_FILENAME, &taxa_count);
  printf("Read %s: %u taxa\n", TRE_FILENAME, taxa_count);

  inner_nodes_count = taxa_count - 2;
  nodes_count  = taxa_count + inner_nodes_count;
  branch_count = 2*taxa_count - 3;

  /* build fixed structures */
  travbuffer = (pll_utree_t **)malloc(nodes_count * sizeof(pll_utree_t *));
  branch_lengths = (double *)malloc(branch_count * sizeof(double));
  matrix_indices = (unsigned int *)malloc(branch_count * sizeof(unsigned int));
  operations = (pll_operation_t *)malloc(inner_nodes_count *
                                                sizeof(pll_operation_t));

  pll_utree_traverse(tree,
                     cb_full_traversal,
                     travbuffer,
                     &traversal_size);

  pll_utree_create_operations(travbuffer,
                              traversal_size,
                              branch_lengths,
                              matrix_indices,
                              operations,
                              &matrix_count,
                              &ops_count);

  printf("  Traversal buffer: "); print_travbuffer(travbuffer, traversal_size);
  printf("  Operations: ");       print_operations(operations, ops_count);

  /* test 1: no ascertainment bias correction */
  printf("\nTEST 1: NO ASC BIAS\n");
  partition = parse_msa(MSA_FILENAME, taxa_count, STATES, rate_cats, 1,
                        tree, attributes);
  printf("Read %s: %u sites\n", MSA_FILENAME, partition->sites);

  eval(partition, tree, alpha);

  pll_partition_destroy(partition);


  /* test 2: ascertainment bias correction */
  printf("\nTEST 2: ASC BIAS LEWIS\n");
  attributes |= PLL_ATTRIB_AB_LEWIS;
  partition = parse_msa(MSA_FILENAME, taxa_count, STATES, rate_cats, 1,
                        tree, attributes);

  pll_set_asc_bias_type(partition, PLL_ATTRIB_AB_LEWIS);
  eval(partition, tree, alpha);

  /* attempt to update invariant sites proportion. This should fail */
  if (pll_update_invariant_sites_proportion(partition, 0, 0.5))
  {
    printf("Error: Setting P-inv with ASC BIAS should fail");
    return 1;
  }

  /* test 2: ascertainment bias correction */
  printf("\nTEST 2: ASC BIAS FELSENSTEIN\n");
  pll_set_asc_bias_type(partition, PLL_ATTRIB_AB_FELSENSTEIN);
  pll_set_asc_state_weights(partition, invar_weights);
  eval(partition, tree, alpha);

  /* test 2: ascertainment bias correction */
  printf("\nTEST 2: ASC BIAS STAMATAKIS\n");
  pll_set_asc_bias_type(partition, PLL_ATTRIB_AB_STAMATAKIS);
  pll_set_asc_state_weights(partition, invar_weights);
  eval(partition, tree, alpha);

  /* clean */
  free(travbuffer);
  free(branch_lengths);
  free(operations);
  free(matrix_indices);
  pll_utree_destroy(tree);
  pll_partition_destroy(partition);

  return 0;
}

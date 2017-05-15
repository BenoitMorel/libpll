/*
    Copyright (C) 2016 Tomas Flouri

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

PLL_EXPORT
unsigned int pll_fastparsimony_edge_score_4x4_sse(const pll_parsimony_t * parsimony,
                                                  unsigned int node1_score_index,
                                                  unsigned int node2_score_index)
{
  unsigned int i;

  unsigned int bits[16] __attribute__ ((aligned(PLL_ALIGNMENT_SSE)));

  unsigned int * node1[4];
  unsigned int * node2[4];

  unsigned int ** vector = parsimony->packedvector;
  unsigned int vector_count = parsimony->packedvector_count;

  unsigned int score = 0;

  /* point to the parsimony vectors for each node and for each state */
  for (i = 0; i < 4; ++i)
  {
    node1[i] = vector[node1_score_index] + i*vector_count;
    node2[i] = vector[node2_score_index] + i*vector_count;
  }

  __m128i xmm0,xmm1,xmm2,xmm3,xmm4,xmm5,xmm6,xmm7;

  /* set all bits to one */
  xmm7 = _mm_set1_epi32(-1);

  for (i = 0; i < parsimony->packedvector_count; i += 4)
  {
    /* load, and, or bit vectors for state 0 */
    xmm0 = _mm_load_si128((__m128i *)(void *)(node1[0]+i));
    xmm1 = _mm_load_si128((__m128i *)(void *)(node2[0]+i));

    xmm2 = _mm_and_si128(xmm0,xmm1);

    /* load, and, or bit vectors for state 1 */
    xmm0 = _mm_load_si128((__m128i *)(void *)(node1[1]+i));
    xmm1 = _mm_load_si128((__m128i *)(void *)(node2[1]+i));

    xmm3 = _mm_and_si128(xmm0,xmm1);

    /* load, and, or bit vectors for state 2 */
    xmm0 = _mm_load_si128((__m128i *)(void *)(node1[2]+i));
    xmm1 = _mm_load_si128((__m128i *)(void *)(node2[2]+i));

    xmm4 = _mm_and_si128(xmm0,xmm1);

    /* load, and, or bit vectors for state 3 */
    xmm0 = _mm_load_si128((__m128i *)(void *)(node1[3]+i));
    xmm1 = _mm_load_si128((__m128i *)(void *)(node2[3]+i));

    xmm5 = _mm_and_si128(xmm0,xmm1);


    /* OR the ANDs of states 0 and 1 */
    xmm0 = _mm_or_si128(xmm2,xmm3);
    /* OR the ANDs of states 2 and 3 */
    xmm1 = _mm_or_si128(xmm4,xmm5);
    /* OR The two vectors */
    xmm6 = _mm_or_si128(xmm0,xmm1);


    xmm0 = _mm_andnot_si128(xmm6, xmm7);

    _mm_store_si128((__m128i *)(void *)bits, xmm0);

    score += (unsigned int)__builtin_popcount(bits[0]);
    score += (unsigned int)__builtin_popcount(bits[1]);
    score += (unsigned int)__builtin_popcount(bits[2]);
    score += (unsigned int)__builtin_popcount(bits[3]);
  }

  unsigned int score1 = parsimony->node_cost[node1_score_index];
  unsigned int score2 = parsimony->node_cost[node2_score_index];

  return score+score1+score2+parsimony->const_cost;
}

PLL_EXPORT
void pll_fastparsimony_update_vector_4x4_sse(pll_parsimony_t * parsimony,
                                             const pll_pars_buildop_t * op)
{
  unsigned int i;

  unsigned int bits[16] __attribute__ ((aligned(PLL_ALIGNMENT_SSE)));

  unsigned int * parent[4];
  unsigned int * child1[4];
  unsigned int * child2[4];

  unsigned int ** vector = parsimony->packedvector;
  unsigned int vector_count = parsimony->packedvector_count;

  unsigned int score = 0;

  /* point to the parsimony vectors for each node and for each state */
  for (i = 0; i < 4; ++i)
  {
    parent[i] = vector[op->parent_score_index] + i*vector_count;
    child1[i] = vector[op->child1_score_index] + i*vector_count;
    child2[i] = vector[op->child2_score_index] + i*vector_count;
  }

  __m128i xmm0,xmm1,xmm2,xmm3,xmm4,xmm5,xmm6,xmm7,xmm8,xmm9,xmm10,xmm11;

  /* set all bits to one */
  xmm11 = _mm_set1_epi32(-1);

  for (i = 0; i < parsimony->packedvector_count; i += 4)
  {
    /* load, and, or bit vectors for state 0 */
    xmm0 = _mm_load_si128((__m128i *)(void *)(child1[0]+i));
    xmm1 = _mm_load_si128((__m128i *)(void *)(child2[0]+i));

    xmm2 = _mm_and_si128(xmm0,xmm1);
    xmm3 = _mm_or_si128(xmm0,xmm1);

    /* load, and, or bit vectors for state 1 */
    xmm0 = _mm_load_si128((__m128i *)(void *)(child1[1]+i));
    xmm1 = _mm_load_si128((__m128i *)(void *)(child2[1]+i));

    xmm4 = _mm_and_si128(xmm0,xmm1);
    xmm5 = _mm_or_si128(xmm0,xmm1);

    /* load, and, or bit vectors for state 2 */
    xmm0 = _mm_load_si128((__m128i *)(void *)(child1[2]+i));
    xmm1 = _mm_load_si128((__m128i *)(void *)(child2[2]+i));

    xmm6 = _mm_and_si128(xmm0,xmm1);
    xmm7 = _mm_or_si128(xmm0,xmm1);

    /* load, and, or bit vectors for state 3 */
    xmm0 = _mm_load_si128((__m128i *)(void *)(child1[3]+i));
    xmm1 = _mm_load_si128((__m128i *)(void *)(child2[3]+i));

    xmm8 = _mm_and_si128(xmm0,xmm1);
    xmm9 = _mm_or_si128(xmm0,xmm1);

    
    /* OR the ANDs of states 0 and 1 */
    xmm0 = _mm_or_si128(xmm2,xmm4);
    /* OR the ANDs of states 2 and 3 */
    xmm1 = _mm_or_si128(xmm6,xmm8);
    /* OR The two vectors */
    xmm10 = _mm_or_si128(xmm0,xmm1);


    /* store them */
    xmm0 = _mm_andnot_si128(xmm10,xmm3);
    xmm1 = _mm_or_si128(xmm2,xmm0);
    _mm_store_si128((__m128i *)(void *)(parent[0]+i),xmm1);

    xmm0 = _mm_andnot_si128(xmm10,xmm5);
    xmm1 = _mm_or_si128(xmm4,xmm0);
    _mm_store_si128((__m128i *)(void *)(parent[1]+i),xmm1);

    xmm0 = _mm_andnot_si128(xmm10,xmm7);
    xmm1 = _mm_or_si128(xmm6,xmm0);
    _mm_store_si128((__m128i *)(void *)(parent[2]+i),xmm1);

    xmm0 = _mm_andnot_si128(xmm10,xmm9);
    xmm1 = _mm_or_si128(xmm8,xmm0);
    _mm_store_si128((__m128i *)(void *)(parent[3]+i),xmm1);


    xmm0 = _mm_andnot_si128(xmm10, xmm11);

    _mm_store_si128((__m128i *)(void *)bits, xmm0);

    score += (unsigned int)__builtin_popcount(bits[0]);
    score += (unsigned int)__builtin_popcount(bits[1]);
    score += (unsigned int)__builtin_popcount(bits[2]);
    score += (unsigned int)__builtin_popcount(bits[3]);
  }

  unsigned int score1 = parsimony->node_cost[op->child1_score_index];
  unsigned int score2 = parsimony->node_cost[op->child2_score_index];

  parsimony->node_cost[op->parent_score_index] = score+score1+score2;
}

PLL_EXPORT
void pll_fastparsimony_update_vector_sse(pll_parsimony_t * parsimony,
                                         const pll_pars_buildop_t * op)
{
  unsigned int i,j;
  unsigned int states = parsimony->states;

  unsigned int bits[16] __attribute__ ((aligned(PLL_ALIGNMENT_SSE)));

  unsigned int * parent;
  unsigned int * child1;
  unsigned int * child2;

  unsigned int ** vector = parsimony->packedvector;
  unsigned int vector_count = parsimony->packedvector_count;

  unsigned int score = 0;

  __m128i xmm0,xmm1,xmm2,xmm3,xmm4,xmm5;

  /* set all bits to one */
  xmm5 = _mm_set1_epi32(-1);

  for (i = 0; i < parsimony->packedvector_count; i += 4)
  {
    xmm4 = _mm_setzero_si128();

    /* load, and, or bit vectors for state 0 */
    child1 = vector[op->child1_score_index];
    child2 = vector[op->child2_score_index];
    for (j = 0; j < states; ++j)
    {
      xmm0 = _mm_load_si128((__m128i *)(void *)(child1+i));
      xmm1 = _mm_load_si128((__m128i *)(void *)(child2+i));

      xmm2 = _mm_and_si128(xmm0,xmm1);

      /* combine (OR) all ANDs for all states */
      xmm4 = _mm_or_si128(xmm4,xmm2);

      child1 += vector_count;
      child2 += vector_count;
    }

    child1 = vector[op->child1_score_index];
    child2 = vector[op->child2_score_index];
    parent = vector[op->parent_score_index];
    for (j=0; j<states; ++j)
    {
      /* load, and, or bit vectors for state j */
      xmm0 = _mm_load_si128((__m128i *)(void *)(child1+i));
      xmm1 = _mm_load_si128((__m128i *)(void *)(child2+i));

      xmm2 = _mm_and_si128(xmm0,xmm1);          /* vand */
      xmm3 = _mm_or_si128(xmm0,xmm1);           /* vor */

      xmm0 = _mm_andnot_si128(xmm4,xmm3);
      xmm1 = _mm_or_si128(xmm2,xmm0);
      _mm_store_si128((__m128i *)(void *)(parent+i),xmm1); 

      child1 += vector_count;
      child2 += vector_count;
      parent += vector_count;
    }
    xmm0 = _mm_andnot_si128(xmm4,xmm5);

    _mm_store_si128((__m128i *)(void *)bits, xmm0);

    score += (unsigned int)__builtin_popcount(bits[0]);
    score += (unsigned int)__builtin_popcount(bits[1]);
    score += (unsigned int)__builtin_popcount(bits[2]);
    score += (unsigned int)__builtin_popcount(bits[3]);
  }

  unsigned int score1 = parsimony->node_cost[op->child1_score_index];
  unsigned int score2 = parsimony->node_cost[op->child2_score_index];

  parsimony->node_cost[op->parent_score_index] = score+score1+score2;
}

PLL_EXPORT
unsigned int pll_fastparsimony_edge_score_sse(const pll_parsimony_t * parsimony,
                                              unsigned int node1_score_index,
                                              unsigned int node2_score_index)
{
  unsigned int i,j;
  unsigned int states = parsimony->states;

  unsigned int bits[16] __attribute__ ((aligned(PLL_ALIGNMENT_SSE)));

  unsigned int * node1;
  unsigned int * node2;

  unsigned int vector_count = parsimony->packedvector_count;
  unsigned int ** vector = parsimony->packedvector;

  unsigned int score = 0;

  __m128i xmm0,xmm1,xmm2,xmm4,xmm5;

  /* set all bits to one */
  xmm5 = _mm_set1_epi32(-1);

  for (i = 0; i < parsimony->packedvector_count; i += 4)
  {
    xmm4 = _mm_setzero_si128();

    /* load, and, or bit vectors for each state */
    node1 = vector[node1_score_index];
    node2 = vector[node2_score_index];
    for (j = 0; j < states; ++j)
    {
      xmm0 = _mm_load_si128((__m128i *)(void *)(node1+i));
      xmm1 = _mm_load_si128((__m128i *)(void *)(node2+i));

      xmm2 = _mm_and_si128(xmm0,xmm1);

      /* combine (OR) all ANDs for all states */
      xmm4 = _mm_or_si128(xmm4,xmm2);

      node1 += vector_count;
      node2 += vector_count;
    }

    xmm0 = _mm_andnot_si128(xmm4,xmm5);
    
    _mm_store_si128((__m128i *)(void *)bits, xmm0);

    score += (unsigned int)__builtin_popcount(bits[0]);
    score += (unsigned int)__builtin_popcount(bits[1]);
    score += (unsigned int)__builtin_popcount(bits[2]);
    score += (unsigned int)__builtin_popcount(bits[3]);
  }

  unsigned int score1 = parsimony->node_cost[node1_score_index];
  unsigned int score2 = parsimony->node_cost[node2_score_index];

  return score+score1+score2+parsimony->const_cost;
}

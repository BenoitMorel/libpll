# libpll

[![Build Status](https://travis-ci.org/xflouris/libpll.svg?branch=dev)](https://magnum.travis-ci.com/xflouris/libpll)
[![License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0.en.html)

## Introduction

The aim of this project is to implement a versatile high-performance software
library for phylogenetic analysis. The library should serve as a lower-level
interface of PLL (Flouri et al. 2014) and should have the following
properties:

* open source code with an appropriate open source license.
* 64-bit multi-threaded design that handles very large datasets.
* easy to use and well-documented.
* SIMD implementations of time-consuming parts.
* as fast or faster likelihood computations than RAxML (Stamatakis 2014).
* fast implementation of the site repeats algorithm (Kobert 2016).
* generic and clean design.
* Linux, Mac, and Microsoft Windows compatibility.

## Compilation instructions


Currently, `libpll` requires that [GNU Bison](http://www.gnu.org/software/bison/)
and [Flex](http://flex.sourceforge.net/) are installed on the target system. On
a Debian-based Linux system, the two packages can be installed using the command

`apt-get install flex bison`

The library also requires that a GNU system is available as it uses several
functions (e.g. `asprintf`) which are not present in the POSIX standard.
This, however might change in the future in order to have a more portable
and cross-platform library.

The library can be compiled using the included Makefile:

```bash
git clone https://github.com/xflouris/libpll.git
cd libpll
./autogen.sh
./configure
make
make install    # as root, otherwise run: sudo make install
```

The library will be installed on the operating system's standard paths.  For
some GNU/Linux distributions it might be necessary to add that standard path
(typically `/usr/local/lib`) to `/etc/ld.so.conf` and run `ldconfig`.

Microsoft Windows compatibility was tested with a cross-compiler and seems to
work out-of-the-box using [MingW](http://www.mingw.org/).

## Available functionality

libpll currently implements the General Time Reversible (GTR) model (Tavare
1986) which can be used for nucleotide and amino acid data. It supports models
of variable rates among sites, the Inv+&Gamma; (Gu et al. 1995) and has
functions for computing the discretized rate categories for the gamma model
(Yang 1994). Furthermore, it supports several methods for [pascertainment bias
correction](https://github.com/xflouris/libpll/wiki/Ascertainment-bias-correction)
(Kuhner et al. 2000, McGill et al. 2013, Lewis 2011, Leach&eacute; et al.
2015). Additional functionality includes tree visualization, functions for
parsimony (minimum mutation cost) calculation and ancestral state
reconstruction using Sankoff's method (Sankoff 1975, Sankof and Rousseau 1975).
The functions for computing partials, evaluating the log-likelihood and
updating transition probability matrices are vectorized using both SSE3 and AVX
instruction sets.

## Documentation

Please refer to the [wiki page](https://github.com/xflouris/libpll/wiki).

Below is a list of available functions in the current version.

### Partition (instance) manipulation

* `pll_partition_t * pll_partition_create(unsigned int tips, unsigned int clv_buffers, unsigned int states, unsigned int sites, unsigned int rate_matrices, unsigned int prob_matrices, unsigned int rate_cats, unsigned int scale_buffers, int attributes);`
* `void pll_partition_destroy(pll_partition_t * partition);`

### Linked lists

* `int pll_dlist_append(pll_dlist_t ** dlist, void * data);`
* `int pll_dlist_prepend(pll_dlist_t ** dlist, void * data);`
* `int pll_dlist_remove(pll_dlist_t ** dlist, void * data);`

### Models setup

* `void pll_set_subst_params(pll_partition_t * partition, unsigned int params_index, const double * params);`
* `void pll_set_frequencies(pll_partition_t * partition, unsigned int params_index, const double * frequencies);`
* `void pll_set_category_rates(pll_partition_t * partition, const double * rates);`
* `void pll_update_prob_matrices(pll_partition_t * partition, unsigned int params_index, unsigned int * matrix_indices, double * branch_lenghts, unsigned int count);`
* `int pll_set_tip_states(pll_partition_t * partition, unsigned int tip_index, const unsigned int * map, const char * sequence);`
* `void pll_set_tip_clv(pll_partition_t * partition, unsigned int tip_index, const double * clv);`
* `void pll_set_pattern_weights(pll_partition_t * partition, const unsigned int * pattern_weights);`
* `int pll_update_invariant_sites_proportion(pll_partition_t * partition, unsigned int params_index, double prop_invar);`
* `int pll_update_invariant_sites(pll_partition_t * partition);`
* `int pll_set_asc_bias_type(pll_partition_t * partition, int asc_bias_type);`
* `void pll_set_asc_state_weights(pll_partition_t * partition, const unsigned int * state_weights);`

### Likelihood computation

* `void pll_update_partials(pll_partition_t * partition, const pll_operation_t * operations, unsigned int count);`
* `double pll_compute_root_loglikelihood(pll_partition_t * partition, unsigned int clv_index, int scaler_index, unsigned int freqs_index);`
* `double pll_compute_edge_loglikelihood(pll_partition_t * partition, unsigned int parent_clv_index, int parent_scaler_index, unsigned int child_clv_index, int child_scaler_index, unsigned int matrix_index, unsigned int freqs_index);`

### Output functions

* `void pll_show_pmatrix(pll_partition_t * partition, unsigned int index, unsigned int float_precision);`
* `void pll_show_clv(pll_partition_t * partition, unsigned int index, int scaler_index, unsigned int float_precision);`

### Functions for parsing files

* `pll_fasta_t * pll_fasta_open(const char * filename, const unsigned int * map);`
* `int pll_fasta_getnext(pll_fasta_t * fd, char ** head, long * head_len, char ** seq, long * seq_len, long * seqno);`
* `void pll_fasta_close(pll_fasta_t * fd);`
* `long pll_fasta_getfilesize(pll_fasta_t * fd);`
* `long pll_fasta_getfilepos(pll_fasta_t * fd);`
* `int pll_fasta_rewind(pll_fasta_t * fd);`
* `pll_utree_t * pll_utree_parse_newick(const char * filename, unsigned int * tip_count);`
* `pll_rtree_t * pll_rtree_parse_newick(const char * filename, unsigned int * tip_count);`
* `void pll_utree_destroy(pll_utree_t * root);`
* `void pll_rtree_destroy(pll_rtree_t * root);`
* `pll_msa_t * pll_phylip_parse_msa(const char * filename, unsigned int * msa_count);`
* `void pll_msa_destroy(pll_msa_t * msa);`

### Tree manipulation functions

* `void pll_utree_show_ascii(pll_utree_t * tree);`
* `void pll_rtree_show_ascii(pll_rtree_t * tree);`
* `char * pll_utree_export_newick(pll_utree_t * root);`
* `char * pll_rtree_exprot_newick(pll_rtree_t * root);`
* `int pll_utree_query_tipnodes(pll_utree_t * root, pll_utree_t ** node_list);`
* `int pll_utree_query_innernodes(pll_utree_t * root, pll_utree_t ** node_list);`
* `void pll_utree_create_operations(pll_utree_t ** trav_buffer, unsigned int trav_buffer_size, double * branches, unsigned int * pmatrix_indices, pll_operation_t * ops, unsigned int * matrix_count, unsigned int * ops_count);`
* `int pll_rtree_query_tipnodes(pll_rtree_t * root, pll_rtree_t ** node_list);`
* `int pll_rtree_query_innernodes(pll_rtree_t * root, pll_rtree_t ** node_list);`
* `void pll_rtree_create_operations(pll_utree_t ** trav_buffer, unsigned int trav_buffer_size, double * branches, unsigned int * pmatrix_indices, pll_operation_t * ops, unsigned int * matrix_count, unsigned int * ops_count);`
* `int pll_utree_traverse(pll_utree_t * tree, int (*cbtrav)(pll_utree_t *), pll_utree_t ** outbuffer);`
* `int pll_rtree_traverse(pll_rtree_t * tree, int (*cbtrav)(pll_rtree_t *), pll_rtree_t ** outbuffer);`

### Core functions

* `void pll_core_create_lookup(unsigned int states, unsigned int rate_cats, double * lookup, const double * left_matrix, const double * right_matrix, unsigned int * tipmap, unsigned int tipmap_size);`

* `void pll_core_update_partial_tt(unsigned int states, unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const char * left_tipchars, const char * right_tipchars, const unsigned int * tipmap, unsigned int tipmap_size, const double * lookup, unsigned int attrib);`

* `void pll_core_update_partial_ti(unsigned int states, unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const char * left_tipchars, const double * right_clv, const double * left_matrix, const double * right_matrix, const unsigned int * right_scaler, const unsigned int * tipmap, unsigned int attrib);`

* `void pll_core_update_partial_ii(unsigned int states, unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const double * left_clv, const double * right_clv, const double * left_matrix, const double * right_matrix, const unsigned int * left_scaler, const unsigned int * right_scaler, unsigned int attrib);`

* `void pll_core_create_lookup_4x4(unsigned int rate_cats, double * lookup, const double * left_matrix, const double * right_matrix);`

* `void pll_core_update_partial_tt_4x4(unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const char * left_tipchars, const char * right_tipchars, const double * lookup);`

* `void pll_core_update_partial_ti_4x4(unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const char * left_tipchars, const double * right_clv, const double * left_matrix, const double * right_matrix, const unsigned int * right_scaler, unsigned int attrib);`

* `void pll_core_create_lookup_avx(unsigned int states, unsigned int rate_cats, double * lookup, const double * left_matrix, const double * right_matrix, unsigned int * tipmap, unsigned int tipmap_size);`

* `void pll_core_update_partial_tt_avx(unsigned int states, unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const char * left_tipchars, const char * right_tipchars, const double * lookup, unsigned int tipstates_count);`

* `void pll_core_update_partial_ti_avx(unsigned int states, unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const char * left_tipchars, const double * right_clv, const double * left_matrix, const double * right_matrix, const unsigned int * right_scaler, const unsigned int * tipmap);`

* `void pll_core_update_partial_ii_avx(unsigned int states, unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const double * left_clv, const double * right_clv, const double * left_matrix, const double * right_matrix, const unsigned int * left_scaler, const unsigned int * right_scaler);`

* `void pll_core_create_lookup_4x4_avx(unsigned int rate_cats, double * lookup, const double * left_matrix, const double * right_matrix);`

* `void pll_core_update_partial_tt_4x4_avx(unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const char * left_tipchars, const char * right_tipchars, const double * lookup);`

* `void pll_core_update_partial_ti_4x4_avx(unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const char * left_tipchar, const double * right_clv, const double * left_matrix, const double * right_matrix, const unsigned int * right_scaler);`

* `void pll_core_update_partial_ii_4x4_avx(unsigned int sites, unsigned int rate_cats, double * parent_clv, unsigned int * parent_scaler, const double * left_clv, const double * right_clv, const double * left_matrix, const double * right_matrix, const unsigned int * left_scaler, const unsigned int * right_scaler );`

### Auxiliary functions

* `int pll_compute_gamma_cats(double alpha, unsigned int categories, double * output_rates);`
* `void * pll_aligned_alloc(size_t size, size_t alignment);`
* `void pll_aligned_free(void * ptr);`
* `unsigned int * pll_compress_site_patterns(char ** sequence, int count, int * length);`

## Usage examples

Please refer to the [wiki page](https://github.com/xflouris/libpll/wiki) and/or the [examples directory](https://github.com/xflouris/libpll/tree/master/examples).

## libpll license and third party licenses

The code is currently licensed under the [GNU Affero General Public License version 3](http://www.gnu.org/licenses/agpl-3.0.en.html).

## Code

The code is written in C.

    File                   | Description
---------------------------|----------------
**compress.c**             | Functions for compressing alignment into site patterns.
**core_derivatives_avx.c** | AVX vectorized core functions for computing derivatives of the likelihood function.
**core_derivatives.c**     | Core functions for computing derivatives of the likelihood function.
**core_derivatives_sse.c** | SSE vectorized core functions for computing derivatives of the likelihood function.
**core_likelihood_avx.c**  | AVX vectorized core functions for computing the log-likelihood.
**core_likelihood.c**      | Core functions for computing the log-likelihood, that do not require partition instances.
**core_likelihood_sse.c**  | SSE vectorized core functions for computing the log-likelihood.
**core_partials_avx.c**    | AVX vectorized core functions for updating vectors of conditional probabilities (partials).
**core_partials.c**        | Core functions for updating vectors of conditional probabilities (partials).
**core_partials_sse.c**    | SSE vectorized core functions for updating vectors of conditional probabilities (partials).
**core_pmatrix_avx.c**     | AVX vectorized core functions for updating transition probability matrices.
**core_pmatrix.c**         | Core functions for updating transition probability matrices.
**core_pmatrix_sse.c**     | SSE vectorized core functions for updating transition probability matrices.
**derivatives.c**          | Functions for computing derivatives of the likelihood function.
**fasta.c**                | Functions for parsing FASTA files.
**gamma.c**                | Functions related to Gamma (&Gamma;) function and distribution.
**lex_phylip.l**           | Lexical analyzer for parsing phylip files.
**lex_rtree.l**            | Lexical analyzer for parsing newick rooted trees.
**lex_utree.l**            | Lexical analyzer for parsing newick unrooted trees.
**likelihood.c**           | Functions ofr computing the log-likelihood of a tree given a partition instance.
**list.c**                 | (Doubly) Linked-list implementations.
**Makefile**               | Makefile.
**maps.c**                 | Character mapping arrays for converting sequences to the internal representation.
**models.c**               | Model parameters related functions.
**output.c**               | Functions for output in terminal (i.e. conditional likelihood arrays, probability matrices).
**parse_phylip.y**         | Functions for parsing phylip files.
**parse_rtree.y**          | Functions for parsing rooted trees in newick format.
**parse_utree.y**          | Functions for parsing unrooted trees in newick format.
**parsimony.c**            | Parsimony functions.
**partials.c**             | Functions for updating vectors of conditional probabilities (partials).
**pll.c**                  | Functions for setting PLL partitions (instances).
**rtree.c**                | Rooted tree manipulation functions.
**utree.c**                | Unrooted tree manipulation functions.
**utree_moves.c**          | Functions for topological rearrangements on unrooted trees.
**utree_svg.c**            | Functions for SVG visualization of unrooted trees.

## Bugs

The source code in the master branch is thoroughly tested before commits.
However, mistakes may happen. All bug reports are highly appreciated.

## libpll core team

* Tom&aacute;&scaron; Flouri
* Diego Darriba
* Kassian Kobert
* Mark T. Holder
* Alexandros Stamatakis

## Contributing to libpll

Please read the section [Contributing to `libpll`](https://github.com/xflouris/libpll/wiki#contributing-to-libpll)
of the [wiki](https://github.com/xflouris/libpll/wiki).

## References

* Flouri T., Izquierdo-Carrasco F., Darriba D., Aberer AJ, Nguyen LT, Minh BQ, von Haeseler A., Stamatakis A. (2014)
**The Phylogenetic Likelihood Library.**
*Systematic Biology*, 64(2): 356-362.
doi:[10.1093/sysbio/syu084](http://dx.doi.org/10.1093/sysbio/syu084)

* Gu X., Fu YX, Li WH. (1995)
**Maximum Likelihood Estimation of the Heterogeneity of Substitution Rate among Nucleotide Sites.**
*Molecular Biology and Evolution*, 12(4): 546-557.

* Kobert K., Stamatakis A., Flouri T. (2016)
**Efficient detection of repeating sites to accelerate phylogenetic likelihood calculations.**
*Systematic Biology*, in press.
doi:[10.1093/sysbio/syw075](http://dx.doi.org/10.1093/sysbio/syw075)

* Leach&eacute; AL, Banbury LB, Felsenstein J., de Oca ANM, Stamatakis A. (2015)
**Short Tree, Long Tree, Right Tree, Wrong Tree: New Acquisition Bias Corrections for Inferring SNP Phylogenies.**
*Systematic Biology*, 64(6): 1032-1047.
doi:[10.1093/sysbio/syv053](http://dx.doi.org/10.1093/sysbio/syv053)

* Lewis LO. (2001)
**A Likelihood Approach to Estimating Phylogeny from Discrete Morphological Character Data.**
*Systematic Biology*, 50(6): 913-925.
doi:[10.1080/106351501753462876](http://dx.doi.org/10.1080/106351501753462876)

* Sankoff D. (1975)
**Minimal Mutation Trees of Sequences.**
*SIAM Journal on Applied Mathematics*, 28(1): 35-42.
doi:[10.1137/0128004](http://dx.doi.org/10.1137/0128004)

* Sankoff D, Rousseau P. (1975)
**Locating the Vertices of a Steiner Tree in Arbitrary Metric Space.**
*Mathematical Programming*, 9: 240-246.
doi:[10.1007/BF01681346](http://dx.doi.org/10.1007/BF01681346)

* Stamatakis A. (2014)
**RAxML version 8: a tool for phylogenetic analysis and post-analysis of large phylogenies.**
*Bioinformatics*, 30(9): 1312-1313.
doi:[10.1093/bioinformatics/btu033](http://dx.doi.org/10.1093/bioinformatics/btu033)

* Tavar&eacute; S. (1986)
**Some probabilistic and statistical problems in the analysis of DNA sequences.**
*American Mathematical Sciety: Lectures on Mathematics in the Life Sciences*, 17: 57-86.

* Yang Z. (2014)
**Maximum likelihood phylogenetic estimation from dna sequences with variable rates over sites: Approximate methods.**
*Journal of Molecular Evolution*, 39(3): 306-314.
doi:[10.1007/BF00160154](http://dx.doi.org/10.1007/BF00160154)

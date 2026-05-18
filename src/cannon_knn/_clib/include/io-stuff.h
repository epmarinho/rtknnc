/**@<io-stuff.h>::**/
/**********************************
 * AUTHOR: Eraldo Pereira Marinho *
 **********************************/
#pragma once
/** system headers: **/
#include <ctype.h>
#include <stdio.h>
#include <math.h>
#include <libgen.h>
#include <string.h>
#include <strings.h>
char           *strdup(const char *s);

/** project headers: **/
#include <macros.h>
#include <common-types.h>
#include <hyperquadtree.h>
#include <k-nn.h>
#include <errormsgs.h>
#ifdef _K_NN_GRAPH_
#    undef _SHOW_LEADERSHIP_GRAPH_
#    undef _SHOW_TREE_
#    undef _WRITE_LABELED_DATASET_
#endif
#ifdef _SHOW_LEADERSHIP_GRAPH_
#    undef _K_NN_GRAPH_
#    undef _SHOW_TREE_
#    undef _WRITE_LABELED_DATASET_
#endif
#ifdef _SHOW_TREE_
#    undef _SHOW_LEADERSHIP_GRAPH_
#    undef _K_NN_GRAPH_
#    undef _WRITE_LABELED_DATASET_
#endif
#ifdef _WRITE_LABELED_DATASET_
#    undef _SHOW_TREE_
#    undef _SHOW_LEADERSHIP_GRAPH_
#    undef _K_NN_GRAPH_
#endif
extern info_t **x;
extern dim_t    dimension;
extern size_t   n;
extern size_t   grade;
extern char    *cmdname;
extern char    *inputfname;
extern char    *outputprefix;
extern FILE    *input,
               *output;
int             skipseparators(FILE *);
size_t          getdata(FILE *);

// void            show_graphs(int n);
void            usage(FILE *output, const char *cmdname);
FILE           *cmdline_parser(int argc, const char *argv[]);

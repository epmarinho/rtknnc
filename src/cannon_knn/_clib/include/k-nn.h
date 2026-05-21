/**@<k-NN.h>::**/

#pragma once

/**********************************
 * AUTHOR: Eraldo Pereira Marinho *
 **********************************/

#pragma once
/** system headers: **/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
/** project-specific headers: **/
#include "hyperquadtree.h"
#include "macros.h"
#include "common-types.h"

/*****************************************************************************************
 * Subjective first person description:
 * WHO IS ME?
 * A query particle who knows my k-nearest-neighbors so that I can inquire them about
 * who are their individual k-nearest neighbor and then performing a deeper search
 * to contributing to close connections required to do the hierarchical cluster analysis.
 ****************************************************************************************/
typedef struct {
    ID_t           *list;   // Lists the neighbor identifiers (not indices).
    ID_t           *backward;   // Points to whom has considered myself one of their k-nearest neighbors.
    distance_t     *d;  // Synchronous to int int *list; -- it is the list of the k-nearest neighbor distances.
    size_t          k_visible;  // Number of visible particles amongst the K nearest ones. It is in principle used
    // in conjunction with kdbg or similar to take into account the fraction of my k-nearest
    // neighbors that were really viewd by myself along the roundtrip k-NN clustering.
    // A bad choice on the visibility model implies a low fraction of visible k-nearest
    // neighbors, or, otherwise, all the k nearest neighbors are always seen, which is also
    // wrong.
    size_t          hits;   // The number of incoming k-NN connections as the backward neighborings (my fan-club).
    distance_t      visibility; // The mean radius (square maybe) computed from double *d.
    distance_t      radius; // Maximum k-nearest neighbor distance: redundant with d[k-1].
    info_t          popularity; // It is my popularity, measured along with the backward poll
    // -- it can be taken as the reciprocal distance, 1/d, from the backward k-NN.
#ifdef __GRAPHVIZ_OUTPUT_SYNTAX__
    //     info_t      myvote;         // My vote goes to the top popular amongst my k-NN.
    // Better used in the leadership-inspired cluster detection.
    //     info_t      score;          // The score of the best voted member amongst my k nearest neighbors.
    //     ID_t        myleader;       // Stores the ID of the best voted k-NN member.
#endif
} KNN_struct;

extern KNN_struct *KNN;

extern KNN_struct *bkwrdKNN;

typedef struct {
    double          r;
    int             left;
    int             right;
    int             ID;
} KNN_indexer;

extern size_t   K;
extern size_t   K0;
extern size_t   KNN_population;

/***** k-NN methods ****/
extern void     KNN_sort(index_t);
extern void     knn_search(size_t);
extern void     symmetric_closure(size_t);
distance_t      query2point_distance(const info_t *, const info_t *);

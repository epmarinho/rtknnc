/**@<octree.h>::**/

/*******************************************************************************
 * Author: Eraldo Pereira Marinho, PhD
 * SPMD Quadtree code is a single program stream multiple data stream approach
 * to run on vector architectures like gpus etc or even in clusters.
 *******************************************************************************/

#pragma once

#include <strings.h>
#include "constants.h"

typedef struct __QDTREE__ {
    /* particle identifier is a positive integer */
    int             identifier;
    /* the number of particles belonging to the present quadrant */
    int             population;
    /* population mass */
    double          mass;
    /* node's size */
    double          size;
    /* vertex is the quadrant's division point or the leaf position */
    double          vertex[DIM];
    /* a cursor to the internal cluster, aka child cluster */
    int             child[MAXGRD];
    /* a parental link might be necessary */
    int             parent;
} QDTREE;

extern QDTREE   Octree[];
extern int      hashTB[NMAX][4];
extern int      levelbegin[MAXDEPTH];
extern int      levelend[MAXDEPTH];
extern int      depth;
extern double   mean[DIM];
extern double   Mtot;
extern double   mass[NMAX];
extern double   x[NMAX][DIM];
extern double   v[NMAX][DIM];
extern double   g[NMAX][DIM];

int             ihash(int i1, int i2);
void            ehash(int n);

void            Octree_build(int n);

/** time-step is required on performing treecode or treesph simulations **/
extern double   dt;
extern int      crossreference[];

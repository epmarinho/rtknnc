/**@<hyperquadtree.h>::**/
/*******************************************************************************
 * Author: Eraldo Pereira Marinho, PhD
 * SPMD Quadtree code is a single program stream multiple data stream approach
 * to run on vector architectures like gpus etc or even in clusters.
 *******************************************************************************/
#pragma once

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <macros.h>
#include <common-types.h>
#include <io-stuff.h>

#define ROOT  0
#define HASHTBCOLS  4
#define INDEX 0
#define EHASH 1
#define IHASH 2
#define LEAF  3

/** @HQDTREE::
 */
typedef struct __HQDTREE__ {
    /** @identifier:: particle identifier is a positive integer */
    ID_t            identifier;
    /** @population:: the number of particles belonging to the present quadrant */
    size_t          population;
    /** @vertex:: vertex is the quadrant's division point or the leaf position */
    info_t         *vertex; //[DIM];
    /** @child:: a cursor to the internal cluster, aka child cluster.
     *           use child[hyperquadrand] = -1 to mean empty */
    node_t         *child;
    /** @parent:: a parental link might be necessary */
    int             parent;
} HQDTREE;

extern HQDTREE *hyperquadtree;
extern index_t **hashTB;
extern node_t  *levelbegin;
extern node_t  *levelend;
extern size_t   depth;  // (0 <= depth && depth <= MAXDEPTH)
extern node_t  *crossreference;
void            ihash(int i1, int i2);
void            ehash(size_t n);
void            hyperquadtree_build(size_t n);
void            hyperquadtree_destroy(void);

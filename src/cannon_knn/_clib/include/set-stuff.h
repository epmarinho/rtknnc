/**@<set-stuff.h>::**/

/**********************************
 * AUTHOR: Eraldo Pereira Marinho *
 * 1st VERSION DATE: 23 Aug, 2018 *
 **********************************/

#pragma once

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <strings.h>
#include <common-types.h>
#include <macros.h>
#include <k-nn.h>

/*===============================================================
             SET OPERATIONS HAVE TO BE DECLARED HERE
 ===============================================================*/

typedef struct _SET_ {
    int             id;
    int             n;
    info_t          size;
    info_t         *center_of_mass;
    struct _SET_   *left,
                   *right;
} SET;

size_t          set__insert(index_t id, SET * t);
void            set__destroy(SET * t);
extern ID_t    *unclustered;
extern size_t   n_unclustered;

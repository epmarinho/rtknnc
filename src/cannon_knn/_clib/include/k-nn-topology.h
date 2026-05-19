/**@<k-nn-topology.h>::**/
/**********************************
 * AUTHOR: Eraldo Pereira Marinho *
 * 1st VERSION DATE: 17 Aug, 2018 *
 **********************************/
#pragma once
#include <stddef.h>
#include <io-stuff.h>
#include "show-KNN-graph.h"
#include "set-stuff.h"
#include "k-nn.h"

extern flag_t  *already_inserted;

/** @k-nn-topology.h::
 * the following directive was introduced to doing a rollback test of the primitive idea of the method.
 * NOTE: turn it off before using the newest methods
 */
//#define __ORIGINAL_ROUNDTRIP_CLUSTER_RECOGNITION__

void            KNNbacktrip(index_t p); //, index_t q);

size_t          round_trip_KNN_explore(index_t q);

void            roundtrip_KNN_cluster_partition(size_t n);

void            showresults(void);

flag_t          tallyho(index_t, index_t, distance_t);

extern size_t   clcount;

extern SET    **cluster;

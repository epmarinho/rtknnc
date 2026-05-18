/**@<k-nn-topology.c>::**/

/*=====================================================================================
 *  Originally developed by Eraldo Pereira Marinho, Ph.D.
 *  Creation date: 17 Aug, 2018
 *  Last modification date: 29 Aug, 2018, at 12:00 GMT -3 (Sao Paulo local time)
 *
 *  The round-trip KNN cluster (RTKNNC), initiated from a query particle [q], is a
 *  heuristic approach to find out which particles share the neighbors k nearest [q].
 *  Such an approach is recursively propagated until there are no more particles to
 *  insert into the RTKNNC.
 *
 *  Further contributions to the present module shall be addressed hereafter by adding
 *  comment lines below:
 *
 * 20 Sep, 2018, at 17:21 GMT -3:
 *      Once introduced the procedure void KNNbacktrip(index_t q_index);
 *      it is nolonger necessary to perform an exhaustive recursion trhough all
 *      the nearest neighbors of q_index. The old code segment was removed and moved
 *      in to the file k-NN-topology-removed-piece.c in the src directory
 *
 * =====================================================================================*/

/*** Global definitions. Uncomment your choices. All commented returns to the original method: ***/

/*** Show the KNN graph using the GraphViz syntax: ***/
#define __GRAPHVIZ_OUTPUT_SYNTAX__

/*** Check reciprocity before including the particle as a cluster head:
    Default: commented ***/
// #define __CHECK_RECIPROCITY_BEFORE_INCLUDING__

/*** Ignore starting from an unclustered particle:
    Default: commented ***/
// #define __IGNORE_HEADING_UNCLUSTERED__

/*** Refuse singleton cluster:
    Default: all commented ***/
// #ifndef __CHECK_RECIPROCITY_BEFORE_INCLUDING__
// #   define __REFUSE_SINGLETON_CLUSTER__
// #else
// #   undef  __REFUSE_SINGLETON_CLUSTER__
// #endif
// #define __REFUSE_SINGLETON_CLUSTER__

/*** Emulate visual range ***/
#define __EMULATE_VISIBILITY__

// #define __RMS_VISIBILITY__
#define __USE_VISIBILITY_TOL_METHOD__

#ifdef __USE_VISIBILITY_TOL_METHOD__
#    define VISIBTOL1 1./16.
#    define VISIBTOL2 1./32.
#endif

/*** refuse clusters with less than K points within: ***/
#define __REFUSE_LESS_THAN_K_POINT_CLUSTERS__

#ifndef __EMULATE_VISIBILITY__
#    undef __REFUSE_LESS_THAN_K_POINT_CLUSTERS__
#endif
#ifndef __REFUSE_LESS_THAN_K_POINT_CLUSTERS__
#    undef __RECLUSTER_BY_PSEUDO_KMEANS__
#endif

#ifdef __REFUSE_LESS_THAN_K_POINT_CLUSTERS__
/*** Apply pseudo k-means to regroup unclustered (low connectivity refused) particles: ***/
#    define __RECLUSTER_BY_PSEUDO_KMEANS__
#    ifdef __RECLUSTER_BY_PSEUDO_KMEANS__
/*** choose only one, or none, of the weighted distance method: ***/
// #       define __WEIGHTED_INV_SQRT_N_DISTANCE__
// #       define __WEIGHTED_INV_N_DISTANCE__
#        if defined(__WEIGHTED_INV_N_DISTANCE__) && defined(__WEIGHTED_INV_SQRT_N_DISTANCE__)
#            undef __WEIGHTED_INV_N_DISTANCE__
#            undef __WEIGHTED_INV_SQRT_N_DISTANCE__
#        endif
#    endif
#else
#    undef __RECLUSTER_BY_PSEUDO_KMEANS__
#    undef __WEIGHTED_INV_SQRT_N_DISTANCE__
#    undef __WEIGHTED_INV_N_DISTANCE__
#endif
/*** turn on neighborhood popularity measuring ***/
#define __COMPUTE_NEIGHBORHOOD_POPULARITY__
/***************************************************************************************
 * Last changes has began on 03 Mar 2020:
 * Radical change: the code is now simulating the previous idea of the full round-trip
 * KNN until exhaustion -- see function roundtrip_KNN_find below.
 **************************************************************************************/

#include <k-nn-topology.h>

/*** clcount reports the number of identified clusters for the
     currently selected number of nearest neighbors, K ***/
size_t          clcount = 0;

/*** already inserted array is a way of locking a particle to prevent
     it to be assigned to more than one cluster ***/
index_t        *already_inserted;

/*** the cluster itself is a set abstract data type: ***/
SET           **cluster;

/*** lists the set content in order: ***/
void cluster_list(SET * set)
{
    int             j;

    if (set->left != NULL) {
        cluster_list(set->left);
    }

    fprintf(output, "%i %i\t", already_inserted[set->id - 1], set->id);
    //     fprintf(output, "%zu %i\t", clcount + 1, set->id);

    for (j = 0; j < dimension - 1; j++) {
        fprintf(output, "%lg\t", x[set->id - 1][j]);
    }

    fprintf(output, "%lg\n", x[set->id - 1][j]);

    if (set->right != NULL) {
        cluster_list(set->right);
    }
}

/** @cluster_insert -> set__insert as if cluster were a c++ object. A cluster is abstracted
 * as a pointer-linked binary tree to ease the insert and list methods */
int cluster_insert(index_t clcount, ID_t queryID)
{
    index_t         q = queryID - 1;

    if (cluster[clcount] == NULL) {

        /*** on the first run, create the root-datastructure which simulates
             a cluster as a ID set of the 1-pass roundtrip KNN members: ***/
        cluster[clcount] = calloc(1, sizeof(SET));
        cluster[clcount]->n = 1;
        cluster[clcount]->id = queryID;
        /** @cluster[clcount]->size @cluster_insert:: still in tests */
        cluster[clcount]->size = KNN[q].visibility;
        /****/
        cluster[clcount]->center_of_mass =
            memcpy(cluster[clcount]->center_of_mass = calloc(sizeof(double), dimension), x[q], dimension * sizeof(double)
            );

    } else if (set__insert(queryID, cluster[clcount])) {
        cluster[clcount]->size = (cluster[clcount]->size * (cluster[clcount]->n - 1) + KNN[q].visibility) / cluster[clcount]->n;
    } else {
        return 0;
    }

    /*------------------------------------------------------------------------
     *      The just inserted point must be marked as already included.
     *      clcount + 1 works as a cluster descriptor for which a particle is
     *      allocated
     *     ------------------------------------------------------------------------*/
    already_inserted[q] = clcount + 1;

    return cluster[clcount]->n;
}   // end cluster_insert

void do_backward_connection(size_t n)
{
    /*** It is better to clean up any possible garbage before computing the k-NN's symmetric closure ***/
    for (index_t q = 0; q < n; q++) {
        KNN[q].hits = KNN[q].popularity = KNN[q].score = 0;
    }
    for (index_t q = 0; q < n; q++) {
        for (index_t j = 0; j < K; j++) {
            index_t         r = KNN[q].list[j] - 1;

            KNN[r].hits++;
#ifdef __COMPUTE_NEIGHBORHOOD_POPULARITY__
            /*** popularity intensity is defined as the incoming-flux density ***/
            KNN[r].popularity += 1.0 / KNN[q].d[j];
#endif
        }
    }
    for (index_t q = 0; q < n; q++) {
        KNN[q].backward = realloc(KNN[q].backward, KNN[q].hits * sizeof(*KNN[q].backward));
        KNN[q].hits = 0;
    }
    for (index_t q = 0; q < n; q++) {
        ID_t            qID = q + 1;

        for (index_t j = 0; j < K; j++) {
            ID_t            pID = KNN[q].list[j];
            index_t         p = pID - 1;

            KNN[p].backward[KNN[p].hits++] = qID;
        }
    }
    /***
        visibility is by now defined as the mean squared distance (otherwise RMS)
        from the query to its k-NN members -- visibility is a questionable bio-inspired
        model and it shall be exhaustively tested
    ***/
#ifdef __EMULATE_VISIBILITY__
    for (index_t q = 0; q < n; q++) {
        KNN[q].visibility = 0;
        int             j;

        for (KNN[q].visibility = j = 0; j < K; j++) {
#    ifdef __RMS_VISIBILITY__
            KNN[q].visibility += pow(KNN[q].d[j], 2);
        }
        KNN[q].visibility = sqrt(KNN[q].visibility /= j);
#    else
            KNN[q].visibility += KNN[q].d[j];
        }
        KNN[q].visibility /= j;
#    endif

        KNN[q].visibility *= (1 + KNN[q].popularity * KNN[q].radius / n);
    }
#endif
}   // end do_backward_connection(size_t n)

/** @KNNbacktrip::
 *   exhaustively visits the backward neighbors of p if p != q */
void KNNbacktrip(index_t q)
{
    /*** if KNN[p].hits == 0, there is nothing to backward to ***/
    if (KNN[q].hits == 0)
        return;

    /*** k-NN back-trip starts here: **/
    for (int i = 0; i < KNN[q].hits; i++) {
        ID_t            pID = KNN[q].backward[i];
        index_t         p = pID - 1;

        if (already_inserted[p]) {
            /*** condition inserted on 28 Feb 2020 14:22 -03 ***/
            continue;
        }
#ifdef __EMULATE_VISIBILITY__
        if (KNN[p].hits == 0) {
            roundtrip_KNN_find(p);
        } else {
            ID_t            qID = q + 1;
            double          dpq = 0;

            for (int j = 0; j < K; j++) {
                if (KNN[p].list[j] == qID) {
                    dpq = KNN[p].d[j];
                    break;
                }
            }
            /*** backward trip only if [q] is "seen" by [p] ***/
#    ifdef __USE_VISIBILITY_TOL_METHOD__
            if (dpq <= VISIBTOL1 * KNN[p].radius + KNN[p].visibility) {
#    else
            if (dpq <= KNN[p].visibility) {
#    endif
                roundtrip_KNN_find(p);
            }
        }
#else
        roundtrip_KNN_find(p);
#endif
    }
}   // end KNNbacktrip(index_t q)

/** @KNNtrip::
 * This is in principle a non-deterministic
 * method of simulating the entering in the
 * K states
 */
void KNNtrip(index_t q)
{
    for (int j = 0; j < K; j++) {
        index_t         p = KNN[q].list[j] - 1;

        if (already_inserted[p])
            continue;
#ifdef __EMULATE_VISIBILITY__
        /*** forward trip only if [q] "sees" [p] ***/
        double          dpq = KNN[q].d[j];

#    ifdef __USE_VISIBILITY_TOL_METHOD__
        if (dpq <= VISIBTOL2 * KNN[q].radius + KNN[q].visibility) {
#    else
        if (dpq <= KNN[q].visibility) {
#    endif
            roundtrip_KNN_find(p);
        }
#else
        roundtrip_KNN_find(p);
#endif
    }
}   // end KNNtrip

#ifdef __CHECK_RECIPROCITY_BEFORE_INCLUDING__
int check_reciprocity(index_t q)
{
    /*** check reciprocity: at least one of the k-nearest neighbors of
            q must acknowledges back q ***/
    int             j;

    for (j = 0; j < K; j++) {
        index_t         p = KNN[q].list[j] - 1;
        int             i;

        for (i = 0; i < K; i++) {
            index_t         r = KNN[p].list[i] - 1;

            if (q == r) {
                break;
            }
        }
        if (i < K) {
            break;
        }
    }
    if (j == K) {
        return 0;
    }
    return 1;
}   // end check_reciprocity
#endif

size_t roundtrip_KNN_find(index_t q)
{
    /*** This is the query ID number/token: ***/
    int             queryID = q + 1;

    if (already_inserted[q]) {
        ;
    } else {

#ifdef __CHECK_RECIPROCITY_BEFORE_INCLUDING__
        if (check_reciprocity(q) == 0) {
            return 0;
        }
#endif

        cluster_insert(clcount, queryID);

        /*** do the non-deterministic k-NN trip ***/
        KNNtrip(q);

        /*** Modified on Tue 03 Mar 2020 ***/
        KNNbacktrip(q);
    }

    /*** return the population number of the just detected cluster[clcount]: ***/
    return cluster[clcount]->n;
}   // end roundtrip_KNN_find(index_t q)

/*********************************************
 * Multi layer, hierarchical cluster analysis *
 *********************************************/

flag_t          verbose = 1;

ID_t           *unclustered;

size_t          n_unclustered = 0;

double distance(info_t * a, info_t * b)
{
    double          d_ab = 0;

    for (int j = 0; j < dimension; j++) {
        d_ab += (a[j] - b[j]) * (a[j] - b[j]);
    }
    return d_ab;
}   // end distance

index_t do_pseudo_k_means(size_t k, index_t r)
{
    index_t         nearest_cluster = -1;
    double          mindist = INFINITY;

    for (index_t cl = 0; cl < k; cl++) {
#ifdef __WEIGHTED_INV_N_DISTANCE__
        double          d_cl = distance(x[r], cluster[cl]->center_of_mass) / cluster[cl]->n;
#elif defined(__WEIGHTED_INV_SQRT_N_DISTANCE__)
        double          d_cl = distance(x[r], cluster[cl]->center_of_mass) / sqrt((double) cluster[cl]->n);
#else
        double          d_cl = distance(x[r], cluster[cl]->center_of_mass);
#endif
        if (d_cl < mindist) {
            mindist = d_cl;
            nearest_cluster = cl;
        }
    }
    return nearest_cluster;
}   //end do_pseudo_k_means

void roundtrip_KNN_cluster_partition(size_t n)
{
    int             kmax = K;

    size_t          cl_population;

    size_t          clcount_old = -1;

    size_t          last_K = 1;

    already_inserted = calloc(n, sizeof(*already_inserted));

    cluster = calloc(n, sizeof(*cluster));

    unclustered = calloc(n, sizeof(ID_t));

    clcount = 0;

    for (K = K0; K <= kmax && clcount != 1; K++) {

        n_unclustered = 0;
        for (int cl = clcount - 1; cl > -1; cl--) {
            if (cluster[cl]) {
                set__destroy(cluster[cl]);
                cluster[cl] = NULL;
            }
        }

        bzero(already_inserted, sizeof(index_t) * n);
        bzero(unclustered, sizeof(index_t) * n);
        n_unclustered = 0;

        do_backward_connection(n);

        for (index_t q = clcount = 0; q < n; q++) {
            if (not already_inserted[q]) {

#ifdef __IGNORE_HEADING_UNCLUSTERED__
                /*** ignore unclustered particles ***/
                if (KNN[q].hits < 1) {
                    continue;
                }
#endif
                cl_population = roundtrip_KNN_find(q);

#ifdef __REFUSE_SINGLETON_CLUSTER__
                /*** since cl_population == 1, the particle inside is unclustered ***/
                if (cl_population > 1) {
#elif defined(__REFUSE_LESS_THAN_K_POINT_CLUSTERS__)
                /*** refuse cluster populated with less than K + 1 points ***/
                if (cl_population > K) {
#else
                /*** if cluster is empty we are in trouble!!! ***/
                if (cl_population > 0) {
#endif
                    clcount++;
                } else {
                    /*** release unwanted cluster ***/
                    set__destroy(cluster[clcount]);
                    cluster[clcount] = NULL;
                }
            }   //end if (not already_inserted...
        }   // end for (index_t q = ...

#ifdef __RECLUSTER_BY_PSEUDO_KMEANS__
        /*** perform pseudo k-means if some particle was unclustered ***/
        if (n_unclustered) {
            for (int i = 0; i < n_unclustered; i++) {
                ID_t            pID = unclustered[i];
                index_t         p = pID - 1;

                already_inserted[p] = 0;
                index_t         nearest_cluster = do_pseudo_k_means(clcount, p);

                cluster_insert(nearest_cluster, pID);
            }
        }
#endif

        if (clcount == clcount_old) {
            verbose = 0;
            fprintf(stdout, "ignoring K = %zu since KNN topology is the same of K = %zu\n", K, last_K);
            continue;
        }

        last_K = K;
        clcount_old = clcount;
        verbose = 1;

        showresults();

        /*** uncomment the next line to show KNN graph using graphviz syntax by means of
             *.gv graph script ***/
#ifdef __GRAPHVIZ_OUTPUT_SYNTAX__
        fprintf(stdout, "compiling the KNN graph into a gv-graph file\n");
        show_KNN_graph(n);
#endif

        if (clcount == 1)
            break;

    }   // end for (K = ...

    if (verbose == 0) {
        fprintf(stdout, "nothing more after K = %zu\n", last_K);
    }
    fprintf(stdout, "done!\n");
}

/*****************
 *  Show Results *
 *****************/

char           *outputformat = "%s_K%dCs%d.layer";
char           *outputerrmsg = "%s: cannot open output file %s... exiting\n";

void showresults(void)
{
    char           *outputfname = calloc(strlen(outputprefix) + strlen(outputformat) + 10, 1);
    int             cluster_number = clcount;

    sprintf(outputfname, outputformat, outputprefix, K, clcount);

    output = fopen(outputfname, "w");

    if (output == NULL) {
        fprintf(stderr, outputerrmsg, cmdname, outputfname);
        exit(-666);
    }
    fprintf(stdout, "roundtrip_KNN_cluster_partition: performing the recursive roundtrip k-NN cluster discovery for K=%zu", K);
    fprintf(stdout, " => %zd clusters disclosed\n", clcount);

#ifdef __VERBOSE_CLUSTER_ANALYSIS_OUTCOME__
    fprintf(output, "K = %d\n", K);
    for (clcount = 0; clcount < cluster_number; clcount++) {
        if (cluster[clcount]->n > 0) {
            fprintf(output, "cluster %i with %d data instances ", clcount + 1, cluster[clcount]->n);
            fprintf(output, "with expectation vector\n");
            for (index_t j = 0; j < dimension; j++) {
                fprintf(output, "\t%lg", cluster[clcount]->center_of_mass[j]);
            }
            fprintf(output, "\n");
            cluster_list(cluster[clcount]);
            fprintf(output, "\n");
        }
    }
#else
    for (clcount = 0; clcount < cluster_number; clcount++) {
        if (cluster[clcount]->n > 0) {
            cluster_list(cluster[clcount]);
            fprintf(output, "\n");
        }
    }
#endif

    free(outputfname);
    fclose(output);
}

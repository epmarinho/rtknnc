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
 *      Once introduced the procedure void KNN_backward_trip(index_t q_index);
 *      it is nolonger necessary to perform an exhaustive recursion trhough all
 *      the nearest neighbors of q_index. The old code segment was removed and moved
 *      in to the file k-NN-topology-removed-piece.c in the src directory
 *
 * =====================================================================================*/

/** Global definitions. Uncomment your choices. All commented returns to the original method: **/

/** Force to show the KNN graph using the GraphViz syntax: **/
// #define __GRAPHVIZ_OUTPUT_SYNTAX__

/** Emulate visual range **/
#define __EMULATE_VISIBILITY__

#ifdef __EMULATE_VISIBILITY__

/** NOTE: Hereafter do not comment/uncomment the directive flags!!! **/

/** refuse clusters with less than K points within: **/
// #    define __REFUSE_NEGLIGIBLE_CLUSTERS__
#    define __REFUSE_SINGLETON_CLUSTER__
#    define __RMS_VISIBILITY__

// #    define __USE_VISIBILITY_TOL_METHOD__

#    ifdef __USE_VISIBILITY_TOL_METHOD__
#        define VISIBTOL1 1./16.
#        define VISIBTOL2 1./32.
#    endif

#    ifdef __REFUSE_NEGLIGIBLE_CLUSTERS__
/** Apply pseudo k-means to regroup unclustered (low connectivity refused) particles: **/
#        define __RECLUSTER_BY_PSEUDO_KMEANS__
#        ifdef __RECLUSTER_BY_PSEUDO_KMEANS__

/** choose only one, or none, of the weighted distance method: **/
// #            define __WEIGHTED_INV_SQRT_N_DISTANCE__
// #            define __WEIGHTED_INV_N_DISTANCE__
//           avoid conflicting definitions of the weighted distance methods:
#            if defined(__WEIGHTED_INV_N_DISTANCE__) && defined(__WEIGHTED_INV_SQRT_N_DISTANCE__)
#                undef __WEIGHTED_INV_N_DISTANCE__
#                undef __WEIGHTED_INV_SQRT_N_DISTANCE__
#            endif

#        endif
#    else
#        undef __RECLUSTER_BY_PSEUDO_KMEANS__
#        undef __WEIGHTED_INV_SQRT_N_DISTANCE__
#        undef __WEIGHTED_INV_N_DISTANCE__
#    endif
/** turn on neighborhood popularity measuring **/
#    define __COMPUTE_NEIGHBORHOOD_POPULARITY__
#endif
/**************************************************************************************
 * Last changes has began on 03 Mar 2020:
 * Radical change: the code is now simulating the previous idea of the full round-trip
 * KNN until exhaustion -- see function round_trip_KNN_explore below.
 *************************************************************************************/

#include "k-nn-topology.h"

#include <float.h>

/*
 * Optional, native structural pruning.
 *
 * The historical forward/backward closure remains the default so that the
 * validated C/Python regression is still available unchanged.  Passing
 * --structural-pruning asks the C core to inspect each sufficiently large
 * closure before it is written.  A split is accepted only when a diagonal
 * Gaussian-mixture search strongly prefers exactly two or three populations;
 * four components act as a veto against cutting curved or homogeneous data.
 * Every proposed child must also be coherent in the visible KNN graph.  The
 * first EM initialisation starts at the most popular
 * point in the component, retaining the leadership idea as the deterministic
 * seed rather than treating every weak connection as equally authoritative.
 */
#define PRUNING_MAX_MODELS 4
#define PRUNING_RESTARTS 5
#define PRUNING_KMEANS_ITERATIONS 24
#define PRUNING_EM_ITERATIONS 200
#define PRUNING_MIN_COMPONENT_SIZE 50
#define PRUNING_MIN_COMPONENT_FRACTION 0.15
#define PRUNING_MIN_CHILD_SIZE 20
#define PRUNING_MIN_CHILD_FRACTION 0.05
#define PRUNING_MIN_BIC_GAIN 10.0
#define PRUNING_REG_COVAR 1.e-6
#define PRUNING_LOG_2PI 1.83787706640934548356

flag_t          structural_pruning = 0;
vote_kernel_t   vote_kernel = VOTE_KERNEL_INV_R2;
double          vote_sigmoid_alpha = 8.;

/** clcount reports the number of identified clusters for the
     currently selected number of nearest neighbors, K **/
size_t          clcount = 0;

/** already inserted array is a way of locking a particle to prevent
     it to be assigned to more than one cluster **/
index_t        *already_inserted;

/** the cluster itself is a set abstract data type: **/
SET           **cluster;

/** lists the set content in order: **/
void cluster_list(SET * set)
{
    int             j;

    if (set->left != NULL) {
        cluster_list(set->left);
    }

    fprintf(output, " %i %i\t", already_inserted[set->id - 1], set->id);
    //     fprintf(output, " %zu %i\t", clcount + 1, set->id);

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

        /** on the first run, create the root-datastructure which simulates
             a cluster as a ID set of the 1-pass roundtrip KNN members: **/
        cluster[clcount] = calloc(1, sizeof(SET));
        cluster[clcount]->n = 1;
        cluster[clcount]->id = queryID;
        /** @cluster[clcount]->size @cluster_insert:: still in tests */
        cluster[clcount]->size = KNN[q].visibility;
         /**/
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

/**
 * Return the shape assigned to one outgoing edge before source-budget
 * normalisation.  Every experimental kernel redistributes the historical
 * inverse-square budget
 *
 *     B_i = sum_j 1 / r_ij^2
 *
 * across source i's K outgoing edges.  Consequently all kernels have the
 * same total vote mass and the same inverse-square physical scale; only the
 * relative allocation amongst neighbours changes.  For inv_r2 the
 * normalisation cancels algebraically and the historical 1/r^2 vote is kept
 * bit-for-bit by a special case in do_backward_connection().
 */
static double reverse_neighbour_vote_shape(index_t source, index_t rank)
{
    double          distance_d2 = KNN[source].d[rank];

    switch (vote_kernel) {
    case VOTE_KERNEL_UNIFORM:
        return 1.;

    case VOTE_KERNEL_INV_R:
        if (distance_d2 <= 0.) {
            fprintf(stderr,
                    "%s: inv_r requires positive neighbour distances\n",
                    cmdname);
            exit(EXIT_FAILURE);
        }
        return 1. / sqrt(distance_d2);

    case VOTE_KERNEL_INV_R2:
        if (distance_d2 <= 0.) {
            fprintf(stderr,
                    "%s: inv_r2 requires positive neighbour distances\n",
                    cmdname);
            exit(EXIT_FAILURE);
        }
        return 1. / distance_d2;

    case VOTE_KERNEL_SIGMOID: {
        double          radius_d2 = KNN[source].radius;

        if (radius_d2 <= 0.) {
            fprintf(stderr,
                    "%s: sigmoid requires a positive local KNN radius\n",
                    cmdname);
            exit(EXIT_FAILURE);
        }
        double          z = vote_sigmoid_alpha
            * (sqrt(distance_d2 / radius_d2) - 1.);

        if (z >= 700.)
            return 0.;
        if (z <= -700.)
            return 1.;
        return 1. / (1. + exp(z));
    }
    }

    fprintf(stderr, "%s: invalid internal vote-kernel selector\n", cmdname);
    exit(EXIT_FAILURE);
}

void do_backward_connection(size_t n)
{
    /** It is better to clean up any possible garbage before computing the k-NN's symmetric closure **/
    for (index_t q = 0; q < n; q++) {
        KNN[q].hits = KNN[q].popularity = 0;
    }

    for (index_t q = 0; q < n; q++) {
#ifdef __COMPUTE_NEIGHBORHOOD_POPULARITY__
        double          historical_budget = 0.;
        double          shape_sum = 0.;

        for (index_t j = 0; j < K; j++) {
            double          distance_d2 = KNN[q].d[j];

            if (distance_d2 <= 0.) {
                fprintf(stderr,
                        "%s: vote kernels require positive neighbour distances\n",
                        cmdname);
                exit(EXIT_FAILURE);
            }
            historical_budget += 1. / distance_d2;
            if (vote_kernel != VOTE_KERNEL_INV_R2)
                shape_sum += reverse_neighbour_vote_shape(q, j);
        }
        if (vote_kernel != VOTE_KERNEL_INV_R2 && shape_sum <= 0.) {
            fprintf(stderr, "%s: vote-kernel shape has zero source mass\n",
                    cmdname);
            exit(EXIT_FAILURE);
        }
#endif

        for (index_t j = 0; j < K; j++) {
            index_t         r = KNN[q].list[j] - 1;

            /** Preliminarily count the number of incoming connections (hits) to allow the memory
                allocation forecast for the backward connections: **/
            KNN[r].hits++;

#ifdef __COMPUTE_NEIGHBORHOOD_POPULARITY__
            /** Accumulate a source-budget-matched incoming vote.  The
                historical inverse-square rule remains exactly unchanged. **/
            if (vote_kernel == VOTE_KERNEL_INV_R2)
                KNN[r].popularity += 1. / KNN[q].d[j];
            else
                KNN[r].popularity += historical_budget
                    * reverse_neighbour_vote_shape(q, j) / shape_sum;
#endif

        }
    }

    /** Allocate the memory space of the backward connection: **/
    for (index_t q = 0; q < n; q++) {
        KNN[q].backward = realloc(KNN[q].backward, KNN[q].hits * sizeof(*KNN[q].backward));
        /** Clear to reuse the hit counter as the backward array cursor: **/
        KNN[q].hits = 0;
    }

    for (index_t q = 0; q < n; q++) {
        ID_t            qID = q + 1;

        for (index_t j = 0; j < K; j++) {
            ID_t            pID = KNN[q].list[j];
            index_t         p = pID - 1;

            /** qID is fan of pID: **/
            KNN[p].backward[KNN[p].hits++] = qID;
        }
    }

#ifdef __EMULATE_VISIBILITY__
    /**
     The first idea that came to mind to simulate a nebulous way for particles
     to see the closest amongst their k nearest neighbors was through averaging
     the k-nearest neighbor distance. Subsequently, I introduced a factor that
     increases the chance of eventually seeing all of the k nearest neighbors
     using the concept of popularity, defined above.
    **/
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

    double          mean = 0;

    for (index_t q = 0; q < n; q++) {
        for (int j = KNN[q].k_visible = 0; j < K; j++) {
            if (KNN[q].d[j] <= KNN[q].visibility) {
                KNN[q].k_visible++;
            }
        }
        mean += KNN[q].k_visible;
    }
    mean /= n;
    fprintf(stdout, "emulate visibility: mean number of viewed neighbors is %g out from %zu\n", mean, K - 1);
#endif

}   // end do_backward_connection(size_t n)

typedef struct {
    double          log_likelihood;
    double          bic;
    index_t        *labels;
} diagonal_gmm_fit;

typedef struct {
    flag_t          accepted;
    index_t        *side;
    int             child_count;
    double          bic[PRUNING_MAX_MODELS];
    double          bic_gain;
    size_t          child_size[PRUNING_MAX_MODELS];
    size_t          removed_edges;
} pruning_decision;

static void *pruning_calloc(size_t count, size_t size)
{
    void           *memory = calloc(count, size);

    if (memory == NULL) {
        fprintf(stderr, "%s: structural pruning could not allocate %zu bytes\n",
                cmdname, count * size);
        exit(EXIT_FAILURE);
    }
    return memory;
}

static double squared_row_distance(const double *a, const double *b, dim_t dim)
{
    double          value = 0;

    for (dim_t j = 0; j < dim; j++) {
        double          delta = a[j] - b[j];

        value += delta * delta;
    }
    return value;
}

static void collect_set_ids(const SET *set, index_t *ids, size_t *cursor)
{
    if (set->left != NULL)
        collect_set_ids(set->left, ids, cursor);

    ids[(*cursor)++] = set->id - 1;

    if (set->right != NULL)
        collect_set_ids(set->right, ids, cursor);
}

static double *standardize_component(const index_t *ids, size_t count)
{
    double         *mean = pruning_calloc(dimension, sizeof(*mean));
    double         *scale = pruning_calloc(dimension, sizeof(*scale));
    double         *standardized = pruning_calloc(count * dimension, sizeof(*standardized));

    for (size_t i = 0; i < count; i++) {
        for (dim_t j = 0; j < dimension; j++)
            mean[j] += x[ids[i]][j];
    }
    for (dim_t j = 0; j < dimension; j++)
        mean[j] /= count;

    for (size_t i = 0; i < count; i++) {
        for (dim_t j = 0; j < dimension; j++) {
            double          centered = x[ids[i]][j] - mean[j];

            scale[j] += centered * centered;
        }
    }
    for (dim_t j = 0; j < dimension; j++) {
        scale[j] = sqrt(scale[j] / count);
        if (scale[j] <= DBL_EPSILON)
            scale[j] = 1;
    }

    for (size_t i = 0; i < count; i++) {
        for (dim_t j = 0; j < dimension; j++)
            standardized[i * dimension + j] = (x[ids[i]][j] - mean[j]) / scale[j];
    }

    free(scale);
    free(mean);
    return standardized;
}

static size_t most_popular_component_point(const index_t *ids, size_t count)
{
    size_t          best = 0;

    for (size_t i = 1; i < count; i++) {
        double          candidate = KNN[ids[i]].popularity;
        double          incumbent = KNN[ids[best]].popularity;

        if (candidate > incumbent || (candidate == incumbent && ids[i] < ids[best]))
            best = i;
    }
    return best;
}

static size_t deterministic_seed(size_t count, int model, int restart)
{
    unsigned long   value = (unsigned long) (restart + 1) * 104729UL;

    value += (unsigned long) model * 15485863UL;
    return value % count;
}

static void initialize_diagonal_gmm(const double *data,
                                    const index_t *ids,
                                    size_t count,
                                    int model,
                                    int restart,
                                    double *weight,
                                    double *mean,
                                    double *variance)
{
    index_t        *assignment = pruning_calloc(count, sizeof(*assignment));
    size_t         *population = pruning_calloc(model, sizeof(*population));
    flag_t         *chosen = pruning_calloc(count, sizeof(*chosen));
    size_t          first = restart == 0
        ? most_popular_component_point(ids, count)
        : deterministic_seed(count, model, restart);

    memcpy(mean, data + first * dimension, dimension * sizeof(*mean));
    chosen[first] = 1;

    for (int component = 1; component < model; component++) {
        size_t          farthest = 0;
        double          farthest_distance = -1;

        for (size_t i = 0; i < count; i++) {
            double          nearest = INFINITY;

            if (chosen[i])
                continue;
            for (int other = 0; other < component; other++) {
                double          candidate = squared_row_distance(
                    data + i * dimension,
                    mean + other * dimension,
                    dimension
                );

                if (candidate < nearest)
                    nearest = candidate;
            }
            if (nearest > farthest_distance) {
                farthest_distance = nearest;
                farthest = i;
            }
        }
        memcpy(mean + component * dimension,
               data + farthest * dimension,
               dimension * sizeof(*mean));
        chosen[farthest] = 1;
    }

    for (int iteration = 0; iteration < PRUNING_KMEANS_ITERATIONS; iteration++) {
        flag_t          changed = 0;

        bzero(population, model * sizeof(*population));
        for (size_t i = 0; i < count; i++) {
            int             best = 0;
            double          best_distance = squared_row_distance(
                data + i * dimension, mean, dimension
            );

            for (int component = 1; component < model; component++) {
                double          candidate = squared_row_distance(
                    data + i * dimension,
                    mean + component * dimension,
                    dimension
                );

                if (candidate < best_distance) {
                    best = component;
                    best_distance = candidate;
                }
            }
            if (iteration > 0 && assignment[i] != best)
                changed = 1;
            assignment[i] = best;
            population[best]++;
        }

        double         *updated = pruning_calloc(model * dimension, sizeof(*updated));

        for (size_t i = 0; i < count; i++) {
            for (dim_t j = 0; j < dimension; j++)
                updated[assignment[i] * dimension + j] += data[i * dimension + j];
        }
        for (int component = 0; component < model; component++) {
            if (population[component] == 0) {
                size_t          replacement = deterministic_seed(
                    count, model + component, restart + iteration + 1
                );

                memcpy(updated + component * dimension,
                       data + replacement * dimension,
                       dimension * sizeof(*updated));
                population[component] = 1;
                changed = 1;
            } else {
                for (dim_t j = 0; j < dimension; j++)
                    updated[component * dimension + j] /= population[component];
            }
        }
        memcpy(mean, updated, model * dimension * sizeof(*mean));
        free(updated);

        if (iteration > 0 && !changed)
            break;
    }

    bzero(population, model * sizeof(*population));
    for (size_t i = 0; i < count; i++)
        population[assignment[i]]++;

    for (int component = 0; component < model; component++) {
        weight[component] = (double) population[component] / count;
        for (dim_t j = 0; j < dimension; j++)
            variance[component * dimension + j] = PRUNING_REG_COVAR;
    }
    for (size_t i = 0; i < count; i++) {
        int             component = assignment[i];

        for (dim_t j = 0; j < dimension; j++) {
            double          delta = data[i * dimension + j] - mean[component * dimension + j];

            variance[component * dimension + j] += delta * delta;
        }
    }
    for (int component = 0; component < model; component++) {
        double          denominator = population[component] > 0 ? population[component] : 1;

        for (dim_t j = 0; j < dimension; j++)
            variance[component * dimension + j] /= denominator;
    }

    free(chosen);
    free(population);
    free(assignment);
}

static double gmm_expectation(const double *data,
                              size_t count,
                              int model,
                              const double *weight,
                              const double *mean,
                              const double *variance,
                              double *responsibility)
{
    double          log_likelihood = 0;
    double         *log_probability = pruning_calloc(model, sizeof(*log_probability));

    for (size_t i = 0; i < count; i++) {
        double          maximum = -INFINITY;

        for (int component = 0; component < model; component++) {
            double          quadratic = 0;
            double          log_determinant = 0;

            for (dim_t j = 0; j < dimension; j++) {
                double          var = max(variance[component * dimension + j], PRUNING_REG_COVAR);
                double          delta = data[i * dimension + j] - mean[component * dimension + j];

                quadratic += delta * delta / var;
                log_determinant += log(var);
            }
            log_probability[component] = log(max(weight[component], DBL_MIN))
                - .5 * (dimension * PRUNING_LOG_2PI + log_determinant + quadratic);
            maximum = max(maximum, log_probability[component]);
        }

        double          normalizer = 0;

        for (int component = 0; component < model; component++)
            normalizer += exp(log_probability[component] - maximum);
        normalizer = maximum + log(normalizer);
        log_likelihood += normalizer;

        for (int component = 0; component < model; component++)
            responsibility[i * model + component] = exp(log_probability[component] - normalizer);
    }

    free(log_probability);
    return log_likelihood;
}

static void gmm_maximization(const double *data,
                             size_t count,
                             int model,
                             const double *responsibility,
                             double *weight,
                             double *mean,
                             double *variance)
{
    double         *effective = pruning_calloc(model, sizeof(*effective));

    bzero(mean, model * dimension * sizeof(*mean));
    bzero(variance, model * dimension * sizeof(*variance));

    for (size_t i = 0; i < count; i++) {
        for (int component = 0; component < model; component++) {
            double          value = responsibility[i * model + component];

            effective[component] += value;
            for (dim_t j = 0; j < dimension; j++)
                mean[component * dimension + j] += value * data[i * dimension + j];
        }
    }

    for (int component = 0; component < model; component++) {
        double          denominator = max(effective[component], DBL_MIN);

        weight[component] = denominator / count;
        for (dim_t j = 0; j < dimension; j++)
            mean[component * dimension + j] /= denominator;
    }

    for (size_t i = 0; i < count; i++) {
        for (int component = 0; component < model; component++) {
            double          value = responsibility[i * model + component];

            for (dim_t j = 0; j < dimension; j++) {
                double          delta = data[i * dimension + j] - mean[component * dimension + j];

                variance[component * dimension + j] += value * delta * delta;
            }
        }
    }
    for (int component = 0; component < model; component++) {
        double          denominator = max(effective[component], DBL_MIN);

        for (dim_t j = 0; j < dimension; j++)
            variance[component * dimension + j] =
                variance[component * dimension + j] / denominator + PRUNING_REG_COVAR;
    }

    free(effective);
}

static diagonal_gmm_fit fit_diagonal_gmm(const double *data,
                                         const index_t *ids,
                                         size_t count,
                                         int model)
{
    diagonal_gmm_fit fit = {
        .log_likelihood = -INFINITY,
        .bic = INFINITY,
        .labels = pruning_calloc(count, sizeof(*fit.labels))
    };
    int             restarts = model == 1 ? 1 : PRUNING_RESTARTS;

    for (int restart = 0; restart < restarts; restart++) {
        double         *weight = pruning_calloc(model, sizeof(*weight));
        double         *mean = pruning_calloc(model * dimension, sizeof(*mean));
        double         *variance = pruning_calloc(model * dimension, sizeof(*variance));
        double         *responsibility = pruning_calloc(count * model, sizeof(*responsibility));
        double          previous = -INFINITY;
        double          likelihood = -INFINITY;

        initialize_diagonal_gmm(data, ids, count, model, restart,
                                weight, mean, variance);

        for (int iteration = 0; iteration < PRUNING_EM_ITERATIONS; iteration++) {
            likelihood = gmm_expectation(data, count, model,
                                         weight, mean, variance, responsibility);
            if (iteration > 0
                && fabs(likelihood - previous) <= 1.e-6 * (1 + fabs(likelihood)))
                break;
            previous = likelihood;
            gmm_maximization(data, count, model, responsibility,
                             weight, mean, variance);
        }
        likelihood = gmm_expectation(data, count, model,
                                     weight, mean, variance, responsibility);

        if (likelihood > fit.log_likelihood) {
            fit.log_likelihood = likelihood;
            for (size_t i = 0; i < count; i++) {
                int             best = 0;

                for (int component = 1; component < model; component++) {
                    if (responsibility[i * model + component]
                        > responsibility[i * model + best])
                        best = component;
                }
                fit.labels[i] = best;
            }
        }

        free(responsibility);
        free(variance);
        free(mean);
        free(weight);
    }

    double          parameters = model * (2. * dimension + 1.) - 1.;

    fit.bic = -2. * fit.log_likelihood + parameters * log((double) count);
    return fit;
}

static flag_t directed_visible_edge(index_t source, index_t target)
{
    ID_t            targetID = target + 1;

    for (size_t j = 0; j < K; j++) {
        if (KNN[source].list[j] == targetID)
            return KNN[source].d[j] <= KNN[source].visibility;
    }
    return 0;
}

static void enqueue_same_side_neighbours(index_t global,
                                         const index_t *local_index,
                                         const index_t *side,
                                         index_t wanted_side,
                                         index_t component_number,
                                         index_t *component,
                                         index_t *queue,
                                         size_t *queue_end)
{
    for (size_t j = 0; j < K; j++) {
        index_t         neighbour = KNN[global].list[j] - 1;
        index_t         local = local_index[neighbour];

        if (local >= 0 && side[local] == wanted_side && component[local] < 0
            && KNN[global].d[j] <= KNN[global].visibility) {
            component[local] = component_number;
            queue[(*queue_end)++] = neighbour;
        }
    }

    for (size_t j = 0; j < KNN[global].hits; j++) {
        index_t         neighbour = KNN[global].backward[j] - 1;
        index_t         local = local_index[neighbour];

        if (local >= 0 && side[local] == wanted_side && component[local] < 0
            && directed_visible_edge(neighbour, global)) {
            component[local] = component_number;
            queue[(*queue_end)++] = neighbour;
        }
    }
}

static index_t fragment_replacement_side(const index_t *ids,
                                         size_t count,
                                         const index_t *local_index,
                                         const index_t *component,
                                         index_t fragment,
                                         const index_t *side,
                                         index_t old_side,
                                         int side_count)
{
    size_t         *support = pruning_calloc(side_count, sizeof(*support));
    size_t         *population = pruning_calloc(side_count, sizeof(*population));

    for (size_t i = 0; i < count; i++)
        population[side[i]]++;

    for (size_t i = 0; i < count; i++) {
        if (component[i] != fragment)
            continue;

        index_t         global = ids[i];

        for (size_t j = 0; j < K; j++) {
            index_t         neighbour = KNN[global].list[j] - 1;
            index_t         local = local_index[neighbour];

            if (local >= 0 && side[local] != old_side
                && KNN[global].d[j] <= KNN[global].visibility)
                support[side[local]]++;
        }
        for (size_t j = 0; j < KNN[global].hits; j++) {
            index_t         neighbour = KNN[global].backward[j] - 1;
            index_t         local = local_index[neighbour];

            if (local >= 0 && side[local] != old_side
                && directed_visible_edge(neighbour, global))
                support[side[local]]++;
        }
    }

    index_t         replacement = -1;

    for (index_t candidate = 0; candidate < side_count; candidate++) {
        if (candidate == old_side)
            continue;
        if (replacement < 0 || support[candidate] > support[replacement]
            || (support[candidate] == support[replacement]
                && population[candidate] > population[replacement]))
            replacement = candidate;
    }

    free(population);
    free(support);
    return replacement;
}

static flag_t regularize_partition(const index_t *ids,
                                    size_t count,
                                    index_t *side,
                                    int side_count,
                                    size_t required_child,
                                    size_t point_count)
{
    index_t        *local_index = pruning_calloc(point_count, sizeof(*local_index));
    index_t        *component = pruning_calloc(count, sizeof(*component));
    index_t        *queue = pruning_calloc(count, sizeof(*queue));
    size_t         *component_size = pruning_calloc(count, sizeof(*component_size));
    flag_t          coherent = 0;

    for (size_t i = 0; i < point_count; i++)
        local_index[i] = -1;
    for (size_t i = 0; i < count; i++)
        local_index[ids[i]] = i;

    for (int pass = 0; pass < 4; pass++) {
        flag_t          changed = 0;

        for (index_t wanted_side = 0; wanted_side < side_count; wanted_side++) {
            for (size_t i = 0; i < count; i++)
                component[i] = -1;
            bzero(component_size, count * sizeof(*component_size));

            index_t         component_count = 0;

            for (size_t seed = 0; seed < count; seed++) {
                if (side[seed] != wanted_side || component[seed] >= 0)
                    continue;

                size_t          queue_begin = 0;
                size_t          queue_end = 0;

                component[seed] = component_count;
                queue[queue_end++] = ids[seed];

                while (queue_begin < queue_end) {
                    index_t         global = queue[queue_begin++];
                    index_t         local = local_index[global];

                    component_size[component_count]++;
                    enqueue_same_side_neighbours(
                        global, local_index, side, wanted_side,
                        component_count, component, queue, &queue_end
                    );
                    (void) local;
                }
                component_count++;
            }

            if (component_count == 0)
                goto cleanup;

            index_t         principal = 0;

            for (index_t part = 1; part < component_count; part++) {
                if (component_size[part] > component_size[principal])
                    principal = part;
            }
            for (index_t part = 0; part < component_count; part++) {
                if (part == principal)
                    continue;
                if (component_size[part] >= required_child)
                    goto cleanup;
                index_t         replacement = fragment_replacement_side(
                    ids, count, local_index, component, part,
                    side, wanted_side, side_count
                );

                if (replacement < 0)
                    goto cleanup;
                for (size_t i = 0; i < count; i++) {
                    if (component[i] == part) {
                        side[i] = replacement;
                        changed = 1;
                    }
                }
            }
        }

        if (!changed) {
            coherent = 1;
            break;
        }
    }

cleanup:
    free(component_size);
    free(queue);
    free(component);
    free(local_index);
    return coherent;
}

static size_t count_crossing_visible_edges(const index_t *ids,
                                            size_t count,
                                            const index_t *side,
                                            size_t point_count)
{
    index_t        *local_index = pruning_calloc(point_count, sizeof(*local_index));
    size_t          removed = 0;

    for (size_t i = 0; i < point_count; i++)
        local_index[i] = -1;
    for (size_t i = 0; i < count; i++)
        local_index[ids[i]] = i;

    for (size_t i = 0; i < count; i++) {
        index_t         source = ids[i];

        for (size_t j = 0; j < K; j++) {
            index_t         target = KNN[source].list[j] - 1;
            index_t         target_local = local_index[target];

            if (target_local < 0 || side[i] == side[target_local]
                || KNN[source].d[j] > KNN[source].visibility)
                continue;
            if (source < target || !directed_visible_edge(target, source))
                removed++;
        }
    }

    free(local_index);
    return removed;
}

static pruning_decision propose_component_pruning(const index_t *ids,
                                                   size_t count,
                                                   size_t point_count)
{
    pruning_decision decision = { 0 };
    double         *standardized = standardize_component(ids, count);
    diagonal_gmm_fit fits[PRUNING_MAX_MODELS];
    int             best_model = 0;

    for (int model = 1; model <= PRUNING_MAX_MODELS; model++) {
        fits[model - 1] = fit_diagonal_gmm(standardized, ids, count, model);
        decision.bic[model - 1] = fits[model - 1].bic;
        if (fits[model - 1].bic < fits[best_model].bic)
            best_model = model - 1;
    }
    decision.child_count = best_model + 1;
    decision.bic_gain = decision.bic[0] - decision.bic[best_model];

    if ((decision.child_count == 2 || decision.child_count == 3)
        && decision.bic_gain >= PRUNING_MIN_BIC_GAIN) {
        size_t          required_child = max(
            PRUNING_MIN_CHILD_SIZE,
            (size_t) ceil(PRUNING_MIN_CHILD_FRACTION * count)
        );

        decision.side = pruning_calloc(count, sizeof(*decision.side));
        memcpy(decision.side, fits[best_model].labels, count * sizeof(*decision.side));

        for (size_t i = 0; i < count; i++) {
            if (decision.side[i] >= 0 && decision.side[i] < decision.child_count)
                decision.child_size[decision.side[i]]++;
        }

        flag_t          children_large_enough = 1;

        for (int child = 0; child < decision.child_count; child++)
            children_large_enough &= decision.child_size[child] >= required_child;

        if (children_large_enough
            && regularize_partition(ids, count, decision.side,
                                    decision.child_count, required_child,
                                    point_count)) {
            bzero(decision.child_size, sizeof(decision.child_size));
            for (size_t i = 0; i < count; i++)
                decision.child_size[decision.side[i]]++;

            children_large_enough = 1;
            for (int child = 0; child < decision.child_count; child++)
                children_large_enough &= decision.child_size[child] >= required_child;

            if (children_large_enough) {
                decision.removed_edges = count_crossing_visible_edges(
                    ids, count, decision.side, point_count
                );
                decision.accepted = decision.removed_edges > 0;
            }
        }
    }

    if (!decision.accepted) {
        free(decision.side);
        decision.side = NULL;
    }
    for (int model = 0; model < PRUNING_MAX_MODELS; model++)
        free(fits[model].labels);
    free(standardized);
    return decision;
}

static size_t structurally_prune_clusters(size_t point_count)
{
    size_t          original_count = clcount;
    size_t          minimum_component = max(
        PRUNING_MIN_COMPONENT_SIZE,
        (size_t) ceil(PRUNING_MIN_COMPONENT_FRACTION * point_count)
    );
    index_t        *final_label = pruning_calloc(point_count, sizeof(*final_label));
    size_t          final_count = 0;
    size_t          accepted_splits = 0;

    for (size_t i = 0; i < point_count; i++)
        final_label[i] = -1;

    for (size_t cl = 0; cl < original_count; cl++) {
        size_t          count = cluster[cl]->n;
        index_t        *ids = pruning_calloc(count, sizeof(*ids));
        size_t          cursor = 0;
        pruning_decision decision = { 0 };

        collect_set_ids(cluster[cl], ids, &cursor);
        if (count >= minimum_component)
            decision = propose_component_pruning(ids, count, point_count);

        if (decision.accepted) {
            for (size_t i = 0; i < count; i++)
                final_label[ids[i]] = final_count + decision.side[i];

            fprintf(stdout,
                    "structural pruning: cluster %zu (%zu points) -> %d children [",
                    cl + 1, count, decision.child_count);
            for (int child = 0; child < decision.child_count; child++)
                fprintf(stdout, "%s%zu", child == 0 ? "" : ", ",
                        decision.child_size[child]);
            fprintf(stdout,
                    "]; BIC=[%.3f, %.3f, %.3f, %.3f], gain=%.3f, removed edges=%zu\n",
                    decision.bic[0], decision.bic[1],
                    decision.bic[2], decision.bic[3],
                    decision.bic_gain, decision.removed_edges);
            final_count += decision.child_count;
            accepted_splits++;
        } else {
            for (size_t i = 0; i < count; i++)
                final_label[ids[i]] = final_count;
            final_count++;
        }

        free(decision.side);
        free(ids);
    }

    for (size_t cl = 0; cl < original_count; cl++) {
        set__destroy(cluster[cl]);
        cluster[cl] = NULL;
    }
    bzero(already_inserted, point_count * sizeof(*already_inserted));
    bzero(unclustered, point_count * sizeof(*unclustered));
    n_unclustered = 0;

    for (size_t point = 0; point < point_count; point++) {
        if (final_label[point] >= 0)
            cluster_insert(final_label[point], point + 1);
    }
    clcount = final_count;

    free(final_label);
    return accepted_splits;
}

static void canonical_partition(index_t *canonical, size_t point_count)
{
    if (clcount == 0) {
        bzero(canonical, point_count * sizeof(*canonical));
        return;
    }

    size_t         *minimum_member = pruning_calloc(clcount, sizeof(*minimum_member));

    for (size_t cl = 0; cl < clcount; cl++)
        minimum_member[cl] = point_count + 1;

    for (size_t point = 0; point < point_count; point++) {
        index_t         label = already_inserted[point];

        if (label > 0 && point + 1 < minimum_member[label - 1])
            minimum_member[label - 1] = point + 1;
    }
    for (size_t point = 0; point < point_count; point++) {
        index_t         label = already_inserted[point];

        canonical[point] = label > 0 ? (index_t) minimum_member[label - 1] : 0;
    }

    free(minimum_member);
}

/** @KNN_backward_trip:: Exhaustively visits the backward neighbors of p if p != q */
void KNN_backward_trip(index_t q)
{
    /** if KNN[p].hits == 0, there is nothing to backward to **/
    if (KNN[q].hits == 0)
        return;

    /** k-NN back-trip starts here: **/
    for (int i = 0; i < KNN[q].hits; i++) {
        ID_t            pID = KNN[q].backward[i];
        index_t         p = pID - 1;

        /** If p is already inserted, it does not make sense to take this into
            account, since p has been already clustered: **/
        if (already_inserted[p]) {
            continue;
        }
#ifdef __EMULATE_VISIBILITY__
        ID_t            qID = q + 1;
        double          dpq = 0;

        for (int j = 0; j < K; j++) {
            if (KNN[p].list[j] == qID) {
                dpq = KNN[p].d[j];
                break;
            }
        }

        /** NOTE: The present tolerance method was a desperate maneuver to avoid
            underestimating the number of visible neighbors. Default: off **/
#    ifdef __USE_VISIBILITY_TOL_METHOD__
        /** backward trip only if [q] is "seen" by [p] **/
        if (dpq <= VISIBTOL1 * KNN[p].radius + KNN[p].visibility) {
#    else
        /** You see me:
                If q is within the visibility range of p, do the round trip k-NN to p: **/
        if (dpq <= KNN[p].visibility) {
#    endif
            /** This is the default call: **/
            round_trip_KNN_explore(p);
        }
#else
        /** This is the greedy method, which is the default: **/
        round_trip_KNN_explore(p);
#endif
    }

}   // end KNN_backward_trip(index_t q)

/** @KNN_forward_trip::
 This is in principle a non-deterministic
 method of simulating the entering in the
 K states */
void KNN_forward_trip(index_t q)
{
    for (int j = 0; j < K; j++) {
        index_t         p = KNN[q].list[j] - 1;

        if (already_inserted[p])
            continue;
#ifdef __EMULATE_VISIBILITY__
        /** This is a qualitative and quite questionable approach of emulating
            the idea that some of k nearest neighbors is visible. In other words,
            this is the relevance that the particle q gives to at least one
            of its k nearest neighbors. Thus, forward trip only if q "sees" p. **/
        double          dpq = KNN[q].d[j];

        /** NOTE: The present tolerance method was a desperate maneuver to avoid
            underestimating the number of visible neighbors: **/
#    ifdef __USE_VISIBILITY_TOL_METHOD__
        if (dpq <= VISIBTOL2 * KNN[q].radius + KNN[q].visibility) {
#    else
        /** I see you:
                If p is within the visibility range of q, do the round trip k-NN to p: **/
        if (dpq <= KNN[q].visibility) {
#    endif
            round_trip_KNN_explore(p);
        }
#else
        /** This is the greedy method, which is the default: **/
        round_trip_KNN_explore(p);
#endif
    }

}   // end KNN_forward_trip

size_t round_trip_KNN_explore(index_t q)
{
    /** This is the query ID: **/
    int             queryID = q + 1;

    /** Rule out particles that has already been clustered: **/
    if (already_inserted[q]) {
        ;
    } else {

        /** Insert the present particle into the cluster in progress: **/
        cluster_insert(clcount, queryID);

        /** Visit the k nearest neighbors of point q and apply the present method for each one of them: **/
        KNN_forward_trip(q);

        /** Visit all particles that has pointed q as one of their k nearest neighbors and then apply this
            method to each of them: **/
        KNN_backward_trip(q);
    }

    /** return the population number of the just detected cluster[clcount]: **/
    return cluster[clcount]->n;

}   // end round_trip_KNN_explore(index_t q)

/********************************************
 * Multi layer, hierarchical cluster analysis *
 ********************************************/

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

ID_t do_pseudo_k_means(size_t k, index_t r)
{
    index_t         nearest_cluster = -1;
    double          mindist = INFINITY;

    for (index_t cl = 0; cl < k; cl++) {
        /** The distance between particles and clusters can be modified so that a richer cluster is considered
            closer than a poorer one, but with the same Euclidean distance. Choose one method:
            __WEIGHTED_INV_N_DISTANCE__ corresponds to dividing the distance by the cluster population;
            _ WEIGHTED_INV_SQRT_N_DISTANCE _ corresponds to dividing the distance by the square root of the
            cluster population - it was suggested to mitigate the decreasing effect with the cluster population. **/
#ifdef __WEIGHTED_INV_N_DISTANCE__
        double          d_cl = distance(x[r], cluster[cl]->center_of_mass) / cluster[cl]->n;
#elif defined(__WEIGHTED_INV_SQRT_N_DISTANCE__)
        double          d_cl = distance(x[r], cluster[cl]->center_of_mass) / sqrt((double) cluster[cl]->n);
#else
        /** This is the default. Weightless distance: **/
        double          d_cl = distance(x[r], cluster[cl]->center_of_mass);
#endif
        if (d_cl < mindist) {
            mindist = d_cl;
            nearest_cluster = cl;
        }
    }
    /** Return the nearest cluster ID: **/
    return nearest_cluster;
}   //end do_pseudo_k_means

void roundtrip_KNN_cluster_partition(size_t n)
{
    int             kmax = K;

    size_t          cl_population;

    size_t          last_K = 1;

    flag_t          have_previous_partition = 0;

    index_t        *previous_partition = pruning_calloc(n, sizeof(*previous_partition));

    index_t        *current_partition = pruning_calloc(n, sizeof(*current_partition));

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
                /** ignore unclustered particles **/
                if (KNN[q].hits < 1) {
                    continue;
                }
#endif
                cl_population = round_trip_KNN_explore(q);

#ifdef __REFUSE_SINGLETON_CLUSTER__
                /** since cl_population == 1, the particle inside is unclustered **/
                if (cl_population > 1) {
#elif defined(__REFUSE_NEGLIGIBLE_CLUSTERS__)
                /** refuse cluster populated with less than K + 1 points **/
                if (cl_population > K) {
#else
                /** if cluster is empty we are in trouble!!! **/
                if (cl_population > 0) {
#endif
                    clcount++;
                } else {
                    /** release unwanted cluster **/
                    set__destroy(cluster[clcount]);
                    cluster[clcount] = NULL;
                }
            }   //end if (not already_inserted...
        }   // end for (index_t q = ...

#ifdef __RECLUSTER_BY_PSEUDO_KMEANS__
        /** perform pseudo k-means if some particle was unclustered **/
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

        if (structural_pruning && clcount > 0)
            structurally_prune_clusters(n);

        fprintf(stdout, "K = %zu => %zu cluster%s\n", K, clcount,
                clcount == 1 ? "" : "s");

        canonical_partition(current_partition, n);
        if (have_previous_partition
            && memcmp(previous_partition, current_partition,
                      n * sizeof(*current_partition)) == 0) {
            verbose = 0;
            fprintf(stdout, "ignoring K = %zu since KNN topology is the same of K = %zu\n", K, last_K);
            continue;
        }

        last_K = K;
        memcpy(previous_partition, current_partition,
               n * sizeof(*previous_partition));
        have_previous_partition = 1;
        verbose = 1;

        showresults();

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

    free(current_partition);
    free(previous_partition);

}

/****************
 *  Show Results *
 ****************/

// char           *outputformat = "%s_K%dCs%d.layer";
// char           *outputerrmsg = "%s: cannot open output file %s... exiting\n";

// void showresults(void)
// {
//     char           *outputfname = calloc(strlen(outputprefix) + strlen(outputformat) + 10, 1);
//     int             cluster_number = clcount;

//     fprintf(stdout, "showresultas: output prefix = %s\n", outputprefix);

//     sprintf(outputfname, outputformat, outputprefix, K, clcount);

//     output = fopen(outputfname, "w");

//     if (output == NULL) {
//         fprintf(stderr, outputerrmsg, cmdname, outputfname);
//         exit(-666);
//     }
//     fprintf(stdout, "output file: %s @ handler %zu\n", outputfname, (size_t) output);
//     fprintf(stdout, "roundtrip_KNN_cluster_partition: performing the recursive roundtrip k-NN cluster discovery for K=%zu", K);
//     fprintf(stdout, " => %zd cluster", clcount);
//     if (clcount > 1)
//         fprintf(stdout, "s");
//     fprintf(stdout, " disclosed\n");

// #ifdef __VERBOSE_CLUSTER_ANALYSIS_OUTCOME__
//     fprintf(output, "K = %d\n", K);
//     for (clcount = 0; clcount < cluster_number; clcount++) {
//         if (cluster[clcount]->n > 0) {
//             fprintf(output, "cluster %i with %d data instances ", clcount + 1, cluster[clcount]->n);
//             fprintf(output, "with expectation vector\n");
//             for (index_t j = 0; j < dimension; j++) {
//                 fprintf(output, "\t%lg", cluster[clcount]->center_of_mass[j]);
//             }
//             fprintf(output, "\n");
//             cluster_list(cluster[clcount]);
//             fprintf(output, "\n");
//         }
//     }
// #else
//     for (clcount = 0; clcount < cluster_number; clcount++) {
//         if (cluster[clcount]->n > 0) {
//             cluster_list(cluster[clcount]);
//             fprintf(output, "\n");
//         }
//     }
// #endif

//     free(outputfname);
//     fclose(output);
// }

/**@<k-nn.c>::**/

 /**********************************
 * AUTHOR: Eraldo Pereira Marinho  *
 *
 * On Thu 20 Sep 15:22:55 -03 2018:
 *
 *  ***************************************
 *  * NOW WORKING WITH DYNAMIC ALLOCATION *
 *  ***************************************
 *
 * Cheers!!!
 *
 */

#include <k-nn.h>

distance_t     *q2nd;
flag_t         *disallow;

distance_t query2node_distance(index_t q, node_t node)
{
    distance_t      d = 0;

    index_t         j;

    for (j = 0; j < dimension; j++) {
        double          a,
                        b;
        int             outside;

        /*
         * query to node-boundary relative position
         */
        a = x[q][j] - hyperquadtree[hyperquadtree[node].parent].vertex[j];
        /*
         * node to node-boundary relative position
         */
        b = hyperquadtree[node].vertex[j] - hyperquadtree[hyperquadtree[node].parent].vertex[j];
        /*
         * (a*b) is positive or zero only if query is inside
         */
        if ((outside = (a * b < 0)) == 1) {
            d = max(d, a * a);
        }
    }

    return q2nd[node] = max(d, q2nd[hyperquadtree[node].parent]);
}

distance_t query2point_distance(const info_t * x, const info_t * y)
{
    int             j;
    double          r;

    for (r = j = 0; j < dimension; j++) {
        double          a = x[j] - y[j];

        r += a * a;
    }
    return r;
}

/*****
 * KNN data-structures:
 */
extern size_t   K;
size_t          KNN_population;
KNN_struct     *KNN;
KNN_indexer    *KNN_index_table;

void KNN_sort(index_t q)
{
    index_t         p = 0;
    int             stack = 0;
    index_t        *cursor = calloc(K, sizeof(index_t));
    index_t        *sortlist = calloc(K, sizeof(index_t));

    cursor[stack] = 0;

    while (1) {
        /*
         * visit left
         */
        if (KNN_index_table[cursor[stack]].left > 0) {
            cursor[stack + 1] = KNN_index_table[cursor[stack]].left;
            stack++;
            continue;
        }
        /*
         * visit current
         */
        if (KNN_index_table[cursor[stack]].left != -1) {
            sortlist[p++] = cursor[stack];
            KNN_index_table[cursor[stack]].left = -1;
        }
        /*
         * visit right
         */
        if (KNN_index_table[cursor[stack]].right > 0) {
            cursor[stack + 1] = KNN_index_table[cursor[stack]].right;
            KNN_index_table[cursor[stack]].right = -1;
            stack++;
            continue;
        }
        /*
         * emulate recursion return
         */
        else if (--stack == -1) {
            break;
        } else {
            if (KNN_index_table[cursor[stack]].left > 0)
                KNN_index_table[cursor[stack]].left = 0;
        }
    }

    for (int i = 0; i < K; i++) {
        int             k = sortlist[i];

        KNN[q].list[i] = KNN_index_table[k].ID;
        KNN[q].d[i] = KNN_index_table[k].r;
    }

    free(sortlist);
    free(cursor);
}

void KNN_insert(int q, int newone, double r)
{
    int             i,
                    j;

    for (i = 0; r > KNN[q].d[i]; ++i);

    for (j = KNN_population - 1; j > i; j--) {
        KNN[q].list[j] = KNN[q].list[j - 1];
        KNN[q].d[j] = KNN[q].d[j - 1];
    }

    KNN[q].list[i] = newone;
    KNN[q].d[i] = r;
    KNN[q].radius = KNN[q].d[KNN_population - 1];
}

void KNN_add(int q, int newone, double r)
{
    int             cursor = 0;

    KNN_index_table[KNN_population].ID = newone;
    KNN[q].radius = max(KNN_index_table[KNN_population].r = r, KNN[q].radius);
    if (KNN_population > 0) {
        while (1) {
            if (KNN_index_table[KNN_population].r < KNN_index_table[cursor].r) {
                if (KNN_index_table[cursor].left) {
                    cursor = KNN_index_table[cursor].left;
                    continue;
                }
                KNN_index_table[cursor].left = KNN_population;
            } else {
                if (KNN_index_table[cursor].right) {
                    cursor = KNN_index_table[cursor].right;
                    continue;
                }
                KNN_index_table[cursor].right = KNN_population;
            }
            break;
        }
    }
    KNN_population++;
    if (KNN_population == K) {
        KNN_sort(q);
    }
}

int isfaraway(index_t q, node_t node)
{
    return KNN_population * (query2node_distance(q, node) > KNN[q].radius);
}

void knn_search(size_t n)
{

  /**
    As an application, we are performing a topdown search for
    the k-nearest neighbor according to Marinho (1997) algorithm. **/

    node_t          level;
    size_t          height = depth + 1;
    node_t          child;
    node_t          node;
    ID_t            pID;
    node_t          hquadrant;
    distance_t      r;

    q2nd = calloc(levelend[depth], sizeof(*q2nd));
    disallow = calloc(levelend[depth], sizeof(*disallow));

    KNN_index_table = calloc(K, sizeof(*KNN_index_table));
    KNN = calloc(n, sizeof(*KNN));

    for (int q = 0; q < n; q++) {

        if (KNN[q].list == NULL) {
            KNN[q].d = calloc(K, sizeof(*KNN[q].d));
            KNN[q].list = calloc(K, sizeof(*KNN[q].list));
        }

        bzero(q2nd, levelend[depth] * sizeof(double));
        bzero(KNN_index_table, K * sizeof(KNN_indexer));
        bzero(disallow, levelend[depth] * sizeof(*disallow));
        KNN_population = 0;

        for (level = 1; level < height; level++) {

            for (node = levelbegin[level]; node < levelend[level]; node++) {

                if ((pID = hyperquadtree[node].identifier) != 0) {
                    int             qID = q + 1;

                    if (qID != pID) {
                        int             p = pID - 1;

                        r = query2point_distance(x[q], x[p]);
                        if (KNN_population < K) {
                            KNN_add(q, pID, r);
                        } else if (r <= KNN[q].radius) {
                            KNN_insert(q, pID, r);
                        }
                    }
                } else if (disallow[node] || (disallow[node] = isfaraway(q, node))) {
                    for (hquadrant = 0; hquadrant < grade; hquadrant++) {
                        if ((child = hyperquadtree[node].child[hquadrant]) > -1) {
                            disallow[child] = 1;
                        }
                    }
                }

            }

        }

    }
}

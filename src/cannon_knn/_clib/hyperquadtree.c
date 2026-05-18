/**@<hyperquadtree.c>::**/
/**********************************
 * AUTHOR: Eraldo Pereira Marinho *
 **********************************/
#include <hyperquadtree.h>

HQDTREE        *hyperquadtree;

/** once known n, and
       nnodes = (grade - 1) * n;
    it must be allocated as
       hyperquadtree = calloc(nnodes, sizeof(HQDTREE));
**/
size_t          nnodes; //nnodes = (grade - 1) * n;
node_t         *levelbegin;
node_t         *levelend;
size_t          depth;
node_t          node = 0;   //this is the node offset
node_t          next_child_node = 1;    //of course the next from 0 is 1

/**
 * It is required a cross-reference index to locate which node a particle is residing in
 */
int            *crossreference; //crossreference[NMAX];

/**
 * Author: Eraldo Pereira Marinho, PhD Reference: Marinho 1997, Doctoral
 * Thesis, Chapter 8. SPMD Quadtree code is a single program stream
 * multiple data stream approach to run on vector architectures like gpus
 * etc or in clusters. The method consists of a level-first building of a
 * quadtree data-structure with the help of a double-hash table.
 */
int           **hashTB; //hashTB[NMAX][HASHTBCOLS];

/**
 * Index calculation:
 *
 * i-hash = \sum_{j=1}^2{(x[i][j]-mu[j] > 0)*2^{2-j}}, where mu[] is the center
 * of mass of the node hosting the i-particle with coordinates x[].
 *
 * e-hash = HASHTBCOLS * e-hash + i-hash.
 */

int            *collision;  //collision[MAXGRD];

void ihash(int i1, int i2)
{
    hyperquadtree[node].vertex = calloc(dimension, sizeof(double));
    /**
     * the [i1, i2)-range is semantically a node in the current quadtree
     * level.
     */

    if (i2 == i1 + 1) {
        int             j;
        int             k = hashTB[i1][INDEX] - 1;

        hashTB[i1][EHASH] = hashTB[i1][IHASH] = -1;
        hashTB[i1][LEAF] = 1;
        hyperquadtree[crossreference[k] = node].identifier = hashTB[i1][INDEX];
        for (j = 0; j < dimension; j++) {
            hyperquadtree[node].vertex[j] = x[k][j];
        }

    } else {

        int /** n is the hyperquadrant population number **/ n = i2 - i1;

        bzero(collision, sizeof(int) * grade);

        /** get first momentum in parallel **/
        for (int i = i1; i < i2; i++) {
            int             k = hashTB[i][INDEX] - 1;

            for (int j = 0; j < dimension; j++) {
                hyperquadtree[node].vertex[j] += x[k][j];
            }
        }

        /** normalize center of mass in parallel **/
        for (int j = 0; j < dimension; j++) {
            hyperquadtree[node].vertex[j] /= n;
        }

        /** compute i-hash in parallel **/
        for (int i = i1; i < i2; i++) {
            int             k = hashTB[i][INDEX] - 1;

            for (int j = hashTB[i][IHASH] = 0; j < dimension; j++) {
                hashTB[i][IHASH] *= 2;
                hashTB[i][IHASH] += (x[k][j] - hyperquadtree[node].vertex[j] > 0);
            }
        }

        for (int i = i1; i < i2; i++) {
            collision[hashTB[i][IHASH]]++;
        }

        /** duplicates check **/
        {
            int             has_singularity,
                            singular_hquadrant;

            for (int i = has_singularity = 0; i < grade; i++) {
                if (collision[i] == 0) {
                    singular_hquadrant = i;
                    has_singularity++;
                }
            }
            has_singularity = (has_singularity == grade - 1);
            if (has_singularity) {
                fprintf(stderr,
                        "hyperquadtree: singularity at hyperquadrant %i in hashTB range [%d-%d] \n", singular_hquadrant, i1, i2);
                exit(-666);
            }
        }

        hyperquadtree[node].child = calloc(grade, sizeof(int));

        for (int i = 0; i < grade; i++) {
            if (collision[i] > 0) {
                hyperquadtree[next_child_node].parent = node;
                hyperquadtree[node].child[i] = next_child_node++;
            } else {
                hyperquadtree[node].child[i] = -1;
            }
        }
    }

    hyperquadtree[node].population = i2 - i1;
}

void ehash(size_t n)
{
    int             newindex;

    for (int i = 0; i < n; i++) {
        hashTB[i][EHASH] *= grade;
        hashTB[i][EHASH] += hashTB[i][IHASH];
    }

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (hashTB[i][EHASH] > hashTB[j][EHASH]) {
                for (int k = 0; k < HASHTBCOLS; k++) {
                    int             aux = hashTB[i][k];

                    hashTB[i][k] = hashTB[j][k];
                    hashTB[j][k] = aux;
                }
            }
        }
    }

    int             i = 1;

    for (newindex = 0; i < n; i++) {
        if (hashTB[i - 1][EHASH] < hashTB[i][EHASH])
            hashTB[i - 1][EHASH] = newindex++;
        else
            hashTB[i - 1][EHASH] = newindex;
    }
    hashTB[i - 1][EHASH] = newindex;

}

size_t leafprune(size_t n)
{
    size_t          j = 0;
    index_t       **TBaux;

    /*** allocate a clean auxiliary heap space ***/
    TBaux = calloc(n, sizeof(int *));
    for (index_t i = 0; i < n; i++) {
        TBaux[i] = calloc(HASHTBCOLS, sizeof(int));
    }

    /*** ignore leafs in the older e-hash table and copy to the temporary table
         the non leaf entries ***/
    for (index_t i = 0; i < n; i++) {
        if (not hashTB[i][LEAF]) {
            for (int k = 0; k < HASHTBCOLS; k++) {
                TBaux[j][k] = hashTB[i][k];
            }
            /*** count the last copy ***/
            j++;
        }
    }

    if (j < n) {
        for (index_t i = 0; i < j; i++) {
            for (index_t k = 0; k < HASHTBCOLS; k++) {
                hashTB[i][k] = TBaux[i][k];
            }
        }
    } else {
        /*** weirdly, no leafs were pruned, so something is wrong ***/
        ;
    }

    /*** release TBaux heap space ***/
    for (index_t i = n; i > 0; i--) {
        free(TBaux[i - 1]);
    }
    free(TBaux);

    return j;
}

void hashTBinit(size_t n)
{
    int             i;

    hashTB = calloc(n, sizeof(*hashTB));
    for (i = 0; i < n; i++) {
        hashTB[i] = calloc(HASHTBCOLS, sizeof(**hashTB));
        hashTB[i][INDEX] = i + 1;
    }
}

/**hyperquadtree_init::
 *    levelbegin      = calloc(n, sizeof(int));
 *    levelend        = calloc(n, sizeof(int));
 *    hyperquadtree   = calloc(nnodes, sizeof(HQDTREE));
 *    collision       = calloc(grade, sizeof(int));
 *    crossreference  = calloc(n, sizeof(int));
 **/
void hyperquadtree_init(size_t n)
{
    nnodes = (grade - 1) * n;
    levelbegin = calloc(n, sizeof(int));
    levelend = calloc(n, sizeof(int));
    hyperquadtree = calloc(nnodes, sizeof(HQDTREE));
    collision = calloc(grade, sizeof(int));
    crossreference = calloc(n, sizeof(int));
}

void hyperquadtree_build(size_t n)
{
    int             level;

    hashTBinit(n);
    hyperquadtree_init(n);

    for (level = 0; n; level++) {

        int             i1,
                        i2;

        levelbegin[level] = node;
        levelend[level] = next_child_node;

        /** foreach node, do **/
        for (i1 = 0; i1 < n; node++, i1 = i2) {

            for (i2 = i1 + 1; i2 < n && hashTB[i2 - 1][EHASH] == hashTB[i2][EHASH]; i2++);

            ihash(i1, i2);

        }

        /** delete all just formed leafs **/
        n = leafprune(n);

        /** if n == 0, there is no more particle to index **/
        if (n == 0) {
            break;
        }

        ehash(n);
    }

    depth = level;
}

extern size_t   N;

void hyperquadtree_destroy(void)
{
    for (int i = N - 1; i < -1; i--) {
        free(hashTB[i]);
    }
    free(hashTB);
    free(crossreference);
    free(collision);
    free(hyperquadtree);
    free(levelend);
    free(levelbegin);
}

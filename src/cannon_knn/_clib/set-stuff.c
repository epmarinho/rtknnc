/**@<set-stuff.c>::**/

 /********************************\
 * AUTHOR: Eraldo Pereira Marinho *
 * 1st VERSION DATE: 23 Aug, 2018 *
 \********************************/

#include "set-stuff.h"
#include "k-nn-topology.h"

 /*==============================================================\
 |            SET OPERATIONS HAVE TO BE IMPLEMENTED HERE         |
 \==============================================================*/

void           *set__singleton_coordinates(SET * set, ID_t id)
{
    if (set->center_of_mass == NULL) {
        set->center_of_mass = calloc(sizeof(double), dimension);
    }
    set->size = KNN[id - 1].visibility;
    return memcpy(set->center_of_mass, x[id - 1], dimension * sizeof(double));
}

void           *set__add_coordinates(SET * A, ID_t id)
{
    if (A == NULL) {

        return x[id - 1];

    } else {

        size_t          n = A->n;
        size_t          n_plus_1 = 1 + n;

        for (index_t j = 0; j < dimension; j++) {
            A->center_of_mass[j] = (x[id - 1][j] + n * A->center_of_mass[j]) / n_plus_1;
        }
        A->size = (KNN[id - 1].visibility + n * A->size) / n_plus_1;
        return A->center_of_mass;
    }
}

size_t set__insert(index_t id, SET * t)
{
    /*==================================================\
    | do the descent through only one of the sub-trees, |
    | either through the left or through the right      |
    \==================================================*/

    if (id < t->id) {

        /*===============================================\
        | descending though, or branching from, the left |
        \===============================================*/

        SET            *left = t->left;

        if (left == NULL) {

            /*=====================\
            | branch from the left |
            \=====================*/

            t->left = left = calloc(1, sizeof(SET));
            left->id = id;
            left->center_of_mass = set__singleton_coordinates(left, id);
            left->n = 1;

        } else {

            /*=========================================\
            |  At this point everything we got is that |
            |  t->left is not empty                    |
            \=========================================*/

            set__insert(id, left);

        }

    } else if (id > t->id) {

        /*================================================\
        | descending though, or branching from, the right |
        \================================================*/

        SET            *right = t->right;

        if (right == NULL) {

            /*======================\
            | branch from the right |
            \======================*/

            t->right = right = calloc(1, sizeof(SET));
            right->id = id;
            right->center_of_mass = set__singleton_coordinates(right, id);
            right->n = 1;

        } else {

            /*=========================================\
            |  At this point everything we got is that |
            |  t->right is not empty                   |
            \=========================================*/

            set__insert(id, right);

        }

    } else {

        /*=========================================\
        | no way to reaching out this point unless |
        | no new member has been just  inserted in |
        | the present recursion level              |
        \=========================================*/

        return 0;
    }

    /*================================================\
    | we are here because id was just inserted either |
    | through the right or through the left subtrees  |
    \================================================*/

    t->center_of_mass = set__add_coordinates(t, id);

    return ++t->n;
}

/*** emulates the membership function by visiting the bin-tree in post-order: ***/
flag_t set__in(ID_t qID, SET * set)
{
    if (set->left != NULL) {
        if (set__in(qID, set->left))
            return 1;
    }

    if (set->right != NULL) {
        if (set__in(qID, set->right))
            return 1;
    }

    if (set->left == NULL && set->right == NULL) {
        if (set->id == qID)
            return 1;
    }

    return 0;
}

void node_destroy(SET * t)
{
    /** release current node **/
    free(t->center_of_mass);
    bzero(t, sizeof(SET));
    free(t);
}

/** @set__destroy:: shall follow the reverse allocation tree */
void set__destroy(SET * t)
{
    /** then, release right subtree **/
    if (t->right != NULL) {
        set__destroy(t->right);
    }
    /** right subtree is released now **/

    /** middle node is here: **/
    unclustered[n_unclustered++] = t->id;

    /** release left subtree **/
    if (t->left != NULL) {
        set__destroy(t->left);
    }
    /** left subtree is released now **/

    /** release the current node **/
    node_destroy(t);
    /** current node released **/
}

void set__list(SET * set)
{
    int j;

    if (set->left != NULL) {
        set__list(set->left);
    }

    fprintf(output, "%i: %i\t", already_inserted[set->id - 1], set->id);
    for (j = 0; j < dimension - 1; j++) {
        fprintf(output, "%lg\t", x[set->id - 1][j]);
    }
    fprintf(output, "%lg\n", x[set->id - 1][j]);

    if (set->right != NULL) {
        set__list(set->right);
    }
}
#include "AI.h"

int main()//(int argc, char *argv[])
{
    // srand(time(NULL));
    srand(4);


    double Xx[X_TRAIN_SIZE][Y_TRAIN_SIZE] = {
            {1, 1},
            {0, 1},
            {1, 0},
            {0, 0}
    };

    double yy[X_TRAIN_SIZE] = {
            1,
            0,
            0,
            1};


    // transform tabs to Matrix
    Matrix *X = malloc(sizeof(Matrix));
    X->sizeX = X_TRAIN_SIZE;
    X->sizeY = Y_TRAIN_SIZE;
    X->data = malloc(X_TRAIN_SIZE * Y_TRAIN_SIZE * sizeof(double));
    for (int i = 0; i < Y_TRAIN_SIZE; i++) {
        for (int j = 0; j < X_TRAIN_SIZE; j++) {
            X->data[i * X_TRAIN_SIZE + j] = Xx[j][i];
        }
    }
    Matrix *y = malloc(sizeof(Matrix));
    y->sizeX = X_TRAIN_SIZE;
    y->sizeY = 1;
    y->data = malloc(X_TRAIN_SIZE * sizeof(double));
    for (size_t i = 0; i < X_TRAIN_SIZE; i++) {
        y->data[i] = yy[i];
    }



    Matrix *W_list;
    Matrix *b_list;
    int dim[] = {2, 3, 4, 3, 1};
    init_network(dim, &W_list, &b_list);

    Matrix *A_list;
    forward_propagation(X, W_list, b_list, &A_list);
    for (size_t i = 0; i < DIMENSION; i++)
    {
        printMatrix(A_list[i]);
    }










    // Free all
    for (size_t i = 0; i < DIMENSION - 1; i++)
    {
        free((W_list[i]).data);
        free((b_list[i]).data);
        free((A_list[i + 1]).data); //because of memcpy
    }

    free(X->data);
    free(X);
    free(y->data);
    free(y);

    free(W_list);
    free(b_list);
    free(A_list);


    return 0;
}
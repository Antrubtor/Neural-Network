#include "AI.h"

int main()//(int argc, char *argv[])
{
    printf("cc");
    // srand(time(NULL));
    srand(4);


    double Xx[Y_TRAIN_SIZE][X_TRAIN_SIZE] = {
            {1, 1},
            {0, 1},
            {1, 0},
            {0, 0}
    };

    double yy[Y_TRAIN_SIZE] = {
            1,
            0,
            0,
            1};



    // transform tabs to malloc
    double *X = malloc(X_TRAIN_SIZE * Y_TRAIN_SIZE * sizeof(double));
    for (size_t i = 0; i < Y_TRAIN_SIZE; i++) {
        for (size_t j = 0; j < X_TRAIN_SIZE; j++) {
            X[i * X_TRAIN_SIZE + j] = Xx[i][j];
        }
    }
    double *y = malloc(Y_TRAIN_SIZE * sizeof(double));
    for (size_t i = 0; i < Y_TRAIN_SIZE; i++) {
        y[i] = yy[i];
    }



    Matrix *W_list;
    Matrix *b_list;
    int dim[] = {2, 3, 4, 3, 1};
    init_network(dim, DIMENSION, &W_list, &b_list);

    double *activations
    forward_propagation()

    for (size_t i = 0; i < DIMENSION - 1; i++)
    {
        free((W_list[i]).data);
        free((b_list[i]).data);
    }
    free(y);
    free(X);
    free(W_list);
    free(b_list);


    return 0;
}



/*
    printf("X: ");
    for (size_t i = 0; i < Y_TRAIN_SIZE * X_TRAIN_SIZE; i++) printf("%f, ", X[i]);
    printf("\n");
    printf("\ny: ");
    for (size_t i = 0; i < Y_TRAIN_SIZE; i++) printf("%f, ", y[i]);
    printf("\n");



 */
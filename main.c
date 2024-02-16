#include "AI.h"

int main(int argc, char *argv[])
{
    int hidden_layers[] = {Y_TRAIN_SIZE, 4,OUTPUT_SIZE};

    srand(time(NULL));


    Matrix *W_list;
    Matrix *b_list;
    Matrix *X;
    Matrix *y;
    if (argc > 1 && strcmp(argv[1], "XOR") == 0)
    {
        printf("XOR\n");
        double Xx[X_TRAIN_SIZE][Y_TRAIN_SIZE] = {
                {1, 1},
                {0, 1},
                {1, 0},
                {0, 0}
        };

        double yy[X_TRAIN_SIZE] = {
                0,
                1,
                1,
                0};
        double learning_rate = 0.1;


        // transform tabs to Matrix
        X = malloc(sizeof(Matrix));
        X->sizeX = X_TRAIN_SIZE;
        X->sizeY = Y_TRAIN_SIZE;
        X->data = malloc(X_TRAIN_SIZE * Y_TRAIN_SIZE * sizeof(double));
        for (int i = 0; i < Y_TRAIN_SIZE; i++) {
            for (int j = 0; j < X_TRAIN_SIZE; j++) {
                X->data[i * X_TRAIN_SIZE + j] = Xx[j][i];
            }
        }
        y = malloc(sizeof(Matrix));
        y->sizeX = X_TRAIN_SIZE;
        y->sizeY = 1;
        y->data = malloc(X_TRAIN_SIZE * sizeof(double));
        for (size_t i = 0; i < X_TRAIN_SIZE; i++) {
            y->data[i] = yy[i];
        }



//        int hidden_layers[] = {Y_TRAIN_SIZE, 4,OUTPUT_SIZE};
//        int dim = sizeof(hidden_layer) / sizeof(hidden_layer[0]);
//        printf("size %i\n", dim);
        neural_network(X, y, hidden_layers, learning_rate, &W_list, &b_list);


        Matrix *pre = malloc(sizeof(Matrix));
        pre->sizeX = 1;
        pre->sizeY = 2;
        pre->data = malloc(2 * sizeof(double));
        pre->data[0] = 1;
        pre->data[1] = 1;
        printf("Predict for (1, 1): %f\n", predict(pre, W_list, b_list));
        pre->data[0] = 0;
        pre->data[1] = 1;
        printf("Predict for (0, 1): %f\n", predict(pre, W_list, b_list));
        pre->data[0] = 1;
        pre->data[1] = 0;
        printf("Predict for (1, 0): %f\n", predict(pre, W_list, b_list));
        pre->data[0] = 0;
        pre->data[1] = 0;
        printf("Predict for (0, 0): %f\n", predict(pre, W_list, b_list));
        free(pre->data);
        free(pre);
    }
    else if (argc > 1 && strcmp(argv[1], "MNIST") == 0)
    {

    }



    // Free all
    for (size_t i = 0; i < DIMENSION - 1; i++)
    {
        free((W_list[i]).data);
        free((b_list[i]).data);
    }

    free(X->data);
    free(X);
    free(y->data);
    free(y);

    free(W_list);
    free(b_list);


    return 0;
}
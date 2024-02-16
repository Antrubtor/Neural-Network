#include "AI.h"

int main(int argc, char *argv[])
{
    srand(time(NULL));
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
        int hidden_layer[] = {2, 4, 4, 1};
        printf("size %lu\n", sizeof(hidden_layer) / sizeof(hidden_layer[0]));
        neural_network(X, y, hidden_layer, learning_rate, 5000, &W_list, &b_list);


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
    }
    else if (argc > 1 && strcmp(argv[1], "MNIST") == 0)
    {

    }




    return 0;
}
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
    int hidden_layer[] = {2, 3, 4, 3, 1};

    neural_network(X, y, hidden_layer, learning_rate, 10000, &W_list, &b_list);


    Matrix *pre = malloc(sizeof(Matrix));
    pre->sizeX = 1;
    pre->sizeY = 2;
    pre->data = malloc(2 * sizeof(double));
    pre->data[0] = 1;
    pre->data[1] = 1;
    printf("Predict for (1, 1): %f\n", predict(pre, W_list, b_list));
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

    return 0;
}















//#include "AI.h"
//
//int main()//(int argc, char *argv[])
//{
//    // srand(time(NULL));
////    srand(4);
//
//
//    double Xx[X_TRAIN_SIZE][Y_TRAIN_SIZE] = {
//            {1, 1},
//            {0, 1},
//            {1, 0},
//            {0, 0}
//    };
//
//    double yy[X_TRAIN_SIZE] = {
//            0,
//            1,
//            1,
//            0};
//    double learning_rate = 0.1;
//
//
//    // transform tabs to Matrix
//    Matrix *X = malloc(sizeof(Matrix));
//    X->sizeX = X_TRAIN_SIZE;
//    X->sizeY = Y_TRAIN_SIZE;
//    X->data = malloc(X_TRAIN_SIZE * Y_TRAIN_SIZE * sizeof(double));
//    for (int i = 0; i < Y_TRAIN_SIZE; i++) {
//        for (int j = 0; j < X_TRAIN_SIZE; j++) {
//            X->data[i * X_TRAIN_SIZE + j] = Xx[j][i];
//        }
//    }
//    Matrix *y = malloc(sizeof(Matrix));
//    y->sizeX = X_TRAIN_SIZE;
//    y->sizeY = 1;
//    y->data = malloc(X_TRAIN_SIZE * sizeof(double));
//    for (size_t i = 0; i < X_TRAIN_SIZE; i++) {
//        y->data[i] = yy[i];
//    }
//
//
//
//    Matrix *W_list;
//    Matrix *b_list;
//    int dim[] = {2, 3, 4, 3, 1};
//
//
//    init_network(dim, &W_list, &b_list);
//    for (float i = 0; i < 1; i++)
//    {
//        Matrix *A_list;
//        forward_propagation(X, W_list, b_list, &A_list);
//        Matrix *dW_gradients;
//        Matrix *db_gradients;
//        back_propagation(y, W_list, A_list, &dW_gradients, &db_gradients);
//        update(dW_gradients, db_gradients, W_list, b_list, learning_rate);
////        printf("Log loss: %f\n", log_loss(y, &A_list[DIMENSION - 1]));
//        for (size_t j = 0; j < DIMENSION - 1; j++)
//        {
//            free((A_list[j + 1]).data); //because of memcpy
//            free(dW_gradients[j].data);
//            free(db_gradients[j].data);
//        }
////        free(A_list[0].data);
//        free(A_list);
//        free(dW_gradients);
//        free(db_gradients);
//    }
//
//
//    for (size_t i = 0; i < DIMENSION - 1; i++)
//    {
//        printf("W_list:\n");
//        printMatrix(W_list[i]);
////        printf("b_list:\n");
////        printMatrix(b_list[i]);
//    }
//    printf("\n\n\n\n\n\n\n\n\n\n");
//    for (size_t i = 0; i < DIMENSION - 1; i++)
//    {
////        printf("W_list:\n");
////        printMatrix(W_list[i]);
//        printf("b_list:\n");
//        printMatrix(b_list[i]);
//    }
//
//
//
//
//    // Free all
//    for (size_t i = 0; i < DIMENSION - 1; i++)
//    {
//        free(W_list[i].data);
//        free(b_list[i].data);
//        printf("oui\n");
//    }
//    printf("oui\n");
//    free(X->data);
//    free(X);
//    free(y->data);
//    free(y);
//
//    free(W_list);
//    free(b_list);
//
//    return 0;
//}
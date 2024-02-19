#include "AI.h"

int main(int argc, char *argv[])
{
    int hidden_layers[] = {Y_TRAIN_SIZE, 32, 32, OUTPUT_SIZE};
    Matrix *W_list;
    Matrix *b_list;
//    Matrix *X;
//    Matrix *y;
    Matrix *X = malloc(X_TRAIN_SIZE * sizeof(Matrix));
    Matrix *y = malloc(X_TRAIN_SIZE * sizeof(Matrix));

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


        // transform tabs to Matrix

        for (int i = 0; i < X_TRAIN_SIZE; i ++) {
            X[i].sizeX = 1;
            X[i].sizeY = Y_TRAIN_SIZE;
            X[i].data = malloc(Y_TRAIN_SIZE * sizeof(double));
            for (int j = 0; j < Y_TRAIN_SIZE; j++) {
                X[i].data[j] = Xx[i][j];
            }
            y[i].sizeX = 1;
            y[i].sizeY = 1;
            y[i].data = malloc(sizeof(double));
            y[i].data[0] = yy[i];
        }


        neural_network(&X, &y, hidden_layers, &W_list, &b_list);

        printf("\nPredictions:\n");
        Matrix *pre = malloc(sizeof(Matrix));
        pre->sizeX = 1;
        pre->sizeY = 2;
        pre->data = malloc(2 * sizeof(double));
        pre->data[0] = 1;
        pre->data[1] = 1;
        printf("Predict for (1, 1): %.8f\n", predict(pre, W_list, b_list));
        pre->data[0] = 0;
        pre->data[1] = 1;
        printf("Predict for (0, 1): %.8f\n", predict(pre, W_list, b_list));
        pre->data[0] = 1;
        pre->data[1] = 0;
        printf("Predict for (1, 0): %.8f\n", predict(pre, W_list, b_list));
        pre->data[0] = 0;
        pre->data[1] = 0;
        printf("Predict for (0, 0): %.8f\n", predict(pre, W_list, b_list));
        free(pre->data);
        free(pre);
    }
    else if (argc > 1 && strcmp(argv[1], "MNIST") == 0)
    {
        char *filename_images = "data/train-images-idx3-ubyte";
        char *filename_labels = "data/train-labels-idx1-ubyte";
        load_mnist(filename_images, filename_labels, X, y, X_TRAIN_SIZE);
//        printMatrix(*images);
//
//        Matrix mat;
//        mat.sizeX = 28;
//        mat.sizeY = 28;
//        mat.data = (double *)malloc(mat.sizeX * mat.sizeY * sizeof(double));
//        for (int j = 0; j < X_TRAIN_SIZE; j++)
//        {
//            for (int i = 0; i < 28 * 28; i++)
//            {
//                mat.data[i] = images->data[j + i * X_TRAIN_SIZE];
//            }
//            char name[] = "img/img1.png";
//            name[7] = '0' + j;
//            matrixToImage(&mat, name);
//        }
//        free(mat.data);
//
        printf("MNIST loaded\n");
        neural_network(&X, &y, hidden_layers, &W_list, &b_list);
        Matrix *pre = malloc(10000 * sizeof(Matrix));
        Matrix *res = malloc(10000 * sizeof(Matrix));
        char *filename_images_train = "data/t10k-images-idx3-ubyte";
        char *filename_labels_train = "data/t10k-labels-idx1-ubyte";
        int nbr_image_test = 10000;
        load_mnist(filename_images_train, filename_labels_train, pre, res, nbr_image_test);
        double accuracy = 0;
        for (int i = 0; i < nbr_image_test; i++)
        {
            int max = 0;
            double max_nbr = 0;
            for (int j = 0; j < 10; j++)
            {
                if (res[i].data[j] > max_nbr)
                {
                    max = j;
                    max_nbr = res[i].data[j];
                }
            }

            accuracy += predict_test(&pre[i], max, W_list, b_list);
        }
        printf("Accuracy: %f\n", accuracy / nbr_image_test);

        for (int i = 0; i < nbr_image_test; i++)
        {
            free(pre[i].data);
            free(res[i].data);
        }
        free(pre);
        free(res);


    }
    else
        printf("Please choose between XOR and MNIST\n");


    if (argc > 1 && (strcmp(argv[1], "XOR") == 0 || strcmp(argv[1], "MNIST") == 0))
    {
        // Free all
        for (size_t i = 0; i < DIMENSION - 1; i++) {
            free((W_list[i]).data);
            free((b_list[i]).data);
        }
        if (strcmp(argv[1], "MNIST") != 0) {
            for (int i = 0; i < X_TRAIN_SIZE; i++)
            {
                free(X[i].data);
                free(y[i].data);
            }
            free(X);
            free(y);
        }

        free(W_list);
        free(b_list);
    }


    return 0;
}
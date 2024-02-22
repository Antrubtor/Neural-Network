#include "AI.h"

int main(int argc, char *argv[])
{
    int hidden_layers[] = {Y_TRAIN_SIZE, 32, 32, OUTPUT_SIZE};  //example for XOR and MNIST
    Matrix *W_list;
    Matrix *b_list;
    Matrix *X = malloc(X_TRAIN_SIZE * sizeof(Matrix));
    Matrix *y = malloc(X_TRAIN_SIZE * sizeof(Matrix));

    char network_file[] = "train/trained_network";

    srand(time(NULL));

    if (argc > 1 && (strcmp(argv[1], "XOR") == 0 || strcmp(argv[1], "UXOR") == 0))
    {
        printf("Learning and saving the network for XOR data\n");
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
        int update_net = 0;
        int epoch_nbr = EPOCH;
        if (strcmp(argv[1], "UXOR") == 0) {
            epoch_nbr += load_network(&W_list, &b_list, network_file);
            update_net = 1;
        }
        neural_network(&X, &y, hidden_layers, &W_list, &b_list, update_net);
        save_network(W_list, b_list, network_file, epoch_nbr);
    }
    else if (argc > 1 && (strcmp(argv[1], "MNIST") == 0 || strcmp(argv[1], "UMNIST") == 0))
    {
        printf("Learning and saving the network for MNIST data\n");
        char *filename_images = "data/train-images-idx3-ubyte";
        char *filename_labels = "data/train-labels-idx1-ubyte";
        load_mnist(filename_images, filename_labels, X, y, X_TRAIN_SIZE);

        int update_net = 0;
        int epoch_nbr = EPOCH;
        if (strcmp(argv[1], "UMNIST") == 0) {
            epoch_nbr += load_network(&W_list, &b_list, network_file);
            update_net = 1;
        }
        neural_network(&X, &y, hidden_layers, &W_list, &b_list, update_net);
        save_network(W_list, b_list, network_file, epoch_nbr);
    }


    if ((argc > 1 && strcmp(argv[1], "TXOR") == 0) || (argc > 2 && strcmp(argv[2], "TXOR") == 0))
    {
        if (argc == 2) {
            printf("Loading and testing the network for XOR data\n");
            load_network(&W_list, &b_list, network_file);
        }
        else
            printf("Testing the network that has just been trained on the XOR data\n");

        printf("\nPredictions:\n");
        Matrix *pre = malloc(sizeof(Matrix));
        pre->sizeX = 1;
        pre->sizeY = 2;
        pre->data = malloc(2 * sizeof(double));

        pre->data[0] = 1; pre->data[1] = 1;
        Matrix *res = predict(pre, W_list, b_list, 1);
        free(res->data);
        free(res);
        pre->data[0] = 0; pre->data[1] = 1;
        res = predict(pre, W_list, b_list, 1);
        free(res->data);
        free(res);
        pre->data[0] = 1; pre->data[1] = 0;
        res = predict(pre, W_list, b_list, 1);
        free(res->data);
        free(res);
        pre->data[0] = 0; pre->data[1] = 0;
        res = predict(pre, W_list, b_list, 1);
        free(res->data);
        free(res);
        free(pre->data);
        free(pre);
    }
    if ((argc > 1 && strcmp(argv[1], "TMNIST1") == 0) || (argc > 2 && strcmp(argv[2], "TMNIST1") == 0) ||
            (argc > 1 && strcmp(argv[1], "TMNIST2") == 0) || (argc > 2 && strcmp(argv[2], "TMNIST2") == 0))
    {
        if (argc == 2) {
            printf("Loading and testing the network for MNIST data\n");
            load_network(&W_list, &b_list, network_file);
        }
        else
            printf("Testing the network that has just been trained on the MNIST data\n");

        if (strcmp(argv[1], "TMNIST1") == 0 || (argc > 2 && strcmp(argv[2], "TMNIST1") == 0))
        {
            int nbr_image_test = 10000;
            Matrix *pre = malloc(nbr_image_test * sizeof(Matrix));
            Matrix *res = malloc(nbr_image_test * sizeof(Matrix));
            char *filename_images_train = "data/t10k-images-idx3-ubyte";
            char *filename_labels_train = "data/t10k-labels-idx1-ubyte";
            load_mnist(filename_images_train, filename_labels_train, pre, res, nbr_image_test);
            printf("Accuracy: %.2f%% on MNIST test data\n", accuracy(pre, res, nbr_image_test, W_list, b_list));

            for (int i = 0; i < nbr_image_test; i++) {
                free(pre[i].data);
                free(res[i].data);
            }
            free(pre);
            free(res);
        }
        else if (strcmp(argv[1], "TMNIST2") == 0 || (argc > 2 && strcmp(argv[2], "TMNIST2") == 0))
        {
            printf("Predictions:\n");
            Matrix *IMG = image_to_matrix("img/test_img.png");
            Matrix *pre = predict(&IMG[0], W_list, b_list, 0);
            int check = 0;
            for (int i = 0; i < pre->sizeX * pre->sizeY; i++) {
                if (pre->data[i] == 1) {
                    printf("It's a %i !\n", i);
                    check = 1;
                    break;
                }
            }
            if (check == 0) printf("Unrecognized\n");
            free(IMG->data);
            free(IMG);
            free(pre->data);
            free(pre);
        }
    }



    if ((argc > 1 && (strcmp(argv[1], "XOR") == 0 || strcmp(argv[1], "MNIST") == 0 ||
                    strcmp(argv[1], "UXOR") == 0 || strcmp(argv[1], "UMNIST") == 0 ||
                    strcmp(argv[1], "TXOR") == 0 || strcmp(argv[1], "TMNIST1") == 0 || strcmp(argv[1], "TMNIST2") == 0)) ||
                    (argc > 2 && (strcmp(argv[2], "TXOR") == 0 || strcmp(argv[2], "TMNIST1") == 0 || strcmp(argv[2], "TMNIST2") == 0)))
    {
        // Free all
        for (size_t i = 0; i < DIMENSION - 1; i++) {
            free((W_list[i]).data);
            free((b_list[i]).data);
        }

        if (strcmp(argv[1], "XOR") == 0 || strcmp(argv[1], "MNIST") == 0 ||
                strcmp(argv[1], "UXOR") == 0 || strcmp(argv[1], "UMNIST") == 0) {
            for (int i = 0; i < X_TRAIN_SIZE; i++) {
                free(X[i].data);
                free(y[i].data);
            }
        }
        free(X);
        free(y);

        free(W_list);
        free(b_list);
    }


    return 0;
}
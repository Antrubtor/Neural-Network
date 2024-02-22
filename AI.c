#include "AI.h"

double random_gaussian()
{
    return ((double)rand() / RAND_MAX) * 2 - 1;
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}


void init_network(int dim[], Matrix **W_list, Matrix **b_list)
{
    *W_list = malloc((DIMENSION - 1) * sizeof(Matrix));
    *b_list = malloc((DIMENSION - 1) * sizeof(Matrix));
    for (int i = 1; i < DIMENSION; i++)
    {
        Matrix *tmp_W = &(*W_list)[i - 1];
        tmp_W->sizeX = dim[i - 1];
        tmp_W->sizeY = dim[i];
        tmp_W->data = malloc(tmp_W->sizeX * tmp_W->sizeY * sizeof(double));
        for (int j = 0; j < tmp_W->sizeX * tmp_W->sizeY; j++)
            tmp_W->data[j] = random_gaussian();
        Matrix *tmp_b = &(*b_list)[i - 1];
        tmp_b->sizeX = 1;
        tmp_b->sizeY = dim[i];
        tmp_b->data = malloc(tmp_b->sizeY * sizeof(double));
        for (int j = 0; j < tmp_b->sizeY; j++)
            tmp_b->data[j] = random_gaussian();
    }
}

void forward_propagation(Matrix *X, Matrix *W_list, Matrix *b_list, Matrix **A_list)
{
    *A_list = malloc(DIMENSION * sizeof(Matrix));
    A_list[0]->sizeX = X->sizeX;
    A_list[0]->sizeY = X->sizeY;
    A_list[0]->data = malloc(X->sizeX * X->sizeY * sizeof(double));
    for (int i = 0; i < X->sizeX * X->sizeY; i++)
        A_list[0]->data[i] = X->data[i];

    for (int i = 0; i < DIMENSION - 1; i++) {
        Matrix *Z = mul(&W_list[i], &(*A_list)[i]);
        Matrix *tmp_A = &(*A_list)[i + 1];
        tmp_A->sizeX = Z->sizeX;
        tmp_A->sizeY = Z->sizeY;
        tmp_A->data = malloc(Z->sizeX * Z->sizeY * sizeof(double));
        for (int j = 0; j < Z->sizeY ; j++) {
            for (int k = 0; k < Z->sizeX; k++) {
                double tmp = Z->data[j * Z->sizeX + k] + b_list[i].data[j];
                tmp_A->data[j * Z->sizeX + k] = 1 / (1 + exp(-(tmp)));
            }
        }
        free(Z->data);
        free(Z);
    }
}

void back_propagation(Matrix *y, Matrix *W_list, Matrix *A_list, Matrix **dW_gradients, Matrix **db_gradients)
{
    *dW_gradients = malloc((DIMENSION - 1) * sizeof(Matrix));
    *db_gradients = malloc((DIMENSION - 1) * sizeof(Matrix));
    Matrix *dZ = minus(&A_list[DIMENSION - 1], y);
    for (int i = DIMENSION - 2; i >= 0; i--) {
        Matrix *tmp_dW = &(*dW_gradients)[DIMENSION - 2 - i];
        Matrix *tmp_tra_dW = transpose(&A_list[i]);
        Matrix *tmp_mul_dW = mul(dZ, tmp_tra_dW);
        tmp_dW->sizeX = tmp_mul_dW->sizeX;
        tmp_dW->sizeY = tmp_mul_dW->sizeY;
        tmp_dW->data = malloc(tmp_mul_dW->sizeX * tmp_mul_dW->sizeY * sizeof(double));
        for (int j = 0; j < tmp_mul_dW->sizeX * tmp_mul_dW->sizeY; j++)
            tmp_dW->data[j] = tmp_mul_dW->data[j];// / X_TRAIN_SIZE;

        Matrix *tmp_db = &(*db_gradients)[DIMENSION - 2 - i];
        Matrix *tmp_sum_db = columns_sum(dZ);
        tmp_db->sizeX = tmp_sum_db->sizeX;
        tmp_db->sizeY = tmp_sum_db->sizeY;
        tmp_db->data = malloc(tmp_sum_db->sizeX * tmp_sum_db->sizeY * sizeof(double));
        for (int j = 0; j < tmp_sum_db->sizeX * tmp_sum_db->sizeY; j++)
            tmp_db->data[j] = tmp_sum_db->data[j];// / X_TRAIN_SIZE;
        if (i > 0)
        {
            Matrix *tmp_t_dZ = transpose(&W_list[i]);
            Matrix *tmp_mul_dZ = mul(tmp_t_dZ, dZ);
            Matrix *tmp_1_dZ = mul_matrix(tmp_mul_dZ, &A_list[i]);
            Matrix *min_dZ = malloc(sizeof(Matrix));
            min_dZ->sizeX = A_list[i].sizeX;
            min_dZ->sizeY = A_list[i].sizeY;
            min_dZ->data = malloc(min_dZ->sizeX * min_dZ->sizeY * sizeof(Matrix));
            for (int j = 0; j < min_dZ->sizeX * min_dZ->sizeY; j++)
                min_dZ->data[j] = 1 - A_list[i].data[j];
            Matrix *mul_dZ = mul_matrix(tmp_1_dZ, min_dZ);

            dZ->sizeX = mul_dZ->sizeX;
            dZ->sizeY = mul_dZ->sizeY;
            free(dZ->data);
            dZ->data = malloc(mul_dZ->sizeX * mul_dZ->sizeY * sizeof(double));
            for (int j = 0; j < mul_dZ->sizeX * mul_dZ->sizeY; j++)
                dZ->data[j] = mul_dZ->data[j];

            free(tmp_t_dZ->data);
            free(tmp_t_dZ);
            free(tmp_mul_dZ->data);
            free(tmp_mul_dZ);
            free(tmp_1_dZ->data);
            free(tmp_1_dZ);
            free(min_dZ->data);
            free(min_dZ);
            free(mul_dZ->data);
            free(mul_dZ);
        }
        free(tmp_tra_dW->data);
        free(tmp_tra_dW);
        free(tmp_mul_dW->data);
        free(tmp_mul_dW);
        free(tmp_sum_db->data);
        free(tmp_sum_db);
    }
    free(dZ->data);
    free(dZ);
}


void update(Matrix *dW_gradients, Matrix *db_gradients, Matrix *W_list, Matrix *b_list)
{
    for (int i = 0; i < DIMENSION - 1; i++) {
        for (int j = 0; j < W_list[i].sizeX * W_list[i].sizeY; j++)
            W_list[i].data[j] = W_list[i].data[j] - LEARNING_RATE * dW_gradients[DIMENSION - 2 - i].data[j];
        for (int j = 0; j < b_list[i].sizeX * b_list[i].sizeY; j++)
            b_list[i].data[j] = b_list[i].data[j] - LEARNING_RATE * db_gradients[DIMENSION - 2 - i].data[j];
    }
}

Matrix* predict(Matrix *X, Matrix *W_list, Matrix *b_list, int print_check)
{
    Matrix *Acti;
    forward_propagation(X, W_list, b_list, &Acti);
    Matrix pre = Acti[DIMENSION - 1];

    Matrix *res = malloc(sizeof(Matrix));
    res->sizeX = pre.sizeX;
    res->sizeY = pre.sizeY;
    res->data = malloc(res->sizeX * res->sizeY * sizeof(double));

    if (print_check) {
        printf("Predict for (");
        for (int i = 0; i < X->sizeX * X->sizeY - 1; i++)
            printf("%0.f, ", X->data[i]);
        printf("%.0f): ", X->data[X->sizeX * X->sizeY]);
    }
    int max_pos = 0;
    double max_nbr = 0;
    for (int i = 0; i < pre.sizeX * pre.sizeY; i++)
    {
        if (pre.data[i] > max_nbr) {
            max_nbr = pre.data[i];
            max_pos = i;
        }
        if (print_check)
            printf("%0.10f ", pre.data[i]);
        res->data[i] = 0;
    }
    if (max_nbr > 0.5) res->data[max_pos] = 1;
    if (print_check)
        printf("\n");
    for (int i = 0; i < DIMENSION; i++)
        free(Acti[i].data);
    free(Acti);
    return res;
}

double accuracy(Matrix *X, Matrix *y, int test_size, Matrix *W_list, Matrix *b_list)
{
    double accuracy = 0;
    for (int i = 0; i < test_size; i++) {
        int add = 1;
        Matrix *prediction = predict(&X[i], W_list, b_list, 0);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (fabs(prediction->data[j] - y[i].data[j]) > 0.01) {    //because malloc not really equal to 0
                add = 0;
                break;
            }
        }
        accuracy += add;
        free(prediction->data);
        free(prediction);
    }
    return (accuracy / test_size) * 100;
}


double log_loss(Matrix *y, Matrix *A)
{
    double loss = 0;
    for (int i = 0; i < A->sizeX; i++) {
        loss += -y->data[i] * log(A->data[i]) - (1 - y->data[i]) * log(1 - A->data[i]);
    }
    return loss;
}

void neural_network(Matrix **X, Matrix **y, int hidden_layers[], Matrix **W_list, Matrix **b_list, int update_net)
{
    if (update_net == 0)
        init_network(hidden_layers, W_list, b_list);
    for (int i = 0; i < EPOCH; i++) {
        for (int j = 0; j < X_TRAIN_SIZE; j++)
        {
            Matrix *A_list;
            forward_propagation(&(*X)[j], *W_list, *b_list, &A_list);
//            printf("A_list\n");
//            for (int k = 0; k < DIMENSION; k++)
//                printMatrix(A_list[k]);
//            printf("\n\n");
            Matrix *dW_gradients;
            Matrix *db_gradients;
            back_propagation(&(*y)[j], *W_list, A_list, &dW_gradients, &db_gradients);
//        printf("dW_gradients\n");
//        for (int j = 0; j < DIMENSION - 1; j++)
//            printMatrix(dW_gradients[j]);
//        printf("\n");
//        printf("db_gradients\n");
//        for (int j = 0; j < DIMENSION - 1; j++)
//            printMatrix(db_gradients[j]);
//        printf("\n\n");
            update(dW_gradients, db_gradients, *W_list, *b_list);
//        printf("W_list\n");
//        for (int j = 0; j < DIMENSION - 1; j++)
//            printMatrix((*W_list)[j]);
//        printf("\n");
//        printf("b_list\n");
//        for (int j = 0; j < DIMENSION - 1; j++)
//            printMatrix((*b_list)[j]);
//        printf("\n\n");
//            printf("Epoch number %i\n", i);
//            printf("Log loss: %f\n", log_loss(y[j], &A_list[DIMENSION - 1]));
//            printf("Accuracy: %f\n", accuracy(X[j], y[j], *W_list, *b_list));

            for (size_t k = 0; k < DIMENSION - 1; k++) {
                free((A_list[k + 1]).data);
                free(dW_gradients[k].data);
                free(db_gradients[k].data);
            }
            free(A_list[0].data);
            free(A_list);
            free(dW_gradients);
            free(db_gradients);
        }
        printf("Epoch number: %i\n", i);
        printf("Accuracy: %f\n", accuracy(*X, *y, X_TRAIN_SIZE, *W_list, *b_list));
    }
    printf("Learning finished:\n\n");
//    Matrix *A_list;
//    forward_propagation(X, *W_list, *b_list, &A_list);
//    printf("Log loss: %f\n", log_loss(y, &A_list[DIMENSION - 1]));
//    printf("Accuracy: %.2f%%\n", accuracy(X, y, *W_list, *b_list) * 100);


//    for (size_t i = 0; i < DIMENSION - 1; i++)
//    {
//        printf("W_list:\n");
//        printMatrix((*W_list)[i]);
//    }
//    printf("\n\n\n\n\n\n\n\n\n\n");
//    for (size_t i = 0; i < DIMENSION - 1; i++)
//    {
//        printf("b_list:\n");
//        printMatrix((*b_list)[i]);
//    }
}


void save_network(Matrix *W_list, Matrix *b_list, char filename[], int epoch_nbr)
{
    FILE *file = fopen(filename, "wb");
    //write network spec
    fwrite(&epoch_nbr, sizeof(epoch_nbr), 1, file);
    int dimsize = DIMENSION;
    fwrite(&dimsize, sizeof(dimsize), 1, file);
    int train_size_x = X_TRAIN_SIZE;
    fwrite(&train_size_x, sizeof(train_size_x), 1, file);
    int train_size_y = Y_TRAIN_SIZE;
    fwrite(&train_size_y, sizeof(train_size_y), 1, file);

    for (int i = 0; i < DIMENSION - 1; i++) {
        fwrite(&(W_list[i].sizeX), sizeof(W_list[i].sizeX), 1, file);
        fwrite(&(W_list[i].sizeY), sizeof(W_list[i].sizeY), 1, file);
        for (int j = 0; j < W_list[i].sizeX * W_list[i].sizeY; j++)
            fwrite(&(W_list[i].data[j]), sizeof(W_list[i].data[j]), 1, file);
    }
    for (int i = 0; i < DIMENSION - 1; i++) {
        fwrite(&(b_list[i].sizeX), sizeof(b_list[i].sizeX), 1, file);
        fwrite(&(b_list[i].sizeY), sizeof(b_list[i].sizeY), 1, file);
        for (int j = 0; j < b_list[i].sizeX * b_list[i].sizeY; j++)
            fwrite(&(b_list[i].data[j]), sizeof(b_list[i].data[j]), 1, file);
    }

    fwrite(&W_list, sizeof(epoch_nbr), 1, file);
    fclose(file);
    printf("Network saved\n");
}

int load_network(Matrix **W_list, Matrix **b_list, char filename[])
{
    FILE *file = fopen(filename, "rb");
    printf("Network specification:\n");

    int epoch_nbr;
    fread(&epoch_nbr, sizeof(epoch_nbr), 1, file);
    printf("  -   %.0i epoch\n", epoch_nbr);
    int nbr;
    fread(&nbr, sizeof(nbr), 1, file);
    printf("  -   %.0i dimension\n", nbr);
    if (nbr != DIMENSION) {
        printf("Impossible to load the network because the specifications in the AI.h are not the same.\n");
        exit(EXIT_FAILURE);
    }
    fread(&nbr, sizeof(nbr), 1, file);
    printf("  -   %.0i X train size length\n", nbr);
    if (nbr != X_TRAIN_SIZE) {
        printf("Impossible to load the network because the specifications in the AI.h are not the same.\n");
        exit(EXIT_FAILURE);
    }
    fread(&nbr, sizeof(nbr), 1, file);
    if (nbr != Y_TRAIN_SIZE) {
        printf("Impossible to load the network because the specifications in the AI.h are not the same.\n");
        exit(EXIT_FAILURE);
    }
    printf("  -   %.0i Y train size length\n", nbr);

    *W_list = malloc((DIMENSION - 1) * sizeof(Matrix));
    *b_list = malloc((DIMENSION - 1) * sizeof(Matrix));
    for (int i = 0; i < DIMENSION - 1; i++) {
        Matrix *tmp_W = &(*W_list)[i];
        fread(&nbr, sizeof(nbr), 1, file);
        tmp_W->sizeX = nbr;
        fread(&nbr, sizeof(nbr), 1, file);
        tmp_W->sizeY = nbr;
        tmp_W->data = malloc(tmp_W->sizeX * tmp_W->sizeY * sizeof(double));
        for (int j = 0; j < tmp_W->sizeX * tmp_W->sizeY; j++) {
            double val;
            fread(&val, sizeof(val), 1, file);
            tmp_W->data[j] = val;
        }
    }
    for (int i = 0; i < DIMENSION - 1; i++) {
        Matrix *tmp_b = &(*b_list)[i];
        fread(&nbr, sizeof(nbr), 1, file);
        tmp_b->sizeX = nbr;
        fread(&nbr, sizeof(nbr), 1, file);
        tmp_b->sizeY = nbr;
        tmp_b->data = malloc(tmp_b->sizeX * tmp_b->sizeY * sizeof(double));
        for (int j = 0; j < tmp_b->sizeX * tmp_b->sizeY; j++) {
            double val;
            fread(&val, sizeof(val), 1, file);
            tmp_b->data[j] = val;
        }
    }

    fclose(file);
    printf("\nThe network has been loaded successfully.\n");
    return epoch_nbr;
}
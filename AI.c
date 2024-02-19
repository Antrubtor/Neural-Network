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
//    (&(*W_list)[0])->data[0] = 0.11;
//    (&(*W_list)[0])->data[1] = 0.21;
//    (&(*W_list)[0])->data[2] = 0.31;
//    (&(*W_list)[0])->data[3] = 0.41;
//    (&(*W_list)[0])->data[4] = 0.51;
//    (&(*W_list)[0])->data[5] = 0.61;
//    (&(*W_list)[0])->data[6] = 0.71;
//    (&(*W_list)[0])->data[7] = 0.81;
//
//    (&(*W_list)[1])->data[0] = 0.82;
//    (&(*W_list)[1])->data[1] = 0.72;
//    (&(*W_list)[1])->data[2] = 0.62;
//    (&(*W_list)[1])->data[3] = 0.52;
//
//
//    (&(*b_list)[0])->data[0] = -0.13;
//    (&(*b_list)[0])->data[1] = -0.23;
//    (&(*b_list)[0])->data[2] = -0.33;
//    (&(*b_list)[0])->data[3] = -0.43;
//
//    (&(*b_list)[1])->data[0] = 0.14;
}

void forward_propagation(Matrix *X, Matrix *W_list, Matrix *b_list, Matrix **A_list)
{
//    printf("X: \n");
//    printMatrix(*X);
    *A_list = malloc(DIMENSION * sizeof(Matrix));
    A_list[0]->sizeX = X->sizeX;
    A_list[0]->sizeY = X->sizeY;
    A_list[0]->data = malloc(X->sizeX * X->sizeY * sizeof(double));
    for (int i = 0; i < X->sizeX * X->sizeY; i++)
        A_list[0]->data[i] = X->data[i];

//    memcpy(A_list[0], X, sizeof(Matrix)); //copy X in the new list, create some bugs
//    memcpy(A_list[0]->data, X->data, X->sizeX * X->sizeY * sizeof(double));
    for (int i = 0; i < DIMENSION - 1; i++) {
        Matrix *Z = mul(&W_list[i], &(*A_list)[i]);
        Matrix *tmp_A = &(*A_list)[i + 1];
        tmp_A->sizeX = Z->sizeX;
        tmp_A->sizeY = Z->sizeY;
        tmp_A->data = malloc(Z->sizeX * Z->sizeY * sizeof(double));
//        printf("LAAAA:---------------\n");
//        printMatrix((*A_list)[i]);
//        printMatrix(W_list[i]);
//        printMatrix(*Z);
//        printf("---------------\n");
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
    for (int i = DIMENSION - 2; i >= 0; i--)
    {
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
    for (int i = 0; i < DIMENSION - 1; i++)
    {
        for (int j = 0; j < W_list[i].sizeX * W_list[i].sizeY; j++)
            W_list[i].data[j] = W_list[i].data[j] - LEARNING_RATE * dW_gradients[DIMENSION - 2 - i].data[j];
        for (int j = 0; j < b_list[i].sizeX * b_list[i].sizeY; j++)
            b_list[i].data[j] = b_list[i].data[j] - LEARNING_RATE * db_gradients[DIMENSION - 2 - i].data[j];
    }
}

double predict(Matrix *X, Matrix *W_list, Matrix *b_list)
{
    Matrix *Acti;
    forward_propagation(X, W_list, b_list, &Acti);
    double pre = Acti[DIMENSION - 1].data[0];

    for (int i = 0; i < DIMENSION; i++)
        free(Acti[i].data);
    free(Acti);
    return pre;
}

int predict_test(Matrix *X, double res, Matrix *W_list, Matrix *b_list)
{
    Matrix *Acti;
    forward_propagation(X, W_list, b_list, &Acti);
//    printf("Prediction:\n");
    int max = 0;
    double max_nbr = Acti[DIMENSION - 1].data[0];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (Acti[DIMENSION - 1].data[i] > max_nbr) {
            max_nbr = Acti[DIMENSION - 1].data[i];
            max = i;
        }
//        printf("%i: %f", i, Acti[DIMENSION - 1].data[i]);
    }
//    printf("\n");
    for (int i = 0; i < DIMENSION; i++)
        free(Acti[i].data);
    free(Acti);
//    printf("Max: %i / res = %f\n", max, res);
    if (max == res) return 1;
    return 0;
}

double accuracy(Matrix *X, Matrix *y, Matrix *W_list, Matrix *b_list)
{
    Matrix *Acti;
    forward_propagation(X, W_list, b_list, &Acti);

    Matrix *tmp_acti = &Acti[DIMENSION - 1];
    double res = 0;
    for (int i = 0; i < tmp_acti->sizeX; i++) {
        double nb = 0;
        if (tmp_acti->data[i] >= 0.5) nb = 1;
        if (nb == y->data[i])
            res ++;
    }
    for (int i = 0; i < DIMENSION; i++)
        free(Acti[i].data);
    free(Acti);
    return res / tmp_acti->sizeX;
}


double log_loss(Matrix *y, Matrix *A)
{
    double loss = 0;
    for (int i = 0; i < A->sizeX; i++) {
        loss += -y->data[i] * log(A->data[i]) - (1 - y->data[i]) * log(1 - A->data[i]);
    }
    return loss;
}

void neural_network(Matrix **X, Matrix **y, int hidden_layers[], Matrix **W_list, Matrix **b_list)
{
//    printMatrix(*X);
//    printMatrix(*y);
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
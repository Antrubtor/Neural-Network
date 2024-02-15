#include "AI.h"

double random_gaussian()
{
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
            tmp_W->data[j] = 1; /*random_gaussian();*/
        Matrix *tmp_b = &(*b_list)[i - 1];
        tmp_b->sizeX = 1;
        tmp_b->sizeY = dim[i];
        tmp_b->data = malloc(tmp_b->sizeY * sizeof(double));
        for (int j = 0; j < tmp_b->sizeY; j++)
            tmp_b->data[j] = 1; /*random_gaussian();*/
    }
}

void forward_propagation(Matrix *X, Matrix *W_list, Matrix *b_list, Matrix **A_list)
{
    *A_list = malloc(DIMENSION * sizeof(Matrix));
    memcpy(A_list[0], X, sizeof(Matrix)); //copy X in the new list
    memcpy(A_list[0]->data, X->data, X->sizeX * X->sizeY * sizeof(double));
    for (int i = 0; i < DIMENSION - 1; i++)
    {
        Matrix *Z = mul(&W_list[i], &(*A_list)[i]);
        Matrix *tmp_A = &(*A_list)[i + 1];
        tmp_A->sizeX = Z->sizeX;
        tmp_A->sizeY = Z->sizeY;
        tmp_A->data = malloc(Z->sizeX * Z->sizeY * sizeof(double));
        for (int j = 0; j < Z->sizeX * Z->sizeY; j++)
        {
            double tmp = Z->data[j] + b_list[i].data[j % b_list[i].sizeX];
            tmp_A->data[j] = 1 / (1 + exp(-(tmp)));
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
        printf("i: %i\n", i);
        Matrix *tmp_dW = &(*dW_gradients)[DIMENSION - 2 - i];
        Matrix *tmp_tra_dW = transpose(&A_list[i]);
        Matrix *tmp_mul_dW = mul(dZ, tmp_tra_dW);
        tmp_dW->sizeX = tmp_mul_dW->sizeX;
        tmp_dW->sizeY = tmp_mul_dW->sizeY;
        tmp_dW->data = malloc(tmp_mul_dW->sizeX * tmp_mul_dW->sizeY * sizeof(double));
        for (int j = 0; j < tmp_mul_dW->sizeX * tmp_mul_dW->sizeY; j++)
            tmp_dW->data[j] = tmp_mul_dW->data[j] / X_TRAIN_SIZE;

        Matrix *tmp_db = &(*db_gradients)[DIMENSION - 2 - i];
        Matrix *tmp_sum_db = columns_sum(dZ);
        tmp_db->sizeX = tmp_sum_db->sizeX;
        tmp_db->sizeY = tmp_sum_db->sizeY;
        tmp_db->data = malloc(tmp_sum_db->sizeX * tmp_sum_db->sizeY * sizeof(double));
        for (int j = 0; j < tmp_sum_db->sizeX * tmp_sum_db->sizeY; j++)
            tmp_db->data[j] = tmp_sum_db->data[j] / X_TRAIN_SIZE;
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


void update(Matrix *dW_gradients, Matrix *db_gradients, Matrix *W_list, Matrix *b_list, double learning_rate)
{

}

//void predict(double **X, double **W, double **b)
//{
//}
//
//double log_loss(double *A, double *y)
//{
//}
//
//void neural_network()
//{
//}
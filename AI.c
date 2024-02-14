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

void forward_propagation(Matrix *X, Matrix *W_list, Matrix *b_list, Matrix **A)
{
    *A = malloc(DIMENSION * sizeof(Matrix));
    memcpy(A[0], X, sizeof(Matrix)); //copy X in the new list
    memcpy(A[0]->data, X->data, X->sizeX * X->sizeY * sizeof(double));
    for (int i = 0; i < DIMENSION - 1; i++)
    {
        Matrix *Z_tmp = mul(&W_list[i], &(*A)[i]);
        Matrix *Z = add(Z_tmp, &b_list[i]);
        Matrix *tmp_A = &(*A)[i + 1];
        tmp_A->sizeX = Z->sizeX;
        tmp_A->sizeY = Z->sizeY;
        tmp_A->data = malloc(Z->sizeX * Z->sizeY * sizeof(double));
        for (int j = 0; j < Z->sizeX * Z->sizeY; j++)
            tmp_A->data[j] = 1 / (1 + exp(-(Z->data[j])));  //sigmoïd
        free(Z->data);
        free(Z_tmp->data);
        free(Z);
        free(Z_tmp);
    }
}
//
//void back_propagation(double **A, double **X, double *y, double **dW, double *db)
//{
//}
//
//
//void update(double **dW, double db, double **W, double b, double learning_rate)
//{
//}
//
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
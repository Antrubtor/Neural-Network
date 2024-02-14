#include "AI.h"

double random_gaussian()
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}


void init_network(int dim[], int dimsize, Matrix **W_list, Matrix **b_list)
{
    *W_list = malloc((dimsize - 1) * sizeof(Matrix));
    *b_list = malloc((dimsize - 1) * sizeof(Matrix));
    for (int i = 1; i < dimsize; i++)
    {
        Matrix *tmp_W = &(*W_list)[i - 1];
        tmp_W->sizeX = dim[i];
        tmp_W->sizeY = dim[i - 1];
        tmp_W->data = malloc(tmp_W->sizeX * tmp_W->sizeY * sizeof(double));
        for (int j = 0; j < tmp_W->sizeX * tmp_W->sizeY; j++)
            tmp_W->data[j] = 1; /*random_gaussian();*/
        Matrix *tmp_b = &(*b_list)[i - 1];
        tmp_b->sizeX = dim[1];
        tmp_b->sizeY = dim[i];
        tmp_b->data = malloc(tmp_b->sizeY * sizeof(double));
        for (int j = 0; j < tmp_b->sizeY; j++)
            tmp_b->data[j] = 1; /*random_gaussian();*/
    }
    printMatrix((*W_list)[0]);
    printMatrix((*W_list)[1]);
    printMatrix((*W_list)[2]);
    printMatrix((*W_list)[3]);
}

void forward_propagation(double **X, double **W, double b, double **A)
{




    *A = malloc(y_size_X * sizeof(double));
    mul(X, W, y_size_X, x_size_X, 1, A);
    for (size_t i = 0; i < y_size_X; i++)
        (*A)[i] = 1 / (1 + exp(-((*A)[i] + b)));
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
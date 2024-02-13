#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "AI.h"

#define x_size_X 2
#define y_size_X 4

double random_gaussian()
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void mul(double **m1, double **m2, size_t r1, size_t c1, size_t c2, double **r)
{
    for (size_t i = 0; i < r1; i++) {
        for (size_t j = 0; j < c2; j++) {
            double add = 0;
            for (size_t k = 0; k < c1; k++) {
                double nb1 = (*m1)[i * c1 + k];
                double nb2 = (*m2)[k * c2 + j];
                add += nb1 * nb2;
            }
            (*r)[i * c2 + j] = add;
        }
    }
}

void transpose(double **m, size_t rows, size_t cols, double **r)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            (*r)[j * rows + i] = (*m)[i * cols + j];
        }
    }
}


void init_network(double **W, double *b)
{
    *W = malloc(x_size_X * sizeof(double));
    for (int i = 0; i < x_size_X; i++)
    {
        (*W)[i] = random_gaussian();
    }
    *b = random_gaussian();
}

void model(double **A, double **X, double **W, double b)
{
    *A = malloc(y_size_X * sizeof(double));
    mul(X, W, y_size_X, x_size_X, 1, A);
    for (size_t i = 0; i < y_size_X; i++)
        (*A)[i] = 1 / (1 + exp(-((*A)[i] + b)));
}

double log_loss(double *A, double *y)
{
    double loss = 0;
    for (size_t i = 0; i < y_size_X; i++) {
        loss += -y[i] * log(A[i]) - (1 - y[i]) * log(1 - A[i]);
    }
    return loss;
}

void gradients(double **A, double **X, double *y, double **dW, double *db)
{
    // Calcul de dW
    double *A_minus_y = malloc(y_size_X * y_size_X * sizeof(double));
    for (size_t i = 0; i < y_size_X; i++) {
        for (size_t j = 0; j < y_size_X; j++) {
            A_minus_y[i * y_size_X + j] = (*A)[i] - y[j];
        }
    }

    double *X_transposed = malloc(x_size_X * y_size_X * sizeof(double));
    transpose(X, y_size_X, x_size_X, &X_transposed);

    *dW = malloc(x_size_X * y_size_X * sizeof(double));
    mul(&X_transposed, &A_minus_y, x_size_X, y_size_X, y_size_X, dW);
    for (size_t i = 0; i < y_size_X * y_size_X; i++) {
        (*dW)[i] = - (*dW)[i] / y_size_X;
    }

    // Calcul de db
    *db = 0;
    for (size_t i = 0; i < y_size_X * y_size_X; i++) {
        *db += A_minus_y[i];
    }
    *db /= y_size_X;
    free(A_minus_y);
    free(X_transposed);

}


void update(double **dW, double db, double **W, double b, double learning_rate)
{
    for (int i = 0; i < 8; i++)
    {
        printf("%f / ", (*W)[i]);
    }
}

int main()//(int argc, char *argv[])
{
    // srand(time(NULL));
    srand(4);


    double Xx[y_size_X][x_size_X] = {
        {1, 1},
        {0, 1},
        {1, 0},
        {0, 0}
    };

    double yy[y_size_X] = {
        1,
        0,
        0,
        1};

    // Juste mettre sous forme de malloc les tableaux précédents pour simplifier
    double *X = malloc(x_size_X * y_size_X * sizeof(double));
    for (size_t i = 0; i < y_size_X; i++) {
        for (size_t j = 0; j < x_size_X; j++) {
            X[i * x_size_X + j] = Xx[i][j];
        }
    }
    double *y = malloc(y_size_X * sizeof(double));
    for (size_t i = 0; i < y_size_X; i++) {
        y[i] = yy[i];
    }

    printf("X: ");
    for (size_t i = 0; i < y_size_X * x_size_X; i++) printf("%f, ", X[i]);
    printf("\n");
    printf("\ny: ");
    for (size_t i = 0; i < y_size_X; i++) printf("%f, ", y[i]);
    printf("\n");



    double *W;
    double b;
    init_network(&W, &b);

    printf("Matrice W: \n"); for (int i = 0; i < 8; i++) {printf("%f / ", W[i]);}printf("\n");


    /* TESTING PART */
    W[0] = -1.1;
    W[1] = -1.2;
    b = -1.3;
    // printf("W[0]= %f / W[1] = %f\n", W[0], W[1]);


    /* TESTING PART */
    double *A;
    model(&A, &X, &W, b);
    double val = log_loss(A, y);
    printf("\nVALEUR DU LOG LOSS: %f\n", val);

    double *dW;
    double db;
    gradients(&A, &X, y, &dW, &db);

    // Affichage de dW
    printf("Matrice dW:\n");
    for (size_t i = 0; i < 8; i++) {
        printf("%f ", dW[i]);
    }
    printf("\n");

    // Affichage de db
    printf("db: %f\n", db);
    update(&dW, db, &W, b, 0.1);

    free(A);
    free(y);
    free(X);
    free(W);
    free(dW);

    return 0;
}

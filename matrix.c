#include "AI.h"

Matrix* add(Matrix *matrix1, Matrix *matrix2)
{
    int rows = matrix1->sizeY;
    int cols = matrix1->sizeX;
    double *m1 = matrix1->data;
    double *m2 = matrix2->data;

    Matrix *matrixR = malloc(sizeof(Matrix));
    matrixR->sizeX = cols;
    matrixR->sizeY = rows;
    matrixR->data = malloc(rows * cols * sizeof(double));
    double *r = matrixR->data;

    for (int i = 0; i < rows * cols; i++) {
        r[i] = m1[i] + m2[i];
    }
    return matrixR;
}

Matrix* minus(Matrix *matrix1, Matrix *matrix2)
{
    int rows = matrix1->sizeY;
    int cols = matrix1->sizeX;
    double *m1 = matrix1->data;
    double *m2 = matrix2->data;

    Matrix *matrixR = malloc(sizeof(Matrix));
    matrixR->sizeX = cols;
    matrixR->sizeY = rows;
    matrixR->data = malloc(rows * cols * sizeof(double));
    double *r = matrixR->data;

    for (int i = 0; i < rows * cols; i++) {
        r[i] = m1[i] - m2[i];
    }
    return matrixR;
}

Matrix* add_num(Matrix *matrix, double n)
{
    int rows = matrix->sizeY;
    int cols = matrix->sizeX;
    double *m = matrix->data;

    Matrix *matrixR = malloc(sizeof(Matrix));
    matrixR->sizeX = rows;
    matrixR->sizeY = cols;
    matrixR->data = malloc(rows * cols * sizeof(double));
    double *r = matrixR->data;

    for (int i = 0; i < rows * cols; i++) {
        r[i] = m[i] + n;
    }
    return matrixR;
}

Matrix* columns_sum(Matrix *matrix)
{
    Matrix *matrixR = malloc(sizeof(Matrix));
    matrixR->sizeX = 1;
    matrixR->sizeY = matrix->sizeY;
    matrixR->data = malloc(matrixR->sizeY * sizeof(double));

    for (int i = 0; i < matrix->sizeY; i++) {
        double sum = 0;
        for (int j = 0; j < matrix->sizeX; j++) {
            sum += matrix->data[i * matrix->sizeX + j];
        }
        matrixR->data[i] = sum;
    }
    return matrixR;
}

Matrix* mul_num(Matrix *matrix, double n)
{
    int rows = matrix->sizeY;
    int cols = matrix->sizeX;
    double *m = matrix->data;

    Matrix *matrixR = malloc(sizeof(Matrix));
    matrixR->sizeX = rows;
    matrixR->sizeY = cols;
    matrixR->data = malloc(rows * cols * sizeof(double));
    double *r = matrixR->data;

    for (int i = 0; i < rows * cols; i++) {
        r[i] = m[i] * n;
    }
    return matrixR;
}

Matrix* mul(Matrix *matrix1, Matrix *matrix2)
{
    int r1 = matrix1->sizeY;
    int c1 = matrix1->sizeX;
    int c2 = matrix2->sizeX;
    double *m1 = matrix1->data;
    double *m2 = matrix2->data;

    Matrix *matrixR = malloc(sizeof(Matrix));
    matrixR->sizeX = c2;
    matrixR->sizeY = r1;
    matrixR->data = malloc(r1 * c2 * sizeof(double));
    double *r = matrixR->data;
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            double add = 0;
            for (int k = 0; k < c1; k++) {
                add += m1[i * c1 + k] * m2[k * c2 + j];
            }
            r[i * c2 + j] = add;
        }
    }
    return matrixR;
}

Matrix* mul_matrix(Matrix *matrix1, Matrix *matrix2)   //multiply every number from m1 to m2
{
    int rows = matrix1->sizeY;
    int cols = matrix1->sizeX;
    double *m1 = matrix1->data;
    double *m2 = matrix2->data;

    Matrix *matrixR = malloc(sizeof(Matrix));
    matrixR->sizeX = rows;
    matrixR->sizeY = cols;
    matrixR->data = malloc(rows * cols * sizeof(double));
    double *r = matrixR->data;

    for (int i = 0; i < rows * cols; i++) {
        r[i] = m1[i] * m2[i];
    }
    return matrixR;
}

Matrix* transpose(Matrix *matrix)
{
    int rows = matrix->sizeY;
    int cols = matrix->sizeX;
    double *m = matrix->data;

    Matrix *matrixR = malloc(sizeof(Matrix));
    matrixR->sizeX = rows;
    matrixR->sizeY = cols;
    matrixR->data = malloc(rows * cols * sizeof(double));
    double *r = matrixR->data;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            r[j * rows + i] = m[i * cols + j];
        }
    }
    return matrixR;
}

void printMatrix(Matrix mat)
{
    printf("Matrix size: %d x %d :\n", mat.sizeY, mat.sizeX);
    for (int i = 0; i < mat.sizeY; i++) {
        for (int j = 0; j < mat.sizeX; j++) {
            printf("%.6f\t", mat.data[i * mat.sizeX + j]);
        }
        printf("\n");
    }
}
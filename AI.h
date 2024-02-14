#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

//same matrix as collab
typedef struct {
    int sizeX; // Taille en X / Colonnes
    int sizeY; // Taille en Y / Lignes
    double *data; // Pointeur vers la matrice
} Matrix;

Matrix* add(Matrix *matrix1, Matrix *matrix2);
Matrix* add_num(Matrix *matrix, double n);
Matrix* mul_num(Matrix *matrix, double n);
Matrix* mul(Matrix *matrix1, Matrix *matrix2);
Matrix* transpose(Matrix *matrix);
void printMatrix(Matrix mat);


//TODO: a edit
#define INPUT_SIZE 2
#define OUTPUT_SIZE 1
#define X_TRAIN_SIZE 4 //nombre d'entrées différentes
#define Y_TRAIN_SIZE 2  //ombien d'entrées pour 1 valeur (1, 1) par exemple
#define DIMENSION 5

double random_gaussian();
void init_network(int* dim, Matrix **W_list, Matrix **b_list);
void forward_propagation(Matrix *X, Matrix *W_list, Matrix *b_list, Matrix **A);
void back_propagation(double **A, double **X, double *y, double **dW, double *db);
void update(double **dW, double db, double **W, double b, double learning_rate);
void predict(double **X, double **W, double **b);
double log_loss(double *A, double *y);
void neural_network();
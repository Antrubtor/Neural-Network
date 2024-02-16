#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

//TODO: to edit
#define EPOCH 5000      // number of training epochs
#define DIMENSION 3     // depth of neural network, also modify hidden_layers in main.c
#define X_TRAIN_SIZE 4  // number of different entries
#define Y_TRAIN_SIZE 2  // how many entries for 1 value (1, 1) for XOR for example
#define OUTPUT_SIZE 1   // number of output neurons


typedef struct {
    int sizeX;      // Size in X / Columns
    int sizeY;      // Y Size / Lines
    double *data;   // Pointer to matrix data
} Matrix;

// AI.c
double random_gaussian();
void init_network(int* dim, Matrix **W_list, Matrix **b_list);
void forward_propagation(Matrix *X, Matrix *W_list, Matrix *b_list, Matrix **A);
void back_propagation(Matrix *y, Matrix *W_list, Matrix *A_list, Matrix **dW_list, Matrix **db_list);
void update(Matrix *dW_gradients, Matrix *db_gradients, Matrix *W_list, Matrix *b_list, double learning_rate);
double predict(Matrix *X, Matrix *W_list, Matrix *b_list);
double log_loss(Matrix *y, Matrix *A);
void neural_network(Matrix *X, Matrix *y, int hidden_layers[], double learning_rate, Matrix **W_list, Matrix **b_list);

// matrix.c
Matrix* add(Matrix *matrix1, Matrix *matrix2);
Matrix* minus(Matrix *matrix1, Matrix *matrix2);
Matrix* add_num(Matrix *matrix, double n);
Matrix* columns_sum(Matrix *matrix);
Matrix* mul_num(Matrix *matrix, double n);
Matrix* mul(Matrix *matrix1, Matrix *matrix2);
Matrix* mul_matrix(Matrix *matrix1, Matrix *matrix2);
Matrix* transpose(Matrix *matrix);
void printMatrix(Matrix mat);

//loadMNIST.c
void load_minst();
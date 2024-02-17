#pragma once

//TODO: to edit
#define EPOCH 1000          // number of training epochs
#define DIMENSION 5         // depth of neural network, also modify hidden_layers in main.c
#define LEARNING_RATE 0.1   // neural network learning rate
#define X_TRAIN_SIZE 1000      // number of different entries
#define Y_TRAIN_SIZE 784      // how many entries for 1 value (1, 1) for XOR for example
#define OUTPUT_SIZE 1       // number of output neurons

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

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
void update(Matrix *dW_gradients, Matrix *db_gradients, Matrix *W_list, Matrix *b_list);
double predict(Matrix *X, Matrix *W_list, Matrix *b_list);
double log_loss(Matrix *y, Matrix *A);
void neural_network(Matrix *X, Matrix *y, int hidden_layers[], Matrix **W_list, Matrix **b_list);

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
void load_mnist(char *filename_images, char *filename_labels, Matrix *images, Matrix *labels, int image_number);
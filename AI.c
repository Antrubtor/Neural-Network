#include "AI.h"


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
//        double lower = -(1.0 / sqrt(dim[i - 1]));  //Xavier Weight Initialization
//        double upper = (1.0 / sqrt(dim[i - 1]));
        double lower = -(sqrt(6.0) / sqrt(dim[i - 1] + dim[i])); //Normalized Xavier Weight Initialization
        double upper = (sqrt(6.0) / sqrt(dim[i - 1] + dim[i]));
//        for (int j = 0; j < tmp_W->sizeX * tmp_W->sizeY; j++)
//            /*tmp_W->data[j] = lower + ((double)rand() / RAND_MAX) * (upper - lower);*/ tmp_W->data[j] = ((double)rand() / RAND_MAX) * 2 - 1; //better for XOR
        Matrix *tmp_b = &(*b_list)[i - 1];
        tmp_b->sizeX = 1;
        tmp_b->sizeY = dim[i];
        tmp_b->data = malloc(tmp_b->sizeY * sizeof(double));
        for (int j = 0; j < tmp_b->sizeY; j++)
            tmp_b->data[j] = 0.001;
    }
    (&(*W_list)[0])->data[0] = 0.11;
    (&(*W_list)[0])->data[1] = 0.21;
    (&(*W_list)[0])->data[2] = 0.31;
    (&(*W_list)[0])->data[3] = 0.41;
    (&(*W_list)[0])->data[4] = 0.51;
    (&(*W_list)[0])->data[5] = 0.61;
    (&(*W_list)[0])->data[6] = 0.71;
    (&(*W_list)[0])->data[7] = 0.81;

    (&(*W_list)[1])->data[0] = 0.82;
    (&(*W_list)[1])->data[1] = 0.72;
    (&(*W_list)[1])->data[2] = 0.62;
    (&(*W_list)[1])->data[3] = 0.52;


    (&(*b_list)[0])->data[0] = -0.13;
    (&(*b_list)[0])->data[1] = -0.23;
    (&(*b_list)[0])->data[2] = -0.33;
    (&(*b_list)[0])->data[3] = -0.43;

    (&(*b_list)[1])->data[0] = 0.14;
}

void forward_propagation(Matrix *X, Matrix *W_list, Matrix *b_list, Matrix **A_list, int up)
{
    if (up == 1) {
        *A_list = malloc(DIMENSION * sizeof(Matrix));
        A_list[0]->sizeX = X->sizeX;
        A_list[0]->sizeY = X->sizeY;
        A_list[0]->data = malloc(X->sizeX * X->sizeY * sizeof(double));
    }
    for (int i = 0; i < X->sizeX * X->sizeY; i++)
        A_list[0]->data[i] = X->data[i];
    for (int i = 0; i < DIMENSION - 1; i++) {
        Matrix *Z = mul(&W_list[i], &(*A_list)[i]);
        Matrix *A = &(*A_list)[i + 1];
        if (up == 1) {
            A->sizeX = Z->sizeX;
            A->sizeY = Z->sizeY;
            A->data = malloc(Z->sizeX * Z->sizeY * sizeof(double));
        }
        for (int j = 0; j < Z->sizeY ; j++) {
            for (int k = 0; k < Z->sizeX; k++) {
                double tmp = Z->data[j * Z->sizeX + k] + b_list[i].data[j];
                A->data[j * Z->sizeX + k] = 1 / (1 + exp(-(tmp)));
            }
        }
        free(Z->data);
        free(Z);
    }
}

void back_propagation(Matrix *y, Matrix *W_list, Matrix *A_list, Matrix **dW_gradients, Matrix **db_gradients, int up)
{
    if (up == 1) {
        *dW_gradients = malloc((DIMENSION - 1) * sizeof(Matrix));
        *db_gradients = malloc((DIMENSION - 1) * sizeof(Matrix));
    }
    Matrix *dZ = minus(&A_list[DIMENSION - 1], y);
    for (int i = DIMENSION - 2; i >= 0; i--) {
        Matrix *dW = &(*dW_gradients)[DIMENSION - 2 - i];
        //dot(dZ, A[i].T)
        int r1 = dZ->sizeY;
        int c1 = dZ->sizeX;
        int c2 = A_list[i].sizeY;
        double *m1 = dZ->data;
        double *m2 = A_list[i].data;
        if (up == 1) {
            dW->sizeX = c2;
            dW->sizeY = r1;
            dW->data = malloc(r1 * c2 * sizeof(double));
        }
        for (int l = 0; l < r1; l++) {
            for (int j = 0; j < c2; j++) {
                double add = 0;
                for (int k = 0; k < c1; k++) {
                    add += m1[l * c1 + k] * m2[j * c1 + k];
                }
                dW->data[l * c2 + j] = add; // / X_TRAIN_SIZE;
            }
        }

        Matrix *db = &(*db_gradients)[DIMENSION - 2 - i];
        //sum(dZ)
        if (up == 1) {
            db->sizeX = 1;
            db->sizeY = dZ->sizeY;
            db->data = malloc(db->sizeY * sizeof(double));
        }
        for (int l = 0; l < dZ->sizeY; l++) {
            double sum = 0;
            for (int j = 0; j < dZ->sizeX; j++) {
                sum += dZ->data[l * dZ->sizeX + j];
            }
            db->data[l] = sum; // / X_TRAIN_SIZE;
        }
        if (i > 0)
        {
            //dot(W[i].T, dZ)
            r1 = W_list[i].sizeX;
            c1 = W_list[i].sizeY;
            c2 = dZ->sizeX;
            m1 = W_list[i].data;
            m2 = dZ->data;

            Matrix *tmp_mul_dZ = malloc(sizeof(Matrix));
            tmp_mul_dZ->sizeX = c2;
            tmp_mul_dZ->sizeY = r1;
            tmp_mul_dZ->data = malloc(c2 * r1 * sizeof(double));

            for (int l = 0; l < r1; l++) {
                for (int j = 0; j < c2; j++) {
                    double add = 0;
                    for (int k = 0; k < c1; k++) {
                        add += m1[k * r1 + l] * m2[k * c2 + j];
                    }
                    tmp_mul_dZ->data[l * c2 + j] = add;
                }
            }
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
            free(tmp_mul_dZ->data);
            free(tmp_mul_dZ);
            free(tmp_1_dZ->data);
            free(tmp_1_dZ);
            free(min_dZ->data);
            free(min_dZ);
            free(mul_dZ->data);
            free(mul_dZ);
        }
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
    forward_propagation(X, W_list, b_list, &Acti, 1);
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
    res->data[max_pos] = 1;
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
    int update_status = 1;
    Matrix *A_list;
    Matrix *dW_gradients;
    Matrix *db_gradients;
    for (int i = 0; i < EPOCH; i++) {
        for (int j = 0; j < X_TRAIN_SIZE; j++)
        {
            forward_propagation(&(*X)[j], *W_list, *b_list, &A_list, update_status);
            back_propagation(&(*y)[j], *W_list, A_list, &dW_gradients, &db_gradients, update_status);
            update(dW_gradients, db_gradients, *W_list, *b_list);
            update_status = 0;
        }
        if (EPOCH < 10 || (EPOCH >= 10 && i % (EPOCH / 10) == 0)) {
            printf("Epoch number: %i / ", i);
            printf("Accuracy: %f / ", accuracy(*X, *y, X_TRAIN_SIZE, *W_list, *b_list));
            printf("Log loss: %f", log_loss(y[0], &A_list[DIMENSION - 1]));
            printf("\n");
//            printf("Epoch number: %i / Accuracy: %f / Log loss: %f\n", i, accuracy(*X, *y, X_TRAIN_SIZE, *W_list, *b_list), log_loss(y[0], &A_list[DIMENSION - 1]));
        }
    }

    // free all
    for (size_t k = 0; k < DIMENSION - 1; k++) {
        free((A_list[k + 1]).data);
        free(dW_gradients[k].data);
        free(db_gradients[k].data);
    }
    free(A_list[0].data);
    free(A_list);
    free(dW_gradients);
    free(db_gradients);

    printf("Learning finished !\n");
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
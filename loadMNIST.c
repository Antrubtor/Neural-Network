#include "AI.h"

void load_mnist(char *filename_images, char *filename_labels, Matrix *images, Matrix *labels, int image_number)
{
    images->sizeX = 28 * 28;
    images->sizeY = image_number;
    images->data = malloc(images->sizeX * images->sizeY * sizeof(double));
    FILE *file_images = fopen(filename_images, "rb");
    if (file_images == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    fseek(file_images, 16, SEEK_SET);
    for (int i = 0; i < image_number; i++) {
        for (int j = 0; j < 28 * 28; j++) {
            uint8_t nbr;
            fread(&nbr, sizeof(nbr), 1, file_images);
//            if (nbr > 128)
//                printf("nbr: %u\n", nbr);
            if (nbr > 128) nbr = 1;
            else nbr = 0;
            images->data[i * 28 * 28 + j] = nbr;
//            printf("test: %f\n", images->data[i * 28 * 28 + j]);
        }
    }


    labels->sizeX = image_number;
    labels->sizeY = 1;
    labels->data = malloc(labels->sizeX * labels->sizeY * sizeof(double));
    FILE *file_labels = fopen(filename_labels, "rb");
    if (file_labels == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    fseek(file_labels, 8, SEEK_SET);
    for (int i = 0; i < image_number; i++) {
        uint8_t nbr;
        fread(&nbr, sizeof(nbr), 1, file_labels);
        labels->data[i] = nbr;
//        printf("nbr: %u\n", nbr);
    }
}
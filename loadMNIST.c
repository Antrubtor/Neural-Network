#include "AI.h"
//#include <SDL2/SDL.h>
#include <unistd.h>
void load_mnist(char *filename_images, char *filename_labels, Matrix *images, Matrix *labels, int image_number)
{
    FILE *file_images = fopen(filename_images, "rb");
    fseek(file_images, 16, SEEK_SET);
    for (int i = 0; i < image_number; i++) {
        images[i].sizeX = 1;
        images[i].sizeY = IMAGE_SIZE_FULL;
        images[i].data = malloc(images->sizeX * images->sizeY * sizeof(double));
        for (int j = 0; j < IMAGE_SIZE_FULL; j++) {
            uint8_t nbr;
            fread(&nbr, sizeof(nbr), 1, file_images);
            if (nbr > 128) nbr = 1;     //to binarize
            else nbr = 0;
//            images->data[i * IMAGE_SIZE_FULL + j] = nbr;
            images[i].data[j] = nbr;
        }
    }
    printf("ici\n");

    FILE *file_labels = fopen(filename_labels, "rb");
    fseek(file_labels, 8, SEEK_SET);
    for (int i = 0; i < image_number; i++) {
        labels[i].sizeX = 1;
        labels[i].sizeY = 10;
        labels[i].data = malloc(labels->sizeX * labels->sizeY * sizeof(double));
        uint8_t nbr;
        fread(&nbr, sizeof(nbr), 1, file_labels);
        labels[i].data[nbr] = 1;
//        printf("nbr: %u\n", nbr);
    }
}


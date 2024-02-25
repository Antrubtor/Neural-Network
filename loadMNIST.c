#include "AI.h"

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
            images[i].data[j] = nbr;
        }
    }

    FILE *file_labels = fopen(filename_labels, "rb");
    fseek(file_labels, 8, SEEK_SET);
    for (int i = 0; i < image_number; i++) {
        labels[i].sizeX = 1;
        labels[i].sizeY = 10;
        labels[i].data = malloc(labels->sizeX * labels->sizeY * sizeof(double));
        uint8_t nbr;
        fread(&nbr, sizeof(nbr), 1, file_labels);
        labels[i].data[nbr] = 1;
    }
    printf("MNIST loaded\n");
}


Matrix* image_to_matrix(char filename[])
{
    SDL_Surface* nbr_image = IMG_Load(filename);
    SDL_Surface* nbr_image_resized = SDL_CreateRGBSurface(0, 28, 28, 32, 0, 0, 0, 0);
    SDL_BlitScaled(nbr_image, NULL, nbr_image_resized, NULL);


    Matrix *res = malloc(sizeof(Matrix));
    res->sizeX = 1;
    res->sizeY = IMAGE_SIZE_FULL;
    res->data = malloc(784 * sizeof(double));
    Uint32* pixels = nbr_image_resized->pixels;
    Uint8 r, g, b;
    for (int i = 0; i < 784; i++) {
        Uint32 pixel = pixels[i];
        SDL_GetRGB(pixel, nbr_image_resized->format, &r, &g, &b);
        if ((r + g + b) / 3 > 128) res->data[i] = 0;
        else res->data[i] = 1;
    }
    SDL_FreeSurface(nbr_image);
    SDL_FreeSurface(nbr_image_resized);
    SDL_Quit();
    return res;
}
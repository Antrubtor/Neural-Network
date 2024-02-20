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

//    SDL_Surface *image = IMG_Load(filename);
    SDL_Surface *image = SDL_LoadBMP(filename);
    SDL_Surface *resized = SDL_CreateRGBSurface(0, 28, 28, image->format->BitsPerPixel,
                                                image->format->Rmask, image->format->Gmask,
                                                image->format->Bmask, image->format->Amask);
    SDL_BlitScaled(image, NULL, resized, NULL);
    SDL_LockSurface(resized);

    Uint32* pixels = resized->pixels;
    Matrix *res = malloc(sizeof(Matrix));
    res->sizeX = 28;
    res->sizeY = 28;
    res->data = malloc(784 * sizeof(double));
    Uint8 r, g, b;
    for (int i = 0; i < 784; i++) {
        Uint32 pixel = pixels[i];
        SDL_GetRGB(pixel, resized->format, &r, &g, &b);
        printf("r: %u / g: %u / b: %u\n", r, g, b);
        if ((r + g + b) / 3 > 128) res->data[i] = 1;
        else res->data[i] = 0;
    }
    SDL_UnlockSurface(resized);
    IMG_SavePNG(resized, "test.png");
    SDL_FreeSurface(image);
    SDL_FreeSurface(resized);
    return res;
}
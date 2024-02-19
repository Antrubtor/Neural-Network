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




void matrixToImage(Matrix *mat, const char *filename) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow("Matrix Image",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          mat->sizeX, mat->sizeY,
                                          SDL_WINDOW_SHOWN);
    if (window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == NULL) {
        printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

    SDL_Texture *texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_RGBA8888,
                                             SDL_TEXTUREACCESS_TARGET,
                                             mat->sizeX, mat->sizeY);
    if (texture == NULL) {
        printf("Texture could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

    SDL_SetRenderTarget(renderer, texture);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);

    for (int y = 0; y < mat->sizeY; y++) {
        for (int x = 0; x < mat->sizeX; x++) {
            double value = mat->data[y * mat->sizeX + x];
            if (value > 0.6) {
                SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            } else {
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            }
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }

    SDL_SetRenderTarget(renderer, NULL);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);

    SDL_Surface *surface = SDL_CreateRGBSurface(0, mat->sizeX, mat->sizeY, 32,
                                                0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000);
    if (surface == NULL) {
        printf("Surface could not be created! SDL_Error: %s\n", SDL_GetError());
    } else {
        SDL_RenderReadPixels(renderer, NULL, surface->format->format, surface->pixels, surface->pitch);
        SDL_SaveBMP(surface, filename);
        SDL_FreeSurface(surface);
    }
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}
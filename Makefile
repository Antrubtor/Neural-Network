CC = gcc
CFLAGS = -Wall -Wextra -lm `pkg-config --cflags sdl2 SDL2_image`
LDFLAGS = -lm -fsanitize=address
LDLIBS = `pkg-config --libs sdl2 SDL2_image`

SRC = main.c AI.c matrix.c loadMNIST.c
OBJ = $(SRC:.c=.o)

all: main

main: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: clean

clean:
	$(RM) main
	$(RM) $(OBJ)
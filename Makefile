CC = gcc
CFLAGS = -Wall -Wextra -lm `pkg-config --cflags sdl2 SDL2_image`
LDFLAGS = -lm ##-fsanitize=address
LDLIBS = `pkg-config --libs sdl2 SDL2_image`

SRC = main.c AI.c matrix.c loadMNIST.c
OBJ = $(SRC:.c=.o)

all: main

main: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)
	# ./main XOR      // Trains the network for XOR learning
	# ./main UXOR     // Retrains the network for XOR learning
	# ./main MNIST    // Retrains the network from scratch for image recognition using the MNIST dataset
	# ./main UMNIST   // Retrains the network for image recognition using the MNIST dataset
	# ./main TXOR     // Tests the trained neural network with XOR values
	# ./main TMNIST1  // Tests the trained neural network with the MNIST training dataset
	# ./main TMNIST2  // Tests the trained neural network with a custom image located in the img/test_img.png directory
	# ./main UMNIST TMNIST2	// You can directly test the network after training by appending the testing command

.PHONY: clean

clean:
	$(RM) main
	$(RM) $(OBJ)
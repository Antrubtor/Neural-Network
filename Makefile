CC = gcc
CFLAGS = -Wall -Wextra ##-o0 -ggdb
LDFLAGS = -lm	##-fsanitize=address
LDLIBS =

SRC = main.c AI.c matrix.c loadMNIST.c
OBJ = $(SRC:.c=.o)

all: main

main: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: clean

clean:
	$(RM) main
	$(RM) $(OBJ)
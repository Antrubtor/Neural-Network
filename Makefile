CC = gcc
CFLAGS = -Wall -Wextra -o0 -ggdb
LDFLAGS = -fsanitize=address -lm
LDLIBS =

SRC = main.c AI.c matrix.c
OBJ = $(SRC:.c=.o)

all: main

main: $(OBJ)

.PHONY: clean

clean:
	$(RM) main
	$(RM) $(OBJ)

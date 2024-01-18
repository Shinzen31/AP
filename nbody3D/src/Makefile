# Default compiler and flags
CC ?= gcc
CFLAGS ?= -march=native -g3 -fopenmp
OFLAGS ?= -O0 -fopt-info-all=nbody.gcc.optrpt

all: nbody3D

nbody3D: nbody.c
	$(CC) $(CFLAGS) $(OFLAGS) $< -o $@ -lm

clean:
	rm -Rf *~ nbody3D *.optrpt

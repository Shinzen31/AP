# Default compiler and flags
CC ?= gcc
CFLAGS ?= -march=native -g3 -fopenmp
LFLAGS ?= -lm
OFLAGS ?= -O0 -fopt-info-all=nbody.gcc.optrpt

# Check if the compiler is clang, if so, add -lomp to LFLAGS
ifeq ($(CC),clang)
    LFLAGS += -lomp
endif

all: nbody3D

nbody3D: nbody.c
	$(CC) $(CFLAGS) $(OFLAGS) $< -o $@ $(LFLAGS)

clean:
	rm -Rf *~ nbody3D *.optrpt

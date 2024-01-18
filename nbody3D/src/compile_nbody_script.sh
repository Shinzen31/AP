
#!/bin/bash

# Compilers to use
COMPILERS=("gcc" "clang")

# Optimization levels
OPTIMIZATIONS=("O2" "O3" "Ofast")

# Loop through each compiler
for COMPILER in "${COMPILERS[@]}"; do
    # Loop through each optimization level
    for OPTIMIZATION in "${OPTIMIZATIONS[@]}"; do
        # Set the compiler and optimization level in the Makefile
        make CC=$COMPILER OFLAGS=-$OPTIMIZATION

        # Execute the program and redirect output to a .dat file
        ./nbody3D > "${COMPILER}_${OPTIMIZATION}.dat"

        # Clean up (optional)
        make clean
    done
done

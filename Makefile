# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

.PHONY: all
all: qwen35.c
	$(CC) -o qwen35 qwen35.c -lm

.PHONY: debug
debug: qwen35.c
	$(CC) -g -o qwen35 qwen35.c -lm

# -Ofast enables all -O3 optimizations plus -ffast-math etc.
# ISA tuned to the build CPU (auto-vectorization/SIMD)
.PHONY: fast
fast: qwen35.c
	$(CC) -Ofast -march=native -o qwen35 qwen35.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./qwen35 model.bin
.PHONY: omp
omp: qwen35.c
	$(CC) -Ofast -fopenmp -march=native -o qwen35 qwen35.c -lm

# compiles with gnu11 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: gnu
gnu: qwen35.c
	$(CC) -Ofast -std=gnu11 -o qwen35 qwen35.c -lm

.PHONY: clean
clean:
	rm -f qwen35

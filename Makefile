# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

SRCS = qwen35.c csafetensors.c json.c
OBJS = $(SRCS:.c=.o)

.PHONY: all
all: $(SRCS)
	$(CC) -o qwen35 $(SRCS) -lm

.PHONY: debug
debug: $(SRCS)
	$(CC) -g -o qwen35 $(SRCS) -lm

# -Ofast enables all -O3 optimizations plus -ffast-math etc.
# ISA tuned to the build CPU (auto-vectorization/SIMD)
.PHONY: fast
fast: $(SRCS)
	$(CC) -Ofast -march=native -o qwen35 $(SRCS) -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./qwen35 model.bin
.PHONY: omp
omp: $(SRCS)
	$(CC) -Ofast -fopenmp -march=native -o qwen35 $(SRCS) -lm

# compiles with gnu11 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: gnu
gnu: $(SRCS)
	$(CC) -Ofast -std=gnu11 -o qwen35 $(SRCS) -lm

.PHONY: clean
clean:
	rm -f qwen35 $(OBJS)

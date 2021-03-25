NVCC_FLAGS =-Xptxas -allow-expensive-optimizations=true,-O3,-v \
	-Xcompiler "-Ofast -fopenmp -Wall"

all: main_exec
	./main_exec


main_exec: main.cu
	nvcc $(NVCC_FLAGS) main.cu -o main_exec
 
clean:
	rm main_exec

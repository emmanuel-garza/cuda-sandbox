NVCC_FLAGS =-Xptxas -allow-expensive-optimizations=true,-O3,-v \
	-Xcompiler "-Ofast -fopenmp -Wall"

all: main.exe
	export OMP_NUM_THREADS=6; \
	./fd-2d

# main.exe: kernel.cu
# -g for debug

main.exe: fd-2d.cu
	nvcc $(NVCC_FLAGS) fd-2d.cu -o fd-2d
 
clean:
	rm fd-2d

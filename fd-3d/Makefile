NVCC_FLAGS =-Xptxas -allow-expensive-optimizations=true,-O3,-v \
	-Xcompiler "-Ofast -fopenmp -Wall"

all: fd-3d
	export OMP_NUM_THREADS=6; \
	nvprof fd-3d

fd-3d: fd-3d.cu
	nvcc $(NVCC_FLAGS) fd-3d.cu -o fd-3d
 
clean:
	rm fd-3d

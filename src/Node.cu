#include "Node.cuh"

#include <cuda_runtime.h>
#include "Common.cuh"

__host__ Node::Node() : mass(0), particleDensity(0), gx(0), gy(0), u(0), v(0), u2(0), v2(0), ax(0), ay(0), active(false) {
	const int size = numMaterials * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&cgx, size));
	gpuErrchk(cudaMalloc((void**)&cgy, size));
	gpuErrchk(cudaMemset(cgx, 0,  numMaterials * sizeof(float)))
	gpuErrchk(cudaMemset(cgy, 0,  numMaterials * sizeof(float)));
}

__host__ Node::~Node()
{
	cudaFree(cgx);
	cudaFree(cgy);
}

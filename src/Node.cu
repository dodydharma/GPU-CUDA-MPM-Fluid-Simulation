#include "Node.cuh"

#include <cuda_runtime.h>
#include "Common.cuh"

__host__ Node::Node() : mass(0), particleDensity(0), gx(0), gy(0), u(0), v(0), u2(0), v2(0), ax(0), ay(0), active(false) {
	
}

__host__ Node::~Node()
{

}

__host__ void Node::initAttArrays(float* cgx)
{
	const int size = numMaterials * sizeof(float) * 2;
	this->cgx = cgx;
	this->cgy = cgx + numMaterials;
	gpuErrchk(cudaMemset(cgx, 0, size))
}
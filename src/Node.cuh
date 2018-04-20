#ifndef NODE_H
#define NODE_H

#define numMaterials 4

#include "Common.cuh"

struct Node {
	float mass, particleDensity, gx, gy, u, v, u2, v2, ax, ay;
	float cgx[numMaterials];
	float cgy[numMaterials];
	bool active;
	__host__ __device__ Node() : mass(0), particleDensity(0), gx(0), gy(0), u(0), v(0), u2(0), v2(0), ax(0), ay(0), active(false) {

	}

	__host__ __device__ ~Node() {

	}

	__host__ __device__ Node& operator=(const Node& other)
	{
		if (this != &other) {
			this->mass = other.mass;
			this->particleDensity = other.particleDensity;
			this->gx = other.gx;
			this->gy = other.gy;
			this->u = other.u;
			this->v = other.v;
			this->u2 = other.u2;
			this->v2 = other.v2;
			this->ax = other.ax;
			this->ay = other.ay;
			for (int i = 0; i < numMaterials; i++) {
				this->cgx[i] = other.cgx[i];
				this->cgy[i] = other.cgy[i];
			}
			this->active = other.active;
		}
		return *this;
	}

	__host__ __device__ void initAttArrays(float* cgx) {

		const int size = numMaterials * sizeof(float) * 2;
		//this->cgx = cgx;
		//this->cgy = cgx + numMaterials;
#ifdef __CUDA_ARCH__
		memset(cgx, 0, size);
#else
		gpuErrchk(cudaMemset(cgx, 0, size));
#endif	
	}


};

#endif
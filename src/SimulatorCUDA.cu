//#include <stdio.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

#include "SimulatorCUDA.cuh"

#include <math.h>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include "Common.cuh"

using namespace std;
using namespace cinder;

__device__ float uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v)
{
	float dx = x00 - x01;
	float dy = y00 - y10;
	float a = p01 - p00;
	float b = p11 - p10 - a;
	float c = p10 - p00;
	float d = y11 - y01;
	return ((((d - 2 * b - dy) * u - 2 * a + y00 + y01) * v +
		((3 * b + 2 * dy - d) * u + 3 * a - 2 * y00 - y01)) * v +
		((((2 * c - x00 - x10) * u + (3 * b + 2 * dx + x10 - x11)) * u - b - dy - dx) * u + y00)) * v +
		(((x11 - 2 * (p11 - p01 + c) + x10 + x00 + x01) * u +
		(3 * c - 2 * x00 - x10)) * u +
			x00) * u + p00;
}

float SimulatorCUDA::uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v)
{
	float dx = x00 - x01;
	float dy = y00 - y10;
	float a = p01 - p00;
	float b = p11 - p10 - a;
	float c = p10 - p00;
	float d = y11 - y01;
	return ((((d - 2 * b - dy) * u - 2 * a + y00 + y01) * v +
		((3 * b + 2 * dy - d) * u + 3 * a - 2 * y00 - y01)) * v +
		((((2 * c - x00 - x10) * u + (3 * b + 2 * dx + x10 - x11)) * u - b - dy - dx) * u + y00)) * v +
		(((x11 - 2 * (p11 - p01 + c) + x10 + x00 + x01) * u +
		(3 * c - 2 * x00 - x10)) * u +
			x00) * u + p00;
}

SimulatorCUDA::SimulatorCUDA() :scale(1.0f) {
	//default
	materials[0].materialIndex = 0;
	materials[0].mass = 1.0f;
	materials[0].viscosity = 0.04f;

	materials[1].materialIndex = 1;
	materials[1].mass = 1.0f;
	materials[1].restDensity = 10.0f;
	materials[1].viscosity = 1.0f;
	materials[1].bulkViscosity = 3.0f;
	materials[1].stiffness = 1.0f;
	materials[1].meltRate = 1.0f;
	materials[1].kElastic = 1.0f;



	materials[2].materialIndex = 2;
	materials[2].mass = 0.7f;
	materials[2].viscosity = 0.03f;

	materials[3].materialIndex = 3;

	const int size = numMaterials * sizeof(Material);
	cudaMalloc((void**)&d_materials, size);
	cudaMemcpy(d_materials, materials, size, cudaMemcpyHostToDevice);
	app::console() << "Materials intialized" << endl;

	app::console() << "Particle size : " << sizeof(Particle) << endl;
	app::console() << "Node size : " << sizeof(Node) << endl;
}
void SimulatorCUDA::initializeGrid(int sizeX, int sizeY) {
	gSizeX				= sizeX;
	gSizeY				= sizeY;
	gSizeY_3			= sizeY - 3;
	gSize			= gSizeX  *gSizeY;
	const int sizeOfGrid	= gSize * sizeof(Node);
	const int sizeOfActive	= gSize * sizeof(Node*);

	grid = new Node[gSize];
	gpuErrchk(cudaMalloc((void**)&d_grid, sizeOfGrid));

	gpuErrchk(cudaMemcpy(d_grid, grid, sizeOfGrid, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&d_active, sizeOfActive));
	gpuErrchk(cudaMalloc((void**)&d_activeCount, sizeof(int))); 
}

void SimulatorCUDA::addParticles(int n) {
	int pn = (int)sqrt((float)(n/3));
	int dn = n - (pn*pn * 3);
	float bw = 40.0f;
	float bh = 16.0f;
	float mx = bw / (float)pn;
	float my = bh / (float)pn;
	float offset = 10.0f;
	
	this->particleCount = n;
	const int sizeOfParticles		= particleCount * sizeof(Particle);

	gpuErrchk(cudaMalloc((void**)&d_particles, sizeOfParticles));
	
	int i, j;
	// Material 1
	for (i = 0; i < pn; i++) {
		for (j = 0; j < pn; j++) {
			float px = i*mx + offset;
			float py = j*my + offset;
			Particle p(&d_materials[0], px, py, ColorA(1, 0.5, 0.5, 1));
			p.initializeWeights(gSizeY);
			particles.push_back(p);
		}
	}

	// Material 2
	for (i = 0; i < pn; i++) {
		for (j = 0; j < pn; j++) {
			float px = i*mx + (this->gSizeX-bw)/2;
			float py = j*my + offset;
			Particle p(&d_materials[1], px, py, ColorA(0.5, 1, 0.5, 1));
			p.initializeWeights(gSizeY);
			particles.push_back(p);
		}
	}

	// Material 2
	for (i = 0; i < pn; i++) {
		for (j = 0; j < pn; j++) {
			float px = i*mx + this->gSizeX - bw - offset;
			float py = j*my + offset;
			Particle p(&d_materials[2], px, py, ColorA(0.5, 0.5, 1, 1));
			p.initializeWeights(gSizeY);
			particles.push_back(p);
		}
	}
	for(int x = 0; x < dn; x++)
	{
		float px = i*mx + this->gSizeX - bw - offset;
		float py = (j++)*my + offset;
		Particle p(&d_materials[2], px, py, ColorA(0.5, 0.5, 1, 1));
		p.initializeWeights(gSizeY);
		particles.push_back(p);
	}

	gpuErrchk(cudaMemcpy(d_particles, particles.data(), sizeOfParticles, cudaMemcpyHostToDevice));
}

__global__ void loop1(Particle* particles, Node* grid, int nParticles, int gSizeX, int gSizeY, int gSizeY_3)
{
	int pi = blockDim.x * blockIdx.x + threadIdx.x;
	if (pi < nParticles) {
		Particle &p = particles[pi];
		Material& mat = *p.mat;
		
		float gu = 0, gv = 0, dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
		Node* n = &grid[p.gi];
		float* ppx = p.px;
		float* ppy = p.py;
		float* pgx = p.gx;
		float* pgy = p.gy;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;
				gu += phi * n->u2;
				gv += phi * n->v2;			
				float gx = gxi * pyj;
				float gy = pxi * gyj;
				// Velocity gradient
				dudx += n->u2 * gx;
				dudy += n->u2 * gy;
				dvdx += n->v2 * gx;
				dvdy += n->v2 * gy;
			}
		}

		// Update stress tensor
		float w1 = dudy - dvdx;
		float wT0 = .5f * w1 * (p.T01 + p.T01);
		float wT1 = .5f * w1 * (p.T00 - p.T11);
		float D00 = dudx;
		float D01 = .5f * (dudy + dvdx);
		float D11 = dvdy;
		float trace = .5f * (D00 + D11);
		p.T00 += .5f * (-wT0 + (D00 - trace) - mat.meltRate * p.T00);
		p.T01 += .5f * (wT1 + D01 - mat.meltRate * p.T01);
		p.T11 += .5f * (wT0 + (D11 - trace) - mat.meltRate * p.T11);

		float norm = p.T00 * p.T00 + 2 * p.T01 * p.T01 + p.T11 * p.T11;

		if (norm > mat.maxDeformation)
		{
			p.T00 = p.T01 = p.T11 = 0;
		}

		p.x += gu;
		p.y += gv;

		p.gu = gu;
		p.gv = gv;

		p.u += mat.smoothing*(gu - p.u);
		p.v += mat.smoothing*(gv - p.v);

		// Hard boundary correction (Random numbers keep it from clustering)
		curandState_t cstate;
		curand_init(0, 0, 0, &cstate);


		if (p.x < 1) {
			p.x = 1 + .01*curand_uniform(&cstate);
			//p.x = 1;
		}
		else if (p.x > gSizeX - 2) {
			p.x = gSizeX - 2 - .01*curand_uniform(&cstate);
			//p.x = gSizeX - 2;
		}
		if (p.y < 1) {
			p.y = 1 + .01*curand_uniform(&cstate);
			//p.y = 1;
		}
		else if (p.y > gSizeY - 2) {
			p.y = gSizeY - 2 - .01*curand_uniform(&cstate);
			//p.y = gSizeY - 2;
		}

		// Update grid cell index and kernel weights
		int cx = p.cx = (int)(p.x - .5f);
		int cy = p.cy = (int)(p.y - .5f);
		p.gi = cx * gSizeY + cy;

		float x = cx - p.x;
		float y = cy - p.y;

		// Quadratic interpolation kernel weights - Not meant to be changed
		ppx[0] = .5f * x * x + 1.5f * x + 1.125f;
		pgx[0] = x + 1.5f;
		x++;
		ppx[1] = -x * x + .75f;
		pgx[1] = -2 * x;
		x++;
		ppx[2] = .5f * x * x - 1.5f * x + 1.125f;
		pgx[2] = x - 1.5f;

		ppy[0] = .5f * y * y + 1.5f * y + 1.125f;
		pgy[0] = y + 1.5f;
		y++;
		ppy[1] = -y * y + .75f;
		pgy[1] = -2 * y;
		y++;
		ppy[2] = .5f * y * y - 1.5f * y + 1.125f;
		pgy[2] = y - 1.5f;

		float m = p.mat->mass;
		float mu = m * p.u;
		float mv = m * p.v;
		int mi = p.mat->materialIndex;
		float *px = p.px;
		float *gx = p.gx;
		float *py = p.py;
		float *gy = p.gy;
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			float gxi = gx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float gyj = gy[j];
				float phi = pxi * pyj;
				// Add particle mass, velocity and density gradient to grid
//				n->mass += phi * m;
//				n->particleDensity += phi;
//				n->u += phi * mu;
//				n->v += phi * mv;
//				n->cgx[mi] += gxi * pyj;
//				n->cgy[mi] += pxi * gyj;
				atomicAdd(&(n->mass), phi * m);
				atomicAdd(&(n->particleDensity), phi);
				atomicAdd(&(n->u), phi * mu);
				atomicAdd(&(n->v), phi * mv);
				atomicAdd(&(n->cgx[mi]), gxi * pyj);
				atomicAdd(&(n->cgy[mi]), pxi * gyj);
				n->active = true;

			}
		}
	}
}

__global__ void loop2(Node* grid, Node** active, int* activeCount, int  gSize, int* counter)
{
	Node* gi = grid;
	//int gSizeXY = gSizeX * gSizeY;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < gSize) {
		Node* n = &gi[i];
		if (n->active && n->mass > 0) {
			int j = atomicAdd(activeCount, 1);
			active[j] = n;
			n->active = false;
			n->ax = n->ay = 0;
			n->gx = 0;
			n->gy = 0;
			n->u /= n->mass;
			n->v /= n->mass;
			for (int j = 0; j < numMaterials; j++) {
				n->gx += n->cgx[j];
				n->gy += n->cgy[j];
			}
			for (int j = 0; j < numMaterials; j++) {
				n->cgx[j] -= n->gx - n->cgx[j];
				n->cgy[j] -= n->gy - n->cgy[j];
			}
		}
	}
}

__global__ void loop3(Particle* particles, Node* grid, int nParticles, int gSizeX, int gSizeY, int gSizeY_3, float scale)
{
	int pi = blockDim.x * blockIdx.x + threadIdx.x;
	if (pi < nParticles) {
		Particle& p = particles[pi];
		Material& mat = *p.mat;

		float fx = 0, fy = 0, dudx = 0, dudy = 0, dvdx = 0, dvdy = 0, sx = 0, sy = 0;
		Node* n = &grid[p.gi];
		float *ppx = p.px;
		float *pgx = p.gx;
		float *ppy = p.py;
		float *pgy = p.gy;

		int materialId = mat.materialIndex;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;
				float gx = gxi * pyj;
				float gy = pxi * gyj;
				// Velocity gradient
				dudx += n->u * gx;
				dudy += n->u * gy;
				dvdx += n->v * gx;
				dvdy += n->v * gy;

				// Surface tension
				sx += phi * n->cgx[materialId];
				sy += phi * n->cgy[materialId];
			}
		}

		int cx = (int)p.x;
		int cy = (int)p.y;
		int gi = cx * gSizeY + cy;

		Node& n1 = grid[gi];
		Node& n2 = grid[gi + 1];
		Node& n3 = grid[gi + gSizeY];
		Node& n4 = grid[gi + gSizeY + 1];
		float density = uscip(n1.particleDensity, n1.gx, n1.gy,
			n2.particleDensity, n2.gx, n2.gy,
			n3.particleDensity, n3.gx, n3.gy,
			n4.particleDensity, n4.gx, n4.gy,
			p.x - cx, p.y - cy);

		float pressure = mat.stiffness / mat.restDensity * (density - mat.restDensity);
		if (pressure > 2) {
			pressure = 2;
		}

		// Update stress tensor
		float w1 = dudy - dvdx;
		float wT0 = .5f * w1 * (p.T01 + p.T01);
		float wT1 = .5f * w1 * (p.T00 - p.T11);
		float D00 = dudx;
		float D01 = .5f * (dudy + dvdx);
		float D11 = dvdy;
		float trace = .5f * (D00 + D11);
		D00 -= trace;
		D11 -= trace;
		p.T00 += .5f * (-wT0 + D00 - mat.meltRate * p.T00);
		p.T01 += .5f * (wT1 + D01 - mat.meltRate * p.T01);
		p.T11 += .5f * (wT0 + D11 - mat.meltRate * p.T11);

		// Stress tensor fracture
		float norm = p.T00 * p.T00 + 2 * p.T01 * p.T01 + p.T11 * p.T11;

		if (norm > mat.maxDeformation)
		{
			p.T00 = p.T01 = p.T11 = 0;
		}

		float T00 = mat.mass * (mat.kElastic * p.T00 + mat.viscosity * D00 + pressure + trace * mat.bulkViscosity);
		float T01 = mat.mass * (mat.kElastic * p.T01 + mat.viscosity * D01);
		float T11 = mat.mass * (mat.kElastic * p.T11 + mat.viscosity * D11 + pressure + trace * mat.bulkViscosity);

		// Surface tension
		float lenSq = sx * sx + sy * sy;
		if (lenSq > 0)
		{
			float len = sqrtf(lenSq);
			float a = mat.mass * mat.surfaceTension / len;
			T00 -= a * (.5f * lenSq - sx * sx);
			T01 -= a * (-sx * sy);
			T11 -= a * (.5f * lenSq - sy * sy);
		}

		// Wall force
		if (p.x < 4) {
			fx += (4 - p.x);
		}
		else if (p.x > gSizeX - 5) {
			fx += (gSizeX - 5 - p.x);
		}
		if (p.y < 4) {
			fy += (4 - p.y);
		}
		else if (p.y > gSizeY - 5) {
			fy += (gSizeY - 5 - p.y);
		}


		// Add forces to grid
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;

				float gx = gxi * pyj;
				float gy = pxi * gyj;
				//n->ax += -(gx * T00 + gy * T01) + fx * phi;
				//n->ay += -(gx * T01 + gy * T11) + fy * phi;
				atomicAdd(&(n->ax),-(gx * T00 + gy * T01) + fx * phi);
				atomicAdd(&(n->ay),-(gx * T01 + gy * T11) + fy * phi);
			}
		}

		//Assign final particle Position
		p.pos.x = p.x*scale;
		p.pos.y = p.y*scale;
		p.trail.x = (p.x - p.gu)*scale;
		p.trail.y = (p.y - p.gv)*scale;
	}
}

__global__ void loop4(Node** active, int* activeCount)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<*activeCount){
		Node* n = active[i];
		//if (n->active && n->mass > 0) {
		n->u2 = 0;
		n->v2 = 0;
		n->ax /= n->mass;
		n->ay /= n->mass;
		//}
	}
}

__global__ void loop5(Particle* particles, Node* grid, int particleCount, int gSizeY_3, Particle* r_particles)
{
	int pi = blockDim.x * blockIdx.x + threadIdx.x;
	if (pi<particleCount) {
		Particle& p = particles[pi];
		Material& mat = *p.mat;
		// Update particle velocities
		Node* n = &grid[p.gi];
		float *px = p.px;
		float *py = p.py;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float phi = pxi * pyj;
				p.u += phi * n->ax;
				p.v += phi * n->ay;
			}
		}

		p.v += mat.gravity;
		p.u *= 1 - mat.damping;
		p.v *= 1 - mat.damping;

		float m = p.mat->mass;
		float mu = m * p.u;
		float mv = m * p.v;

		// Add particle velocities back to the grid
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float phi = pxi * pyj;
				//n->u2 += phi * mu;
				//n->v2 += phi * mv;
				atomicAdd(&(n->u2), phi*mu);
				atomicAdd(&(n->v2), phi*mv);
			}
		}
		r_particles[pi] = p;
	}
}

__global__ void loop6(Node** active, int* activeCount)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<*activeCount) {
		Node* n =active[i];
		//if (n->active && n->mass > 0) {
		n->u2 /= n->mass;
		n->v2 /= n->mass;

		n->mass = 0;
		n->particleDensity = 0;
		n->u = 0;
		n->v = 0;
		memset(n->cgx, 0, numMaterials * sizeof(float) * 2);
		//n->active = false;
		//}
	}
}

void SimulatorCUDA::updateCUDA(){
	int activeCount;
	dim3 dimBlockP(1024, 1, 1);
	dim3 dimGridP((particleCount+1023)/1024, 1, 1);
	dim3 dimBlockN(gSizeX, 1, 1);
	dim3 dimGridN(gSizeY, 1, 1);
	dim3 dimBlockA;
	dim3 dimGridA;
	Particle* mappedParticles;
	size_t num_bytes;

	loop1 << <dimGridP, dimBlockP >> >(d_particles, d_grid, particleCount, gSizeX, gSizeY, gSizeY_3); //Pass 1
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaMemset(d_activeCount, 0, sizeof(int));
	loop2 << <dimGridN, dimBlockN >> >(d_grid, d_active, d_activeCount, gSize, d_counter); //Pass 2
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	loop3 << <dimGridP, dimBlockP >> >(d_particles, d_grid, particleCount, gSizeX, gSizeY, gSizeY_3, scale); //Pass 3
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	cudaMemcpy(&activeCount, d_activeCount, sizeof(int), cudaMemcpyDeviceToHost);
	//::console() << "ACTIVE NODES : " << activeCount << endl;
	dimBlockA	= dim3( 128, 1, 1 );
	dimGridA	= dim3((activeCount + 127) / 128, 1, 1 );

	loop4 << <dimGridA, dimBlockA >> >(d_active, d_activeCount); //Pass 4
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaGraphicsMapResources(1, &cudaVboResource, 0);
	gpuErrchk(cudaPeekAtLastError());
	cudaGraphicsResourceGetMappedPointer((void **)&mappedParticles, &num_bytes, cudaVboResource);
	gpuErrchk(cudaPeekAtLastError());

	loop5 << <dimGridP, dimBlockP >> >(d_particles, d_grid, particleCount, gSizeY_3, mappedParticles); //Pass 5
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize())

	cudaGraphicsUnmapResources(1, &cudaVboResource, 0);
	gpuErrchk(cudaPeekAtLastError());

	loop6 << <dimGridA, dimBlockA >> >(d_active, d_activeCount); //Pass 6
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void SimulatorCUDA::update() {
	
	int nParticles = particleCount;

#pragma omp parallel for
	for (int pi = 0; pi < nParticles; pi++) {
		Particle &p = particles[pi];
		Material& mat = *p.mat;

		float gu = 0, gv = 0, dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
		Node* n = &grid[p.gi];
		float *ppx = p.px;
		float *ppy = p.py;
		float* pgx = p.gx;
		float* pgy = p.gy;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;
				gu += phi * n->u2;
				gv += phi * n->v2;
				float gx = gxi * pyj;
				float gy = pxi * gyj;
				// Velocity gradient
				dudx += n->u2 * gx;
				dudy += n->u2 * gy;
				dvdx += n->v2 * gx;
				dvdy += n->v2 * gy;
			}
		}

		// Update stress tensor
		float w1 = dudy - dvdx;
		float wT0 = .5f * w1 * (p.T01 + p.T01);
		float wT1 = .5f * w1 * (p.T00 - p.T11);
		float D00 = dudx;
		float D01 = .5f * (dudy + dvdx);
		float D11 = dvdy;
		float trace = .5f * (D00 + D11);
		p.T00 += .5f * (-wT0 + (D00 - trace) - mat.meltRate * p.T00);
		p.T01 += .5f * (wT1 + D01 - mat.meltRate * p.T01);
		p.T11 += .5f * (wT0 + (D11 - trace) - mat.meltRate * p.T11);

		float norm = p.T00 * p.T00 + 2 * p.T01 * p.T01 + p.T11 * p.T11;

		if (norm > mat.maxDeformation)
		{
			p.T00 = p.T01 = p.T11 = 0;
		}

		p.x += gu;
		p.y += gv;

		p.gu = gu;
		p.gv = gv;

		p.u += mat.smoothing*(gu - p.u);
		p.v += mat.smoothing*(gv - p.v);

		// Hard boundary correction (Random numbers keep it from clustering)
		if (p.x < 1) {
			p.x = 1 + .01*rand() / RAND_MAX;
		}
		else if (p.x > gSizeX - 2) {
			p.x = gSizeX - 2 - .01*rand() / RAND_MAX;
		}
		if (p.y < 1) {
			p.y = 1 + .01*rand() / RAND_MAX;
		}
		else if (p.y > gSizeY - 2) {
			p.y = gSizeY - 2 - .01*rand() / RAND_MAX;
		}

		// Update grid cell index and kernel weights
		int cx = p.cx = (int)(p.x - .5f);
		int cy = p.cy = (int)(p.y - .5f);
		p.gi = cx * gSizeY + cy;

		float x = cx - p.x;
		float y = cy - p.y;

		// Quadratic interpolation kernel weights - Not meant to be changed
		ppx[0] = .5f * x * x + 1.5f * x + 1.125f;
		pgx[0] = x + 1.5f;
		x++;
		ppx[1] = -x * x + .75f;
		pgx[1] = -2 * x;
		x++;
		ppx[2] = .5f * x * x - 1.5f * x + 1.125f;
		pgx[2] = x - 1.5f;

		ppy[0] = .5f * y * y + 1.5f * y + 1.125f;
		pgy[0] = y + 1.5f;
		y++;
		ppy[1] = -y * y + .75f;
		pgy[1] = -2 * y;
		y++;
		ppy[2] = .5f * y * y - 1.5f * y + 1.125f;
		pgy[2] = y - 1.5f;

		float m = p.mat->mass;
		float mu = m * p.u;
		float mv = m * p.v;
		int mi = p.mat->materialIndex;
		float *px = p.px;
		float *gx = p.gx;
		float *py = p.py;
		float *gy = p.gy;
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			float gxi = gx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float gyj = gy[j];
				float phi = pxi * pyj;
				// Add particle mass, velocity and density gradient to grid
				n->mass += phi * m;
				n->particleDensity += phi;
				n->u += phi * mu;
				n->v += phi * mv;
				n->cgx[mi] += gxi * pyj;
				n->cgy[mi] += pxi * gyj;
				n->active = true;
			}
		}
	}


	// Add active nodes to list
	active.clear();
	Node* gi = grid;
	int gSizeXY = gSizeX * gSizeY;

	for (int i = 0; i < gSizeXY; i++) {
		Node& n = *(gi);
		if (n.active && n.mass > 0) {
			active.push_back(gi);
			n.active = false;
			n.ax = n.ay = 0;
			n.gx = 0;
			n.gy = 0;
			n.u /= n.mass;
			n.v /= n.mass;
			for (int j = 0; j < numMaterials; j++) {
				n.gx += n.cgx[j];
				n.gy += n.cgy[j];
			}
			for (int j = 0; j < numMaterials; j++) {
				n.cgx[j] -= n.gx - n.cgx[j];
				n.cgy[j] -= n.gy - n.cgy[j];
			}
		}
		gi++;
	}

	int nActive = active.size();

#pragma omp parallel for
	for (int pi = 0; pi < nParticles; pi++) {
		Particle& p = particles[pi];
		Material& mat = *p.mat;

		float fx = 0, fy = 0, dudx = 0, dudy = 0, dvdx = 0, dvdy = 0, sx = 0, sy = 0;
		Node* n = &grid[p.gi];
		float *ppx = p.px;
		float *pgx = p.gx;
		float *ppy = p.py;
		float *pgy = p.gy;

		int materialId = mat.materialIndex;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;
				float gx = gxi * pyj;
				float gy = pxi * gyj;
				// Velocity gradient
				dudx += n->u * gx;
				dudy += n->u * gy;
				dvdx += n->v * gx;
				dvdy += n->v * gy;

				// Surface tension
				sx += phi * n->cgx[materialId];
				sy += phi * n->cgy[materialId];
			}
		}

		int cx = (int)p.x;
		int cy = (int)p.y;
		int gi = cx * gSizeY + cy;

		Node& n1 = grid[gi];
		Node& n2 = grid[gi + 1];
		Node& n3 = grid[gi + gSizeY];
		Node& n4 = grid[gi + gSizeY + 1];
		float density = uscip(n1.particleDensity, n1.gx, n1.gy,
			n2.particleDensity, n2.gx, n2.gy,
			n3.particleDensity, n3.gx, n3.gy,
			n4.particleDensity, n4.gx, n4.gy,
			p.x - cx, p.y - cy);

		float pressure = mat.stiffness / mat.restDensity * (density - mat.restDensity);
		if (pressure > 2) {
			pressure = 2;
		}

		// Update stress tensor
		float w1 = dudy - dvdx;
		float wT0 = .5f * w1 * (p.T01 + p.T01);
		float wT1 = .5f * w1 * (p.T00 - p.T11);
		float D00 = dudx;
		float D01 = .5f * (dudy + dvdx);
		float D11 = dvdy;
		float trace = .5f * (D00 + D11);
		D00 -= trace;
		D11 -= trace;
		p.T00 += .5f * (-wT0 + D00 - mat.meltRate * p.T00);
		p.T01 += .5f * (wT1 + D01 - mat.meltRate * p.T01);
		p.T11 += .5f * (wT0 + D11 - mat.meltRate * p.T11);

		// Stress tensor fracture
		float norm = p.T00 * p.T00 + 2 * p.T01 * p.T01 + p.T11 * p.T11;

		if (norm > mat.maxDeformation)
		{
			p.T00 = p.T01 = p.T11 = 0;
		}

		float T00 = mat.mass * (mat.kElastic * p.T00 + mat.viscosity * D00 + pressure + trace * mat.bulkViscosity);
		float T01 = mat.mass * (mat.kElastic * p.T01 + mat.viscosity * D01);
		float T11 = mat.mass * (mat.kElastic * p.T11 + mat.viscosity * D11 + pressure + trace * mat.bulkViscosity);

		// Surface tension
		float lenSq = sx * sx + sy * sy;
		if (lenSq > 0)
		{
			float len = sqrtf(lenSq);
			float a = mat.mass * mat.surfaceTension / len;
			T00 -= a * (.5f * lenSq - sx * sx);
			T01 -= a * (-sx * sy);
			T11 -= a * (.5f * lenSq - sy * sy);
		}

		// Wall force
		if (p.x < 4) {
			fx += (4 - p.x);
		}
		else if (p.x > gSizeX - 5) {
			fx += (gSizeX - 5 - p.x);
		}
		if (p.y < 4) {
			fy += (4 - p.y);
		}
		else if (p.y > gSizeY - 5) {
			fy += (gSizeY - 5 - p.y);
		}


		// Add forces to grid
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = ppx[i];
			float gxi = pgx[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = ppy[j];
				float gyj = pgy[j];
				float phi = pxi * pyj;

				float gx = gxi * pyj;
				float gy = pxi * gyj;
				n->ax += -(gx * T00 + gy * T01) + fx * phi;
				n->ay += -(gx * T01 + gy * T11) + fy * phi;
			}
		}

		//Assign final particle Position
		p.pos.x = p.x*scale;
		p.pos.y = p.y*scale;
		p.trail.x = (p.x - p.gu)*scale;
		p.trail.y = (p.y - p.gv)*scale;

	}

	// Update acceleration of nodes
#pragma omp parallel for
	for (int i = 0; i < nActive; i++) {
		Node& n = *active[i];
		n.u2 = 0;
		n.v2 = 0;
		n.ax /= n.mass;
		n.ay /= n.mass;
	}

#pragma omp parallel for
	for (int pi = 0; pi < nParticles; pi++) {
		Particle& p = particles[pi];
		Material& mat = *p.mat;
		// Update particle velocities
		Node* n = &grid[p.gi];
		float *px = p.px;
		float *py = p.py;
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float phi = pxi * pyj;
				p.u += phi * n->ax;
				p.v += phi * n->ay;
			}
		}

		p.v += mat.gravity;
		p.u *= 1 - mat.damping;
		p.v *= 1 - mat.damping;

		float m = p.mat->mass;
		float mu = m * p.u;
		float mv = m * p.v;

		// Add particle velocities back to the grid
		n = &grid[p.gi];
		for (int i = 0; i < 3; i++, n += gSizeY_3) {
			float pxi = px[i];
			for (int j = 0; j < 3; j++, n++) {
				float pyj = py[j];
				float phi = pxi * pyj;
				n->u2 += phi * mu;
				n->v2 += phi * mv;
			}
		}
	}

	// Update node velocities
	for (int i = 0; i < nActive; i++) {
		Node& n = *active[i];
		n.u2 /= n.mass;
		n.v2 /= n.mass;

		n.mass = 0;
		n.particleDensity = 0;
		n.u = 0;
		n.v = 0;
		memset(n.cgx, 0, 2 * numMaterials * sizeof(float));
	}
}



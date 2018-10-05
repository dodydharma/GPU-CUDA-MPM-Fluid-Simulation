#pragma once

#include <stdio.h>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cinder/app/App.h"
#include "Material.cuh"
#include "Node.cuh"
#include "Particle.cuh"


class SimulatorCUDA{
	int gSize, gSizeX, gSizeY, gSizeY_3;

	Node* grid;
	Node* d_grid;

	int* d_counter;

	std::vector<Node*> active;
	Node** d_active;

	int* d_activeCount;

	Material materials[numMaterials];
	Material* d_materials;

	dim3 dimBlockP;
	dim3 dimGridP;

	dim3 dimBlockP2;
	dim3 dimGridP2;

	dim3 dimBlockN;
	dim3 dimGridN;

	dim3 dimBlockA;
	dim3 dimGridA;

	float uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v);
public:
	std::vector<Particle> particles;
	Particle* d_particles;
	int particleCount;
	struct cudaGraphicsResource *cudaVboResource;

	int step;
	int maxStep;

	std::chrono::high_resolution_clock::time_point begin;
	std::chrono::high_resolution_clock::time_point end;

	float scale;
	int blockDim;

	SimulatorCUDA();
	void initializeGrid(int sizeX, int sizeY);
	void addParticles(int n);
	void initializeCUDA(int blockDim);
	void update();
};
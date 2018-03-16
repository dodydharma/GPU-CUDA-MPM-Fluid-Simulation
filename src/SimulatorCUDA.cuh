#pragma once

#include <stdio.h>
#include "cinder/app/App.h"
#include "Material.cuh"
#include "Node.cuh"
#include "Particle.cuh"


class SimulatorCUDA{
	int gSize, gSizeX, gSizeY, gSizeY_3;
	

	Node* grid;
	Node* d_grid;
	int*d_counter;
	std::vector<Node*> active;
	Node** d_active;
	int* d_activeCount;

	float* d_gridAtt;
	float* d_particleAtt;

	Material materials[numMaterials];
	Material* d_materials;

	float uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v);
public:
	std::vector<Particle> particles;
	Particle* d_particles;
	int particleCount;
	struct cudaGraphicsResource *cudaVboResource;

	float scale;

	SimulatorCUDA();
	void initializeGrid(int sizeX, int sizeY);
	void addParticles(int n);
	void update();
	void updateCUDA();
};
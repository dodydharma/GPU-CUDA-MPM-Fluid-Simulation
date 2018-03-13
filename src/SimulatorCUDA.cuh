#pragma once

#include <stdio.h>
#include "cinder/app/App.h"
#include "Material.cuh"
#include "Node.cuh"
#include "Particle.cuh"


class SimulatorCUDA{
	int gSizeX, gSizeY, gSizeY_3;

	Node* grid;
	Node* d_grid;

	std::vector<Node*> active;
	//Node** d_active;
	//int* d_nActive;

	Material materials[numMaterials];
	Material* d_materials;

	float uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v);
public:
	std::vector<Particle> particles;

	Particle* d_particles;


	float scale;

	SimulatorCUDA();
	void initializeGrid(int sizeX, int sizeY);
	void addParticles(int n);
	void update();
	void updateCUDA();
	int* cuda_main(void);
};
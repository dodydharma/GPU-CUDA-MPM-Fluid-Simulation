#ifndef PARTICLE_H
#define PARTICLE_H

#include "cinder/app/App.h"
#include "Material.cuh"

struct Particle {
	cinder::vec3		pos;
	cinder::vec3        trail;
	cinder::ColorA		color;

	Material* mat;
	float x, y, u, v, gu, gv, T00, T01, T11;
	int cx, cy, gi;
	float *px;
	float *py;
	float *gx;
	float *gy;

	Particle(Material* mat);

	Particle(Material* mat, float x, float y);

	Particle(Material* mat, float x, float y, cinder::ColorA c);

	Particle(Material* mat, float x, float y, float u, float v);

	~Particle();

	void initializeArrays();

	void initializeWeights(int gSizeY);
};

#endif
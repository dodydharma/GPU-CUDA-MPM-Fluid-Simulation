#pragma once

#define numMaterials 4

#include <stdio.h>
#include "cinder/app/App.h"

using namespace std;
using namespace cinder;

#ifndef STRUCTDEF
#define STRUCTDEF

struct Material {
	float mass, restDensity, stiffness, bulkViscosity, surfaceTension, kElastic, maxDeformation, meltRate, viscosity, damping, friction, stickiness, smoothing, gravity;
	int materialIndex;

	Material() : mass(1), restDensity(2), stiffness(1), bulkViscosity(1), surfaceTension(0), kElastic(0), maxDeformation(0), meltRate(0), viscosity(.02), damping(.001), friction(0), stickiness(0), smoothing(.02), gravity(.03) {};
};

struct Particle {
	vec3		pos;
	vec3        trail;
	ColorA		color;

	Material* mat;
	float x, y, u, v, gu, gv, T00, T01, T11;
	int cx, cy, gi;
	float px[3];
	float py[3];
	float gx[3];
	float gy[3];

	Particle(Material* mat) : pos(0, 0, 0), color(.1, .5, 1, 1), mat(mat), x(0), y(0), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {
		memset(px, 0, 12 * sizeof(float));
	}

	Particle(Material* mat, float x, float y) : pos(x, y, 0), color(.1, .5, 1, 1), mat(mat), x(x), y(y), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {
		memset(px, 0, 12 * sizeof(float));
	}

	Particle(Material* mat, float x, float y, ColorA c) : pos(x, y, 0), color(c), mat(mat), x(x), y(y), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {
		memset(px, 0, 12 * sizeof(float));
	}


	Particle(Material* mat, float x, float y, float u, float v) :pos(x, y, 0), color(.1, .5, 1, 1), mat(mat), x(x), y(y), u(u), v(v), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {
		memset(px, 0, 12 * sizeof(float));
	}

	void initializeWeights(int gSizeY) {
		cx = (int)(x - .5f);
		cy = (int)(y - .5f);
		gi = cx * gSizeY + cy;

		float cx_x = cx - x;
		float cy_y = cy - y;

		// Quadratic interpolation kernel weights - Not meant to be changed
		px[0] = .5f * cx_x * cx_x + 1.5f * cx_x + 1.125f;
		gx[0] = cx_x + 1.5f;
		cx_x++;
		px[1] = -cx_x * cx_x + .75f;
		gx[1] = -2 * cx_x;
		cx_x++;
		px[2] = .5f * cx_x * cx_x - 1.5f * cx_x + 1.125f;
		gx[2] = cx_x - 1.5f;

		py[0] = .5f * cy_y * cy_y + 1.5f * cy_y + 1.125f;
		gy[0] = cy_y + 1.5f;
		cy_y++;
		py[1] = -cy_y * cy_y + .75f;
		gy[1] = -2 * cy_y;
		cy_y++;
		py[2] = .5f * cy_y * cy_y - 1.5f * cy_y + 1.125f;
		gy[2] = cy_y - 1.5f;
	}
};

struct Node {
	float mass, particleDensity, gx, gy, u, v, u2, v2, ax, ay;
	float cgx[numMaterials];
	float cgy[numMaterials];
	bool active;
	Node() : mass(0), particleDensity(0), gx(0), gy(0), u(0), v(0), u2(0), v2(0), ax(0), ay(0), active(false) {
		memset(cgx, 0, 2 * numMaterials * sizeof(float));
	}
};

#endif

class Simulator {
protected:
	int gSizeX, gSizeY, gSizeY_3;
	Node* grid;
	vector<Node*> active;
	Material materials[numMaterials];

	float uscip(float p00, float x00, float y00, float p01, float x01, float y01, float p10, float x10, float y10, float p11, float x11, float y11, float u, float v);

public:
	vector<Particle> particles;
	float scale;

	Simulator();
	void initializeGrid(int sizeX, int sizeY);
	void addParticles();
	void update();
};
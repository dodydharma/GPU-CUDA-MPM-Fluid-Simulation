#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>
#include "cinder/app/App.h"
#include "Material.cuh"
#include "Common.cuh"

struct Particle {
	cinder::vec3		pos;
	cinder::vec3        trail;
	cinder::ColorA		color;

	Material* mat;
	float x, y, u, v, gu, gv, T00, T01, T11;
	int cx, cy, gi;
	float px[3];
	float py[3];
	float gx[3];
	float gy[3];

	Particle(Material* mat) : pos(0, 0, 0), color(.1, .5, 1, 1), mat(mat), x(0), y(0), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {}

	Particle(Material* mat, float x, float y) : pos(x, y, 0), color(.1, .5, 1, 1), mat(mat), x(x), y(y), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {}

	Particle(Material* mat, float x, float y, cinder::ColorA c) : pos(x, y, 0), color(c), mat(mat), x(x), y(y), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {}

	Particle(Material* mat, float x, float y, float u, float v) : pos(x, y, 0), color(.1, .5, 1, 1), mat(mat), x(x), y(y), u(u), v(v), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {}

	~Particle() {}

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

#endif
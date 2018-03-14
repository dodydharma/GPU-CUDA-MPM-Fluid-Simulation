#include "Particle.cuh"

#include <cuda_runtime.h>
#include "Common.cuh"

using namespace cinder;

__host__ Particle::Particle(Material* mat) : pos(0, 0, 0), color(.1, .5, 1, 1), mat(mat), x(0), y(0), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {

}

__host__ Particle::Particle(Material* mat, float x, float y) : pos(x, y, 0), color(.1, .5, 1, 1), mat(mat), x(x), y(y), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {

}

__host__ Particle::Particle(Material* mat, float x, float y, ColorA c) : pos(x, y, 0), color(c), mat(mat), x(x), y(y), u(0), v(0), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {

}


__host__ Particle::Particle(Material* mat, float x, float y, float u, float v) :pos(x, y, 0), color(.1, .5, 1, 1), mat(mat), x(x), y(y), u(u), v(v), T00(0), T01(0), T11(0), cx(0), cy(0), gi(0) {

}

__host__ Particle::~Particle()
{

}


__host__ void Particle::initAttArrays(float* px)
{
	this->px = px;
	this->py = px + 3;
	this->gx = px + 6;
	this->gy = px + 9;
}

__host__ void Particle::initializeWeights(int gSizeY) {
	cx = (int)(x - .5f);
	cy = (int)(y - .5f);
	gi = cx * gSizeY + cy;

	float cx_x = cx - x;
	float cy_y = cy - y;

	float tpx[12];
	float* tpy = &tpx[3];
	float* tgx = &tpx[6];
	float* tgy = &tpx[9];
	memset(tpx, 0, 12 * sizeof(float));

	// Quadratic interpolation kernel weights - Not meant to be changed
	tpx[0] = .5f * cx_x * cx_x + 1.5f * cx_x + 1.125f;
	tgx[0] = cx_x + 1.5f;
	cx_x++;
	tpx[1] = -cx_x * cx_x + .75f;
	tgx[1] = -2 * cx_x;
	cx_x++;
	tpx[2] = .5f * cx_x * cx_x - 1.5f * cx_x + 1.125f;
	tgx[2] = cx_x - 1.5f;

	tpy[0] = .5f * cy_y * cy_y + 1.5f * cy_y + 1.125f;
	tgy[0] = cy_y + 1.5f;
	cy_y++;
	tpy[1] = -cy_y * cy_y + .75f;
	tgy[1] = -2 * cy_y;
	cy_y++;
	tpy[2] = .5f * cy_y * cy_y - 1.5f * cy_y + 1.125f;
	tgy[2] = cy_y - 1.5f;

	const int size = 12 * sizeof(float);
	
	gpuErrchk(cudaMemcpy(px, tpx, size, cudaMemcpyHostToDevice));
}
#ifndef NODE_H
#define NODE_H

#define numMaterials 4

struct Node {
	float mass, particleDensity, gx, gy, u, v, u2, v2, ax, ay;
	float* cgx;
	float* cgy;
	bool active;
	Node();
	~Node();
};

#endif
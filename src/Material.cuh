#ifndef MATERIAL_H
#define MATERIAL_H

struct Material {
	float mass, restDensity, stiffness, bulkViscosity, surfaceTension, kElastic, maxDeformation, meltRate, viscosity, damping, friction, stickiness, smoothing, gravity;
	int materialIndex;

	Material();
};

#endif

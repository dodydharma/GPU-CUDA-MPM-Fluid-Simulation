#ifndef MATERIAL_H
#define MATERIAL_H

struct Material {
	float mass, restDensity, stiffness, bulkViscosity, surfaceTension, kElastic, maxDeformation, meltRate, viscosity, damping, friction, stickiness, smoothing, gravity;
	int materialIndex;

	Material() : mass(1), restDensity(2), stiffness(1), bulkViscosity(1), surfaceTension(0), kElastic(0), maxDeformation(0), meltRate(0), viscosity(.02), damping(.001), friction(0), stickiness(0), smoothing(.02), gravity(.03) {}
};

#endif

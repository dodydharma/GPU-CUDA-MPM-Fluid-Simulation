#include "Material.cuh"

#include <cuda_runtime.h>

__host__ Material::Material() : mass(1), restDensity(2), stiffness(1), bulkViscosity(1), surfaceTension(0), kElastic(0), maxDeformation(0), meltRate(0), viscosity(.02), damping(.001), friction(0), stickiness(0), smoothing(.02), gravity(.03) {};

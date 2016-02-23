
#ifndef VALIS_DISTANCE_PRIMATIVE_CUH
#define VALIS_DISTANCE_PRIMATIVE_CUH

#include <glm\vec3.hpp>

#include "device_launch_parameters.h"
#include "AABB.cuh"

class DistancePrimitive
{
public:

	__host__ __device__ virtual inline float
	distanceFromPoint(glm::vec3 point) = 0;

	__host__ __device__ virtual inline AABB
	calculateBoundingVolume() = 0;
};

#endif



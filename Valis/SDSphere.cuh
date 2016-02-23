#ifndef VALIS_SDSPHERE_CUH
#define VALIS_SDSPHERE_CUH

#include <glm\vec3.hpp>

#include "DistancePrimitive.cuh"
#include "GLMUtil.cuh"

class SDSphere : public DistancePrimitive
{
public:

	__host__ __device__
	SDSphere(float radius, glm::vec3 position) : radius(radius), position(position)
	{

	}

	__host__ __device__ inline float
	distanceFromPoint(glm::vec3 point)
	{
		return GLMUtil::length(point - position) - radius;
	}

	__host__ __device__ virtual inline AABB
	calculateBoundingVolume()
	{
		return AABB(glm::vec2(position.x - radius, position.y - radius), glm::vec2(position.x + radius, position.y + radius));
	}

private:
	float radius;
	glm::vec3 position;
};

#endif
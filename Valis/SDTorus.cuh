#ifndef VALIS_SDTORUS_CUH
#define VALIS_SDTORUS_CUH

#include <glm\vec3.hpp>
#include <glm\vec2.hpp>

#include "DistancePrimitive.cuh"
#include "GLMUtil.cuh"

class SDTorus : public DistancePrimitive
{
public:

	__host__ __device__
	SDTorus(float outer, float radius, glm::vec3 position) : dimensions(glm::vec2(outer, radius)), position(position)
	{

	}

	__host__ __device__ inline float
	distanceFromPoint(glm::vec3 point)
	{
		point -= position;
		glm::vec2 q = glm::vec2(GLMUtil::length(glm::vec2(point.x, point.y)) - dimensions.x, point.z);
		return GLMUtil::length(q) - dimensions.y;
	}

	__host__ __device__ virtual inline AABB
	calculateBoundingVolume()
	{
		return AABB(glm::vec2(0, 0), glm::vec2(0, 0));
	}

private:
	glm::vec3 position;
	glm::vec2 dimensions;
};

#endif
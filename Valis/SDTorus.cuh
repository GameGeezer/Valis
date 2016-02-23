#ifndef VALIS_SDTORUS_CUH
#define VALIS_SDTORUS_CUH

#include <glm\vec3.hpp>
#include <glm\vec2.hpp>

#include "DistancePrimitive.cuh"
#include "GLMUtil.cuh"

class SDTorus : public DistancePrimitive
{
public:

	SDTorus(float outer, float radius) : dimensions(glm::vec2(outer, radius))
	{

	}

	__host__ __device__ inline float
	distanceFromPoint(glm::vec3 point)
	{
		glm::vec2 q = glm::vec2(GLMUtil::length(glm::vec2(point.x, point.y)) - dimensions.x, point.z);
		return GLMUtil::length(q) - dimensions.y;
	}

private:
	glm::vec2 dimensions;
};

#endif
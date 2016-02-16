
#ifndef VALIS_DISTANCEFUNCTIONS_H
#define VALIS_DISTANCEFUNCTIONS_H

#include <glm\glm.hpp>
#include <glm\vec3.hpp>
#include <glm\gtx\extented_min_max.hpp>

#include "GLMUtil.cuh"

__host__ __device__ float
sdSphere(glm::vec3& position, float radius)
{
	return GLMUtil::length(position) - radius;
}

__host__ __device__ float
sdBox(glm::vec3 position, glm::vec3 corner)
{
	glm::vec3 d = glm::abs(position);
	
	return  0;// min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

__host__ __device__ float
sdTorous(glm::vec3 position, glm::vec2 t)
{
	glm::vec2 q = glm::vec2(GLMUtil::length(glm::vec2(position.x, position.y)) - t.x, position.z);

	return GLMUtil::length(q) - t.y;
}


//Broke
__host__ __device__ float
sdCylinder(glm::vec3 position, glm::vec3 c)
{
	glm::vec2 d = glm::vec2(position.x, position.z) - glm::vec2(c.x, c.y);

	return GLMUtil::length(d) - c.z;
}

__host__ __device__ float
sdCone(glm::vec3 position, glm::vec2 c)
{
	c = glm::normalize(c);

	float q = GLMUtil::length(glm::vec2(position.x, position.y));

	return glm::dot(c, glm::vec2(q, position.z));
}

#endif
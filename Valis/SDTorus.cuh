#ifndef VALIS_SDTORUS_CUH
#define VALIS_SDTORUS_CUH

#include <glm\vec3.hpp>
#include <glm\vec2.hpp>

#include "DistancePrimitive.cuh"
#include "GLMUtil.cuh"

#include "cuda_runtime.h"
#include "CudaHelper.cuh"

class SDTorus : public DistancePrimitive
{
public:

	__host__ __device__
		SDTorus(float outer, float radius, glm::vec3 scale, glm::vec3 translation, glm::vec3 rotationAxis, float angle) : DistancePrimitive(1, scale, translation, rotationAxis, angle), dimensions(glm::vec2(outer, radius))
	{

	}

	__host__ __device__
		SDTorus(float outer, float radius, glm::vec3 scale, glm::mat4 lookat) : DistancePrimitive(1, scale, lookat), dimensions(glm::vec2(outer, radius))
	{

	}

	__host__ inline DistancePrimitive*
	copyToDevice()
	{
		SDTorus* deviceTorus;

		assertCUDA(cudaMalloc((void **)&deviceTorus, sizeof(SDTorus)));
		assertCUDA(cudaMemcpy(deviceTorus, this, sizeof(SDTorus), cudaMemcpyHostToDevice));

		return deviceTorus;
	}

	__host__ __device__ inline float
	distanceFromPoint(glm::vec4 point)
	{
		point = transform * point;
		glm::vec2 q = glm::vec2(GLMUtil::length(glm::vec2(point.x, point.y)) - dimensions.x, point.z);
		return GLMUtil::length(q) - dimensions.y;
	}

	__host__ __device__ inline AABB
	calculateBoundingVolume() 
	{
		return AABB(glm::vec2(0, 0), glm::vec2(0, 0));
	}

	glm::vec2 dimensions;
};

#endif
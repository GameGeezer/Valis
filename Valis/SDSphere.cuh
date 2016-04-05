#ifndef VALIS_SDSPHERE_CUH
#define VALIS_SDSPHERE_CUH

#include <glm\vec3.hpp>
#include "cuda_runtime.h"

#include "DistancePrimitive.cuh"
#include "GLMUtil.cuh"
#include "CudaHelper.cuh"

class SDSphere : public DistancePrimitive
{
public:

	__host__ __device__
	SDSphere(float radius, glm::vec3 scale, glm::vec3 translation, glm::vec3 rotationAxis, float angle) : DistancePrimitive(0, scale, translation, rotationAxis, angle), radius(radius)
	{

	}

	__host__ __device__
	SDSphere(float radius, glm::vec3 scale, glm::mat4 lookat) : DistancePrimitive(0, scale, lookat), radius(radius)
	{

	}

	__host__ inline DistancePrimitive*
	copyToDevice()
	{
		SDSphere* deviceSphere;

		assertCUDA(cudaMalloc((void **)&deviceSphere, sizeof(SDSphere)));
		assertCUDA(cudaMemcpy(deviceSphere, this, sizeof(SDSphere), cudaMemcpyHostToDevice));

		return deviceSphere;
	}

	__host__ __device__ inline float
	distanceFromPoint(glm::vec4 point)
	{
		return GLMUtil::length(transform * point) - radius;
	}

	__host__ __device__ inline AABB
	calculateBoundingVolume()
	{
		return AABB(glm::vec2(0,0),glm::vec2(0,0)); //AABB(glm::vec2(position.x - radius, position.y - radius), glm::vec2(position.x + radius, position.y + radius));
	}

	float radius;
};

#endif
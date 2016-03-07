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
	SDSphere(float radius, glm::vec3 position) : DistancePrimitive(0), radius(radius), position(position)
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
	distanceFromPoint(glm::vec3 point)
	{
		return GLMUtil::length(point - position) - radius;
	}

	__host__ __device__ inline AABB
	calculateBoundingVolume()
	{
		return AABB(glm::vec2(position.x - radius, position.y - radius), glm::vec2(position.x + radius, position.y + radius));
	}

	float radius;
	glm::vec3 position;
};

#endif
#ifndef VALIS_SDCYLINDER_CUH
#define VALIS_SDCYLINDER_CUH

#include <glm\vec3.hpp>
#include <glm\vec2.hpp>

#include "DistancePrimitive.cuh"
#include "GLMUtil.cuh"

#include "cuda_runtime.h"
#include "CudaHelper.cuh"

class SDCylinder : public DistancePrimitive
{
public:

	__host__ __device__
		SDCylinder(glm::vec3 dimensions, glm::vec3 scale, glm::vec3 translation, glm::vec3 rotationAxis, float angle) : DistancePrimitive(3, scale, translation, rotationAxis, angle), dimensions(dimensions)
	{

	}

	__host__ __device__
		SDCylinder(glm::vec3 dimensions, glm::vec3 scale, glm::mat4 lookat) : DistancePrimitive(3, scale, lookat), dimensions(dimensions)
	{

	}

	__host__ inline DistancePrimitive*
		copyToDevice()
	{
		SDCylinder* deviceCylinder;

		assertCUDA(cudaMalloc((void **)&deviceCylinder, sizeof(SDCylinder)));
		assertCUDA(cudaMemcpy(deviceCylinder, this, sizeof(SDCylinder), cudaMemcpyHostToDevice));

		return deviceCylinder;
	}

	__host__ __device__ inline float
		distanceFromPoint(glm::vec4 point)
	{
		point = transform * point;
		return glm::length(glm::vec2(point.x, point.z) - glm::vec2(dimensions.x, dimensions.y)) - dimensions.z;
	}

	__host__ __device__ inline AABB
		calculateBoundingVolume()
	{
		return AABB(glm::vec2(0, 0), glm::vec2(0, 0));
	}

	glm::vec3 dimensions;
};

#endif
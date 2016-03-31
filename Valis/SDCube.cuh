#ifndef VALIS_SDCUBE_CUH
#define VALIS_SDCUBE_CUH

#include <glm\vec3.hpp>
#include <glm\vec2.hpp>

#include "DistancePrimitive.cuh"
#include "GLMUtil.cuh"

#include "cuda_runtime.h"
#include "CudaHelper.cuh"

class SDCube : public DistancePrimitive
{
public:

	__host__ __device__
		SDCube(glm::vec3 corner, glm::vec3 scale, glm::vec3 translation, glm::vec3 rotationAxis, float angle) : DistancePrimitive(2, scale, translation, rotationAxis, angle), corner(corner)
	{

	}

	__host__ inline DistancePrimitive*
		copyToDevice()
	{
		SDCube* deviceCube;

		assertCUDA(cudaMalloc((void **)&deviceCube, sizeof(SDCube)));
		assertCUDA(cudaMemcpy(deviceCube, this, sizeof(SDCube), cudaMemcpyHostToDevice));

		return deviceCube;
	}

	__host__ __device__ inline float
		distanceFromPoint(glm::vec4 point)
	{
		glm::vec3 d = glm::vec3(abs(point.x), abs(point.x), abs(point.x)) - corner;
		return fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.0) + GLMUtil::length(d);
	}

	__host__ __device__ inline AABB
		calculateBoundingVolume()
	{
		return AABB(glm::vec2(0, 0), glm::vec2(0, 0));
	}

	glm::vec3 corner;
};

#endif
#ifndef VALIS_SDFDEVICE_CUH
#define VALIS_SDFDEVICE_CUH

#include <thrust/device_vector.h>
#include <glm/vec3.hpp>

#include "device_launch_parameters.h"

#include "DistancePrimitive.cuh"
#include "SDModification.cuh"
#include "SDSphere.cuh"
#include "SDTorus.cuh"
#include "GLMUtil.cuh"

class SDFDevice
{
public:

	__host__
	SDFDevice(DistancePrimitive** primitives, SDModification** modifications, size_t modificationCount) : primitives(primitives), modifications(modifications), modificationCount(modificationCount)
	{

	}

	__device__ float
	distanceFromPoint(glm::vec3 position)
	{
		float distance = selectDistanceFunction(primitives[0], position);

		for (int i = 0; i < modificationCount; ++i)
		{
			float distance2 = selectDistanceFunction(primitives[i + 1], position); //COME BACK AND FIGURE OUT HOW TO INDEX PROPPERLY
			distance = selectModificationFunction(modifications[i], distance, distance2); //COME BACK AND FIGURE OUT HOW TO INDEX PROPPERLY
		}
		
		return distance;
	}

	__device__ float
	selectDistanceFunction(DistancePrimitive* primitive, glm::vec3 position)
	{
		switch (primitive->functionId)
		{
			case 0:
			{
				SDSphere *sphereCast = ((SDSphere*)primitive);

				float distance = distanceFromSphere(sphereCast->position, sphereCast->radius, position);
				return distance;
			}
			case 1:
			{
				SDTorus *torusCast = ((SDTorus*)primitive);

				float distance2 = distanceFromTorus(torusCast->position, torusCast->dimensions, position);
				return distance2;
			}
		}

		return 0.0f;
	}

	__device__ inline float
	distanceFromSphere(glm::vec3 position, float radius, glm::vec3 point)
	{
		return GLMUtil::length(point - position) - radius;
	}

	__device__ inline float
	distanceFromTorus(glm::vec3 position, glm::vec2 dimensions, glm::vec3 point)
	{
		point -= position;
		glm::vec2 q = glm::vec2(GLMUtil::length(glm::vec2(point.x, point.y)) - dimensions.x, point.z);
		return GLMUtil::length(q) - dimensions.y;
	}

	__device__ float
	selectModificationFunction(SDModification* modification, float distance1, float distance2)
	{
		switch (modification->functionId)
		{
			case 0:
				//PlaceSDPrimitive *placeCast = ((PlaceSDPrimitive*)modification);
				return placeModification(distance1, distance2);
			case 1:
				return carveModification(distance1, distance2);
		}
		return 0;
	}

	__device__ inline float
	placeModification(float originalDistance, float modifierDistance)
	{
		return fminf(originalDistance, modifierDistance);
	}

	__device__ inline float
		carveModification(float originalDistance, float modifierDistance)
	{
		return fmaxf(originalDistance, -modifierDistance);
	}

	size_t modificationCount;
	DistancePrimitive** primitives;
	SDModification** modifications;

};

#endif
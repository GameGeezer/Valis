#ifndef VALIS_SDFDEVICE_CUH
#define VALIS_SDFDEVICE_CUH

#include <thrust/device_vector.h>
#include <glm/vec3.hpp>

#include "device_launch_parameters.h"

#include "DistancePrimitive.cuh"
#include "SDModification.cuh"
#include "SDSphere.cuh"
#include "SDTorus.cuh"
#include "SDCube.cuh"
#include "SDCylinder.cuh"
#include "GLMUtil.cuh"
#include "CudaHelper.cuh"
#include "BlendSDModification.cuh"

class SDFDevice
{
public:

	__host__
	SDFDevice(DistancePrimitive** primitives, SDModification** modifications, size_t modificationCount) : primitives(primitives), modifications(modifications), modificationCount(modificationCount)
	{

	}

	__host__
	~SDFDevice()
	{

	}

	__device__ __inline__ float
	distanceFromPoint(glm::vec3 position)
	{
		float distance = selectDistanceFunction(primitives[0], position);

		for (int i = 0; i < modificationCount; ++i)
		{
			float distance2 = selectDistanceFunction(primitives[i + 1], position); 
			distance = selectModificationFunction(modifications[i], distance, distance2); 
		}
		
		return distance;
	}

	__device__ __inline__ float
	selectDistanceFunction(DistancePrimitive* primitive, glm::vec3 position)
	{
		switch (primitive->functionId)
		{
			case 0:
			{
				SDSphere *sphereCast = ((SDSphere*)primitive);

				float distance = distanceFromSphere(glm::vec4(position, 1), sphereCast->transform, sphereCast->radius);
				return distance;
			}
			case 1:
			{
				SDTorus *torusCast = ((SDTorus*)primitive);

				float distance2 = distanceFromTorus(glm::vec4(position, 1), torusCast->transform, torusCast->dimensions);
				return distance2;
			}
			case 2:
			{
				SDCube *cubeCast = ((SDCube*)primitive);

				return distanceFromCube(glm::vec4(position, 1), cubeCast->transform, cubeCast->corner);
			}
			case 3:
			{
				SDCylinder *cylinderCast = ((SDCylinder*)primitive);

				return distanceFromCylinder(glm::vec4(position, 1), cylinderCast->transform, cylinderCast->dimensions);
			}
		}

		return 0.0f;
	}

	/*
	__device__ inline float
	distanceFromSphere(glm::vec3 position, float radius, glm::vec3 point)
	{
		return GLMUtil::length(point - position) - radius;
	}
	*/

	__host__ __device__ __inline__ float
	distanceFromSphere(glm::vec4 point, glm::mat4 transform, float radius)
	{
		point = transform * point;
		return GLMUtil::length(glm::vec3(point)) - radius;
	}
	/*
	__device__ inline float
	distanceFromTorus(glm::vec3 position, glm::vec2 dimensions, glm::vec3 point)
	{
		point -= position;
		glm::vec2 q = glm::vec2(GLMUtil::length(glm::vec2(point.x, point.y)) - dimensions.x, point.z);
		return GLMUtil::length(q) - dimensions.y;
	}
	*/
	__host__ __device__ __inline__ float
		distanceFromTorus(glm::vec4 point, glm::mat4 transform, glm::vec2 dimensions)
	{
		point = transform * point;
		glm::vec2 q = glm::vec2(GLMUtil::length(glm::vec2(point.x, point.y)) - dimensions.x, point.z);
		return GLMUtil::length(q) - dimensions.y;
	}

	__host__ __device__ __inline__ float
	distanceFromCube(glm::vec4 point, glm::mat4 transform, glm::vec3 corner)
	{
		point = transform * point;
		glm::vec3 d = glm::vec3(abs(point.x), abs(point.y), abs(point.z)) - corner;
		return fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.0) + GLMUtil::length(glm::vec3(fmaxf(d.x, 0), fmaxf(d.y, 0), fmaxf(d.z, 0)));
	}

	__host__ __device__ __inline__ float
		distanceFromCylinder(glm::vec4 point, glm::mat4 transform, glm::vec3 dimensions)
	{
		point = transform * point;
		return glm::length(glm::vec2(point.x, point.z) - glm::vec2(dimensions.x, dimensions.y)) - dimensions.z;
	}

	__device__ __inline__ float
	selectModificationFunction(SDModification* modification, float distance1, float distance2)
	{
		switch (modification->functionId)
		{
			case 0:
			{
				//PlaceSDPrimitive *placeCast = ((PlaceSDPrimitive*)modification);
				return placeModification(distance1, distance2);
			}
			case 1:
			{
				return carveModification(distance1, distance2);
			}
			case 2:
			{
				BlendSDModification *blendCast = ((BlendSDModification*)modification);
				return blendModification(distance1, distance2, blendCast->smoothness);
			}
		}
		return 0;
	}

	__device__ __inline__ float
	placeModification(float originalDistance, float modifierDistance)
	{
		return fminf(originalDistance, modifierDistance);
	}

	__device__ __inline__ float
		carveModification(float originalDistance, float modifierDistance)
	{
		return fmaxf(originalDistance, -modifierDistance);
	}

	__device__ __inline__ float
	blendModification(float originalDistance, float modifierDistance, float k)
	{
		float h = glm::clamp(0.5 + 0.5*(modifierDistance - originalDistance) / k, 0.0, 1.0);
		return glm::mix(modifierDistance, originalDistance, h) - k * h * (1.0 - h);
	}




	size_t modificationCount;
	DistancePrimitive** primitives;
	SDModification** modifications;

};

#endif
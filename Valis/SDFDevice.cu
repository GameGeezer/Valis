/*
#include "SDFDevice.cuh"

__host__
SDFDevice::SDFDevice(DistancePrimitive** primitives, SDModification** modifications, size_t modificationCount) : primitives(primitives), modifications(modifications), modificationCount(modificationCount)
{

}

__host__
SDFDevice::~SDFDevice()
{

}

__device__ float
SDFDevice::distanceFromPoint(glm::vec3 position)
{
	float distance = selectDistanceFunction(primitives[0], position);

	for (int i = 0; i < modificationCount; ++i)
	{
		float distance2 = selectDistanceFunction(primitives[i + 1], position);
		distance = selectModificationFunction(modifications[i], distance, distance2);
	}

	return distance;
}

__device__ float
SDFDevice::selectDistanceFunction(DistancePrimitive* primitive, glm::vec3 position)
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
	}

	return 0.0f;
}
*/
/*
__device__ inline float
distanceFromSphere(glm::vec3 position, float radius, glm::vec3 point)
{
return GLMUtil::length(point - position) - radius;
}
*/
/*
__host__ __device__ inline float
SDFDevice::distanceFromSphere(glm::vec4 point, glm::mat4 transform, float radius)
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
/*
__host__ __device__ inline float
SDFDevice::distanceFromTorus(glm::vec4 point, glm::mat4 transform, glm::vec2 dimensions)
{
	point = transform * point;
	glm::vec2 q = glm::vec2(GLMUtil::length(glm::vec2(point.x, point.y)) - dimensions.x, point.z);
	return GLMUtil::length(q) - dimensions.y;
}

__device__ float
SDFDevice::selectModificationFunction(SDModification* modification, float distance1, float distance2)
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
SDFDevice::placeModification(float originalDistance, float modifierDistance)
{
	return fminf(originalDistance, modifierDistance);
}

__device__ inline float
SDFDevice::carveModification(float originalDistance, float modifierDistance)
{
	return fmaxf(originalDistance, -modifierDistance);
}
*/
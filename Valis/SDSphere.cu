
/*
#include "SDSphere.cuh"

#include "cuda_runtime.h"

SDSphere::SDSphere(float radius, glm::vec3 position) : radius(radius), position(position)
{

}

inline DistancePrimitive*
SDSphere::copyToDevice()
{
	SDSphere* deviceSphere;

	cudaMalloc((void **)&deviceSphere, sizeof(SDSphere));
	cudaMemcpy(deviceSphere, this, sizeof(SDSphere), cudaMemcpyHostToDevice);

	return deviceSphere;
}

inline float
SDSphere::distanceFromPoint(glm::vec3 point)
{
	return GLMUtil::length(point - position) - radius;
}

inline AABB
SDSphere::calculateBoundingVolume()
{
	return AABB(glm::vec2(position.x - radius, position.y - radius), glm::vec2(position.x + radius, position.y + radius));
}
*/
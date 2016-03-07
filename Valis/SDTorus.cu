

/*
#include "SDTorus.cuh"

#include "cuda_runtime.h"

SDTorus::SDTorus(float outer, float radius, glm::vec3 position) : dimensions(glm::vec2(outer, radius)), position(position)
{

}

inline DistancePrimitive*
SDTorus::copyToDevice()
{
	SDTorus* deviceTorus;

	cudaMalloc((void **)&deviceTorus, sizeof(SDTorus));
	cudaMemcpy(deviceTorus, this, sizeof(SDTorus), cudaMemcpyHostToDevice);

	return deviceTorus;
}

inline float
SDTorus::distanceFromPoint(glm::vec3 point)
{
	point -= position;
	glm::vec2 q = glm::vec2(GLMUtil::length(glm::vec2(point.x, point.y)) - dimensions.x, point.z);
	return GLMUtil::length(q) - dimensions.y;
}

inline AABB
SDTorus::calculateBoundingVolume()
{
	return AABB(glm::vec2(0, 0), glm::vec2(0, 0));
}
*/

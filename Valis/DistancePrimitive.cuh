
#ifndef VALIS_DISTANCE_PRIMATIVE_CUH
#define VALIS_DISTANCE_PRIMATIVE_CUH

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm\vec3.hpp>
#include <glm\mat4x4.hpp>

#include "device_launch_parameters.h"
#include "AABB.cuh"

class DistancePrimitive
{
public:
	const int functionId;
	glm::mat4 transform;
	DistancePrimitive(int functionId, glm::vec3 scale, glm::vec3 translation, glm::vec3 rotationAxis, float angle) : functionId(functionId)
	{
		transform = glm::inverse(glm::translate(glm::mat4(1.0f), translation) * glm::rotate(glm::mat4(1.0f), angle, rotationAxis) * glm::scale(glm::mat4(1.0f),  scale));
	}

	__host__ virtual inline DistancePrimitive*
	copyToDevice() = 0;

	__host__ __device__ virtual inline float
	distanceFromPoint(glm::vec4 point) = 0;

	__host__ __device__ virtual inline AABB
	calculateBoundingVolume() = 0;
	
};

#endif



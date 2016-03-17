#ifndef GLMUTIL_CUH
#define GLMUTIL_CUH

#include <glm\glm.hpp>
#include <glm\vec3.hpp>

#include "device_launch_parameters.h"

class GLMUtil
{
public:

	__host__ __device__
	static inline float
	length(glm::vec2 vector)
	{
		return sqrt(vector.x * vector.x + vector.y * vector.y);
	}

	__host__ __device__ 
	static inline float
	length(glm::vec3 vector)
	{
		return sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
	}

	__host__ __device__
	static inline float
	length(glm::vec4 vector)
	{
		return sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z + vector.w * vector.w);
	}
};

#endif
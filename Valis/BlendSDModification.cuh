#ifndef VALIS_BLEND_SD_MODIFICATION_CUH
#define VALIS_BLEND_SD_MODIFICATION_CUH

#include "cuda_runtime.h"

#include "SDModification.cuh"

#include <glm/glm.hpp>

class BlendSDModification : public SDModification
{
public:
	float smoothness;


	__host__ __device__
		BlendSDModification(float smoothness) : SDModification(2), smoothness(smoothness)
	{

	}

	__host__ inline SDModification*
		copyToDevice()
	{
		BlendSDModification* deviceMod;

		cudaMalloc((void **)&deviceMod, sizeof(BlendSDModification));
		cudaMemcpy(deviceMod, this, sizeof(BlendSDModification), cudaMemcpyHostToDevice);

		return deviceMod;
	}

	__host__ __device__ inline float
		modify(float originalDistance, float modifierDistance, float k)
	{
		float h = glm::clamp(0.5 + 0.5*(modifierDistance - originalDistance) / k, 0.0, 1.0);
		return glm::mix(modifierDistance, originalDistance, h) - k * h * (1.0 - h);
	}
};

#endif
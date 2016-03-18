#ifndef VALIS_BLEND_SD_MODIFICATION_CUH
#define VALIS_BLEND_SD_MODIFICATION_CUH

#include "cuda_runtime.h"

#include "SDModification.cuh"

class BlendSDModification : public SDModification
{
public:

	__host__ __device__
		BlendSDModification() : SDModification(1)
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
		modify(float originalDistance, float modifierDistance)
	{
		return fmaxf(originalDistance, -modifierDistance);
	}
};

#endif
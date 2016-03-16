#ifndef VALIS_CARVE_SD_PRIMITIVE_CUH
#define VALIS_CARVE_SD_PRIMITIVE_CUH

#include "cuda_runtime.h"

#include "SDModification.cuh"

class CarveSDPrimitive : public SDModification
{
public:

	__host__ __device__
		CarveSDPrimitive() : SDModification(1)
	{

	}

	__host__ inline SDModification*
	copyToDevice()
	{
		CarveSDPrimitive* deviceMod;

		cudaMalloc((void **)&deviceMod, sizeof(CarveSDPrimitive));
		cudaMemcpy(deviceMod, this, sizeof(CarveSDPrimitive), cudaMemcpyHostToDevice);

		return deviceMod;
	}

	__host__ __device__ inline float
		modify(float originalDistance, float modifierDistance)
	{
		return fmaxf(originalDistance, -modifierDistance);
	}
};

#endif
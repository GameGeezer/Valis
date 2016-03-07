#ifndef VALIS_PLACE_SD_PRIMITIVE_CUH
#define VALIS_PLACE_SD_PRIMITIVE_CUH

#include "cuda_runtime.h"

#include "SDModification.cuh"

class PlaceSDPrimitive : public SDModification
{
public:

	__host__ __device__
	PlaceSDPrimitive() : SDModification(0)
	{

	}

	__host__ inline SDModification*
	copyToDevice()
	{
		PlaceSDPrimitive* deviceMod;

		cudaMalloc((void **)&deviceMod, sizeof(PlaceSDPrimitive));
		cudaMemcpy(deviceMod, this, sizeof(PlaceSDPrimitive), cudaMemcpyHostToDevice);

		return deviceMod;
	}

	__host__ __device__ inline float
	modify(float originalDistance, float modifierDistance)
	{
		return fminf(originalDistance, modifierDistance);
	}
};

#endif
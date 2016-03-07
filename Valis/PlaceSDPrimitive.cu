/*

#include "PlaceSDPrimitive.cuh"

inline SDModification*
PlaceSDPrimitive::copyToDevice()
{
	PlaceSDPrimitive* deviceMod;

	cudaMalloc((void **)&deviceMod, sizeof(PlaceSDPrimitive));
	cudaMemcpy(deviceMod, this, sizeof(PlaceSDPrimitive), cudaMemcpyHostToDevice);

	return deviceMod;
}

inline float
PlaceSDPrimitive::modify(float originalDistance, float modifierDistance)
{
	fminf(originalDistance, modifierDistance);
}

*/
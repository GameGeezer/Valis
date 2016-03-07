#include "SDFHost.cuh"

#include "SDFDevice.cuh"

#include "CudaHelper.cuh"

SDFHost::SDFHost(DistancePrimitive *primative) : host_primitives(*(new thrust::host_vector<DistancePrimitive*>())), host_modifications(*(new thrust::host_vector<SDModification*>())), device_primitives(*(new thrust::device_vector<DistancePrimitive*>())), device_modifications(*(new thrust::device_vector<SDModification*>()))
{
	DistancePrimitive* devicePrimitive = primative->copyToDevice();

	host_primitives.push_back(devicePrimitive);
}

SDFDevice*
SDFHost::copyToDevice()
{
	device_primitives = host_primitives;
	device_modifications = host_modifications;

	DistancePrimitive** primitivesBegin = thrust::raw_pointer_cast(device_primitives.data());
	SDModification** modificationsBegin = thrust::raw_pointer_cast(device_modifications.data());

	SDFDevice sdfInfo(primitivesBegin, modificationsBegin, modificationCount);

	SDFDevice* deviceSDF;

	assertCUDA(cudaMalloc((void **)&deviceSDF, sizeof(SDFDevice)));
	assertCUDA(cudaMemcpy(deviceSDF, &sdfInfo, sizeof(SDFDevice), cudaMemcpyHostToDevice));

	return deviceSDF;
}

void
SDFHost::modify(DistancePrimitive *primative, SDModification *modification)
{
	DistancePrimitive* devicePrimitive = primative->copyToDevice();
	SDModification* deviceModification = modification->copyToDevice();

	host_primitives.push_back(devicePrimitive);
	host_modifications.push_back(deviceModification);

	++modificationCount;
}
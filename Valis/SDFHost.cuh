#ifndef VALIS_SDFHOST_CUH
#define VALIS_SDFHOST_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <glm/vec3.hpp>

#include "device_launch_parameters.h"

#include "DistancePrimitive.cuh"
#include "SDModification.cuh"

#include "SDFDevice.cuh"

class SDFHost
{
public:

	__host__
	SDFHost(DistancePrimitive* primative, size_t editCapacity);

	__host__ SDFDevice*
	copyToDevice();

	__host__ void
	modify(DistancePrimitive* primative, SDModification* modification);

	__host__ void
	popEdit();

	//void
	//normalize();

private:
	SDFDevice* deviceSDF;
	AABB bounds;
	size_t modificationCount = 0;
	thrust::host_vector<DistancePrimitive*>& host_primitives;
	thrust::host_vector<SDModification*>& host_modifications;
	thrust::device_vector<DistancePrimitive*>& device_primitives;
	thrust::device_vector<SDModification*>& device_modifications;
};

#endif
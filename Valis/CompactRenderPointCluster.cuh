#ifndef VALIS_COMPACT_RENDER_POINT_CLUSTER_CUH
#define VALIS_COMPACT_RENDER_POINT_CLUSTER_CUH

#include <stdint.h>

#include "device_launch_parameters.h"

#include "CompactRenderPoint.cuh"

class CompactRenderPointCluster
{
public:
	uint32_t x, y, z, pointsAdded;
	CompactRenderPoint renderPoints[262144];
};

#endif
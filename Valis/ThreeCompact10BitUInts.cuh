#ifndef COMPACT_LOCATION_CUH
#define COMPACT_LOCATION_CUH

#include <stdint.h>

#include "device_launch_parameters.h"

class ThreeCompact10BitUInts
{
public:
	uint32_t compactData;

	__device__ __host__ __inline__
		void pack(uint32_t x, uint32_t y, uint32_t z)
	{
		compactData = x | (y << 10) | (z << 20);
	}

	__device__ __host__ __inline__
	void unpack(uint32_t& out_x, uint32_t& out_y, uint32_t& out_z)
	{
		out_x = compactData & 0x3FF;
		out_y = (compactData & 0xFFC00) >> 10;
		out_z = (compactData & 0x3FF00000) >> 20;
	}
};

#endif
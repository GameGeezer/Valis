#ifndef VALIS_COMPACT_MORTON_POINT_CUH
#define VALIS_COMPACT_MORTON_POINT_CUH

#include <stdint.h>

#include "device_launch_parameters.h"

class CompactMortonPoint
{
public:
	__device__ __host__ __inline__
	void pack(uint32_t morton, uint32_t nx, uint32_t ny, uint32_t nz)
	{
		compactData = (morton & 0x3FFFF) | (nx << 18) | (ny << 21) | (nz << 24);
	}

	__device__ __host__ __inline__
	void unpack(uint32_t& out_morton, uint32_t& out_nx, uint32_t& out_ny, uint32_t& out_nz)
	{
		out_morton = compactData & 0x3FFFF;

		out_nx = (compactData & 0x1C0000) >> 18;
		out_ny = (compactData & 0xE00000) >> 21;
		out_nz = (compactData & 0x7000000) >> 24;
	}

	uint32_t compactData;
private:

};

#endif
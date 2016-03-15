#ifndef VALIS_COMPACT_RENDER_POINT_CUH
#define VALIS_COMPACT_RENDER_POINT_CUH

#include <stdint.h>

#include "device_launch_parameters.h"

class CompactRenderPoint
{
public:
	__device__ __host__ __inline__
		void pack(uint32_t x, uint32_t y, uint32_t z, uint32_t nx, uint32_t ny, uint32_t nz)
	{
		compactData = x | (y << 6) | (z << 12) | (nx << 18) | (ny << 21) | (nz << 24);
	}

	__device__ __host__ __inline__
		void unpack(uint32_t& out_x, uint32_t& out_y, uint32_t& out_z, uint32_t& out_nx, uint32_t& out_ny, uint32_t& out_nz)
	{
		out_x = compactData & 0x3F;
		out_y = (compactData & 0xFC0) >> 6;
		out_z = (compactData & 0x3F000) >> 12;

		out_nx = (compactData & 0x1C0000) >> 18;
		out_ny = (compactData & 0xE00000) >> 21;
		out_nz = (compactData & 0x7000000) >> 24;
	}

	uint32_t compactData;
private:
	
};

#endif
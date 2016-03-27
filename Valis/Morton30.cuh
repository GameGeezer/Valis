#ifndef VALIS_MORTON30_CUH
#define VALIS_MORTON30_CUH

#include <stdint.h>

#include "device_launch_parameters.h"

class Morton30
{
public:

	__host__ __device__ static uint32_t
	encode(uint32_t x, uint32_t y, uint32_t z)
	{
		return splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2);
	}

	__host__ __device__ static void 
	decode(uint32_t morton, uint32_t& x, uint32_t& y, uint32_t& z)
	{
		x = compactBy3(morton);
		y = compactBy3(morton >> 1);
		z = compactBy3(morton >> 2);
	}

private:

	__host__ __device__ __inline__ static uint32_t
	splitBy3(uint32_t value)
	{
		value &= 0x000003ff;
		value |= (value << 16);
		value &= 0x030000ff;
		value |= (value << 8);
		value &= 0x0300f00f;
		value |= (value << 4);
		value &= 0x030c30c3;
		value |= (value << 2);
		value &= 0x09249249;

		return value;
	}

	__host__ __device__ __inline__ static uint32_t
	compactBy3(uint32_t value)
	{
		value &= 0x09249249;
		value |= (value >> 2);
		value &= 0x030c30c3;
		value |= (value >> 4);
		value &= 0x0300f00f;
		value |= (value >> 8);
		value &= 0x030000ff;
		value |= (value >> 16);
		value &= 0x000003ff;

		return value;
	}
};

#endif
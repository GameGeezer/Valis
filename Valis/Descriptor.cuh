#ifndef VALIS_DESCRIPTOR_CUH
#define VALIS_DESCRIPTOR_CUH

#include <stdint.h>

class Descriptor
{
public:

	const int32_t size, type, stride, pointer;
	const bool normalized;

	Descriptor(int32_t size, int32_t type, bool normalized, int32_t stride, int32_t pointer);
};

#endif
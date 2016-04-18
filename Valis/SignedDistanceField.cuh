#ifndef VALIS_SIGNED_DISTANCE_FIELD_CUH
#define VALIS_SIGNED_DISTANCE_FIELD_CUH

#include <stdint.h>

#define SDF_INSIDE_SURFACE 1
#define SDF_OUTSIDE_SURFACE 0

#define SDF_THREAD_BLOCK_DIM 8

class ByteArray;
class ByteArrayChunk;
class DistancePrimitive;
class SDModification;

class SignedDistanceField
{

public:

	SignedDistanceField(uint32_t gridResolution);

	void
	place(DistancePrimitive& primitive, uint32_t material);

	void
	copyInto(SignedDistanceField& other);

	ByteArrayChunk*
	getMaterialDevicePointer();

private:

	dim3 materialBlockSize, materialThreadSize;
	uint32_t gridResolution;
	ByteArray &materialGrid, &normalGrid;
};

#endif
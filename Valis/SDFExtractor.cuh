#ifndef VALIS_SDF_EXTRACTOR_CUH
#define VALIS_SDF_EXTRACTOR_CUH

#include <stdint.h>

#include <thrust/device_vector.h>

#include "ThreeCompact10BitUInts.cuh"

class Nova;

typedef ThreeCompact10BitUInts CompactNormals;

struct ExtractedPoint
{
	uint32_t morton;
	CompactNormals normals;
};

class SDFExtractor
{
public:

	SDFExtractor(uint32_t gridDimension, uint32_t parseDimension);

	~SDFExtractor();

	size_t
	extract(Nova* nova);

private:

	thrust::device_vector< uint32_t >* cornerIntersectionBuffer;
};
#endif
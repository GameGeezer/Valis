#ifndef VALIS_SDF_EXTRACTOR_CUH
#define VALIS_SDF_EXTRACTOR_CUH

#include <stdint.h>

#include <thrust/device_vector.h>

class Nova;

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
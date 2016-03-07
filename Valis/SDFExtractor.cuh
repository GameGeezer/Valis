#ifndef VALIS_SDFEXTRACTOR
#define VALIS_SDFEXTRACTOR

#include <thrust/device_vector.h>
#include <stdint.h>

class RenderPoint;
class ExtractionBlock;

class SDFExtractor
{
public:

	SDFExtractor();

	thrust::device_vector< RenderPoint >*
	extract();

private:
	thrust::device_vector< ExtractionBlock >* pointCoverageBuffer;
	thrust::device_vector< uint32_t >* materialCoverageBuffer;
	thrust::device_vector< RenderPoint >* partialExtractionBuffer;

	thrust::device_vector< RenderPoint >* extractedPoints;

	uint32_t gridResolution = 400;
	uint32_t partialExtractionSize = 200;
};


#endif
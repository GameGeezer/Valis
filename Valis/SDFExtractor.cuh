#ifndef VALIS_SDFEXTRACTOR
#define VALIS_SDFEXTRACTOR

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdint.h>

class RenderPoint;
class ExtractionBlock;
class SDFDevice;

class SDFExtractor
{
public:

	SDFExtractor(uint32_t clusterDensity, uint32_t extractionClusterDensity);

	thrust::host_vector< RenderPoint >*
	extract(SDFDevice& sdf);

private:
	thrust::device_vector< ExtractionBlock >* pointCoverageBuffer;
	thrust::device_vector< ExtractionBlock >* materialCoverageBuffer;
	thrust::device_vector< RenderPoint >* partialExtractionBuffer;

	uint32_t clusterDensity, extractionClusterDensity;
};


#endif
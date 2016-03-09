#ifndef VALIS_SDFEXTRACTOR
#define VALIS_SDFEXTRACTOR

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <stdint.h>

class RenderPoint;
class ExtractionBlock;
class SDFDevice;
class VBO;

class SDFExtractor
{
public:

	SDFExtractor(uint32_t clusterDensity, uint32_t extractionClusterDensity);

	~SDFExtractor();

	thrust::host_vector< RenderPoint >*
	extract(SDFDevice& sdf);

	size_t
	extractDynamic(SDFDevice& sdf, VBO& vbo);

private:

	void
	extractCoverageBuffer(thrust::device_vector< ExtractionBlock >& buffer, SDFDevice& sdf);

	thrust::device_vector< ExtractionBlock >* pointCoverageBuffer;
	thrust::device_vector< ExtractionBlock >* materialCoverageBuffer;
	thrust::device_vector< RenderPoint >* partialExtractionBuffer;

	uint32_t clusterDensity, extractionClusterDensity;
	dim3 coverageExtractBlockDim, partialExtractionBlockDim, parseThreadsDim;
};


#endif
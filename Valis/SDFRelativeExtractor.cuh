#ifndef VALIS_SDF_RELATIVE_EXTRACTOR
#define VALIS_SDF_RELATIVE_EXTRACTOR

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "CudaGLBufferMapping.cuh"
#include "RenderPoint.cuh"
#include "CompactRenderPoint.cuh"

#include <stdint.h>

class ExtractionBlock;
class SDFDevice;
class VBO;
class PBO;

class SDFRelativeExtractor
{
public:

	SDFRelativeExtractor(uint32_t clusterDensity, uint32_t extractionClusterDensity);

	~SDFRelativeExtractor();

	size_t
		extract(SDFDevice& sdf, CudaGLBufferMapping<CompactRenderPoint>& mapping, PBO& pbo);

private:


	thrust::device_vector< CompactRenderPoint >* partialExtractionBuffer;

	uint32_t clusterDensity, extractionClusterDensity;
	dim3 coverageExtractBlockDim, partialExtractionBlockDim, parseThreadsDim;
};

#endif
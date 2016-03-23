#ifndef VALIS_SDF_HILBERT_EXTRACTOR_CUH
#define VALIS_SDF_HILBERT_EXTRACTOR_CUH

#include <stdint.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "CudaGLBufferMapping.cuh"
#include "ThreeCompact10BitUInts.cuh"
#include "CompactRenderPoint.cuh"

class SDFDevice;

typedef ThreeCompact10BitUInts CompactLocation;
typedef ThreeCompact10BitUInts CompactNormals;

struct ExtractedPoint
{
	CompactLocation location;
	CompactNormals normals;
};

class SDFHilbertExtractor
{
public:

	SDFHilbertExtractor(uint32_t gridDimension, uint32_t parseDimension);

	~SDFHilbertExtractor();

	size_t
	extract(SDFDevice& sdf, CudaGLBufferMapping<CompactRenderPoint>& mapping, CudaGLBufferMapping<CompactLocation>& pbo);

private:
	thrust::device_vector< uint32_t >* areVerticiesOutsideIsoBuffer;
	thrust::device_vector< ExtractedPoint >* mortonSortedPointsBuffer;

	dim3 extractInMortonOrderBlockDim, extractInMortonOrderThreadDim;
	uint32_t gridDimension, parseDimension;
};
#endif
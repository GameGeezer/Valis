#ifndef VALIS_SDF_HILBERT_EXTRACTOR_CUH
#define VALIS_SDF_HILBERT_EXTRACTOR_CUH

#include <stdint.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ThreeCompact10BitUInts.cuh"
#include "CompactMortonPoint.cuh"

class SDFDevice;
class Nova;

typedef uint32_t WorldPositionMorton;
typedef ThreeCompact10BitUInts CompactNormals;

struct ExtractedPoint
{
	uint32_t spare;
	uint32_t morton;
	CompactNormals normals;
};

class SDFHilbertExtractor
{
public:

	SDFHilbertExtractor(uint32_t gridDimension, uint32_t parseDimension);

	~SDFHilbertExtractor();

	size_t
		extract(Nova &nova, uint32_t overlapSize);

private:
	thrust::device_vector< uint32_t >* areVerticiesOutsideIsoBuffer;
	thrust::device_vector< ExtractedPoint >* mortonSortedPointsBuffer, *mortonSortedPointsCompactBuffer;

	dim3 extractInMortonOrderBlockDim, extractInMortonOrderThreadDim;
	uint32_t gridDimension, parseDimension, mortonSortedPointBlockSize, mortonSortedPointThreadSize;

	uint32_t *device_worldMortonSizeBucket, *device_compactMortonSizeBucket, *device_indexSizeBucket;
};
#endif
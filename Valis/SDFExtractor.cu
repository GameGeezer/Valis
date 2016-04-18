#include "SDFExtractor.cuh"

#include "cuda_runtime.h"

#include "ByteArray.cuh"
#include "SignedDistanceField.cuh"
#include "Morton30.cuh"
/*
__global__ void
sdfExtractPointsInMortonOrder(ExtractedPoint *d_output, ByteArrayChunk *sdfData, uint32_t *vertexPlacement, uint32_t gridResolution, uint32_t parseDimension, uint32_t dimensionOffsetX, uint32_t dimensionOffsetY, uint32_t dimensionOffsetZ)
{

	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	uint32_t offsetX = x + dimensionOffsetX;
	uint32_t offsetY = y + dimensionOffsetY;
	uint32_t offsetZ = z + dimensionOffsetZ;

	if (x >= parseDimension || y >= parseDimension || z >= parseDimension || offsetX >= gridResolution || offsetY >= gridResolution || offsetZ >= gridResolution)
	{
		return;
	}

	uint32_t materialIndex = offsetX + offsetY * gridResolution + offsetZ * gridResolution * gridResolution;
	uint32_t material = byteArray_getValueAtIndex(sdfData, materialIndex);

	NumericBoolean shouldGeneratePoint = numericNotEqual_uint32_t(material, SDF_INSIDE_SURFACE) * numericNotEqual_uint32_t(material, SDF_OUTSIDE_SURFACE);

	uint32_t index = x + y * gridResolution + z * gridResolution * gridResolution;

	//d_output[index].normals.pack(normalVec.x, normalVec.y, normalVec.z);
	d_output[index].normals.compactData *= shouldGeneratePoint;
	d_output[index].morton = Morton30::encode(offsetX, offsetY, offsetZ);
	d_output[index].morton *= shouldGeneratePoint;
}

*/
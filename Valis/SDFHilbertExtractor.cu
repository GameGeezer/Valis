#include "SDFHilbertExtractor.cuh"

#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "cuda_runtime.h"

#include "NumericBoolean.cuh"

#include "SDFDevice.cuh"
#include "Morton.cuh"


__device__ __inline__ uint32_t
findIndex(uint32_t localX, uint32_t localY, uint32_t localZ, uint32_t parseDimension)
{
	uint32_t index = localX + localY * parseDimension + localZ * parseDimension * parseDimension;
}

__global__ void
areVerticesOutsideIsosurface(uint32_t *d_output, SDFDevice *sdf, uint32_t gridDimension, uint32_t parseDimension, uint32_t dimensionOffsetX, uint32_t dimensionOffsetY, uint32_t dimensionOffsetZ)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	// Check to see if x, y, or z exceeds the bounds of the local grid
	if (x >= parseDimension || y >= parseDimension || z >= parseDimension)
	{
		return;
	}

	uint32_t offsetX = x + dimensionOffsetX;
	uint32_t offsetY = y + dimensionOffsetY;
	uint32_t offsetZ = z + dimensionOffsetZ;

	// Check to see if x, y, ot z exceed the bounds of the entire grid
	if (offsetX >= gridDimension || offsetY >= gridDimension || offsetZ >= gridDimension)
	{
		return;
	}

	uint64_t index = findIndex(x, y, z, parseDimension);

	float divisionsAsFloat = ((float)gridDimension);
	float halfCellDimension = 0.5f / divisionsAsFloat;
	// normalized x, y, and z
	float normalizeX = (((float)offsetX) / divisionsAsFloat) - halfCellDimension;
	float normalizeY = (((float)offsetY) / divisionsAsFloat) - halfCellDimension;
	float normalizeZ = (((float)offsetZ) / divisionsAsFloat) - halfCellDimension;

	// How far the cell is from the sdf
	float distance = sdf->distanceFromPoint(glm::vec3(normalizeX, normalizeY, normalizeZ));

	NumericBoolean isDistancePositive = numericGreaterThan_float(distance, 0);

	d_output[index] = isDistancePositive;
}

__global__ void 
extractPointsInMortonOrder(ExtractedPoint *d_output, SDFDevice *sdf, uint32_t *vertexPlacement, uint32_t gridDimension, uint32_t parseDimension, uint32_t dimensionOffsetX, uint32_t dimensionOffsetY, uint32_t dimensionOffsetZ)
{

	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	// Check to see if x, y, or z exceeds the bounds of the local grid
	if (x >= parseDimension || y >= parseDimension || z >= parseDimension)
	{
		return;
	}

	uint32_t offsetX = x + dimensionOffsetX;
	uint32_t offsetY = y + dimensionOffsetY;
	uint32_t offsetZ = z + dimensionOffsetZ;

	// Check to see if x, y, ot z exceed the bounds of the entire grid
	if (offsetX >= gridDimension || offsetY >= gridDimension || offsetZ >= gridDimension)
	{
		return;
	}

	uint64_t index = Morton::encode(x, y, z);
	
	float divisionsAsFloat = ((float)gridDimension);

	// normalized x, y, and z
	float normalizeX = ((float)offsetX) / divisionsAsFloat;
	float normalizeY = ((float)offsetY) / divisionsAsFloat;
	float normalizeZ = ((float)offsetZ) / divisionsAsFloat;

	// How far the cell is from the sdf
	float distance = sdf->distanceFromPoint(glm::vec3(normalizeX, normalizeY, normalizeZ));

	// Decide whether to generate a point
	float cellDimension = 1.0f / divisionsAsFloat;

	/*
	NumericBoolean bottomLeftBack = vertexPlacement[findIndex(x, y, z, parseDimension)];
	NumericBoolean bottomRightBack = vertexPlacement[findIndex(x + 1, y, z, parseDimension)];
	NumericBoolean topLeftBack = vertexPlacement[findIndex(x, y + 1, z, parseDimension)];
	NumericBoolean bottomLeftForward = vertexPlacement[findIndex(x, y, z + 1, parseDimension)];
	NumericBoolean topRightBack = vertexPlacement[findIndex(x + 1, y + 1, z, parseDimension)];
	NumericBoolean bottomRightForward = vertexPlacement[findIndex(x + 1, y, z + 1, parseDimension)];
	NumericBoolean topLeftForward = vertexPlacement[findIndex(x, y + 1, z + 1, parseDimension)];
	NumericBoolean topRightForward = vertexPlacement[findIndex(x + 1, y + 1, z + 1, parseDimension)];

	glm::vec3 bottomLeftBackVec = glm::vec3(bottomLeftBack, bottomLeftBack, bottomLeftBack);
	glm::vec3 bottomRightBackVec = glm::vec3(-bottomRightBack, bottomRightBack, bottomRightBack);
	glm::vec3 topLeftBackVec = glm::vec3(topLeftBack, -topLeftBack, topLeftBack);
	glm::vec3 bottomLeftForwardVec = glm::vec3(bottomLeftForward, bottomLeftForward, -bottomLeftForward);
	glm::vec3 topRightBackVec = glm::vec3(-topRightBack, -topRightBack, topRightBack);
	glm::vec3 bottomRightForwardVec = glm::vec3(-bottomRightForward, bottomRightForward, -bottomRightForward);
	glm::vec3 topLeftForwardVec = glm::vec3(topLeftForward, -topLeftForward, -topLeftForward);
	glm::vec3 topRightForwardVec = glm::vec3(-topRightForward, -topRightForward, -topRightForward);
	
	glm::vec3 normalAsFloat = bottomLeftBackVec + bottomRightBackVec + topLeftBackVec + bottomLeftForwardVec + topRightBackVec + bottomRightForwardVec + topLeftForwardVec + topRightForwardVec;
	*/
	NumericBoolean shouldGeneratePoint = numericLessThan_float(distance, cellDimension) * numericGreaterThan_float(distance, 0);

	//d_output[index].point.pack(localX * shouldGeneratePoint, localY * shouldGeneratePoint, localZ * shouldGeneratePoint, 0, 0, 0);
	//d_output[index].location.pack(offsetX * shouldGeneratePoint, offsetY * shouldGeneratePoint, offsetZ * shouldGeneratePoint);
}

__global__ void 
clusterPoints(CompactRenderPoint* renderPointBuffer, CompactLocation* offsetBuffer, ExtractedPoint* sortedPoints, uint32_t overlapSize)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t renderPointBufferWriteIndex = x * 64;
	uint32_t startingIndex = renderPointBufferWriteIndex - (overlapSize * x);
	
	uint32_t offsetX, offsetY, offsetZ;
	sortedPoints[startingIndex].location.unpack(offsetX, offsetY, offsetZ);

	for (uint32_t i = 0; i < 64; ++i)
	{
		uint32_t pointX, pointY, pointZ;
		sortedPoints[startingIndex + i].location.unpack(pointX, pointY, pointZ);
		pointX -= offsetX;
		pointY -= offsetY;
		pointZ -= offsetZ;
		NumericBoolean isXWithingBounds = numericLessThan_uint32_t(pointX, 64);
		NumericBoolean isYWithingBounds = numericLessThan_uint32_t(pointY, 64);
		NumericBoolean isZWithingBounds = numericLessThan_uint32_t(pointZ, 64);

		NumericBoolean isLegal = isXWithingBounds * isYWithingBounds * isZWithingBounds;

		uint32_t normalX, normalY, normalZ;
		sortedPoints[startingIndex + i].normals.unpack(normalX, normalY, normalZ);

		renderPointBuffer[renderPointBufferWriteIndex].pack(pointX, pointY, pointZ, normalX, normalY, normalZ);
	}

	offsetBuffer[x].compactData = sortedPoints[startingIndex].location.compactData;
}


struct is_extracted_not_zero
{
	__host__ __device__
		bool operator()(const ExtractedPoint& point)
	{
		return point.location.compactData != 0;
	}
};

struct is_uint32_t_not_zero
{
	__host__ __device__
		bool operator()(const uint32_t& point)
	{
		return point != 0;
	}
};

SDFHilbertExtractor::SDFHilbertExtractor(uint32_t gridDimension, uint32_t parseDimension) :
	gridDimension(gridDimension),
	parseDimension(parseDimension),
	extractInMortonOrderBlockDim(parseDimension / 8, parseDimension / 8, parseDimension / 8),
	extractInMortonOrderThreadDim(8, 8, 8),
	mortonSortedPointsBuffer(new thrust::device_vector< ExtractedPoint >(parseDimension * parseDimension * parseDimension)),
	areVerticiesOutsideIsoBuffer(new thrust::device_vector< uint32_t >((parseDimension + 2) * (parseDimension + 2) * (parseDimension + 2)))
{

}

size_t
SDFHilbertExtractor::extract(SDFDevice& sdf, CudaGLBufferMapping<CompactRenderPoint>& mapping, CudaGLBufferMapping<CompactLocation>& pbo)
{
	mapping.map();
	size_t bufferLength = mapping.getSizeInBytes() / sizeof(CompactRenderPoint);
	CompactRenderPoint* bufferPointerRaw = thrust::raw_pointer_cast(mapping.getDeviceOutput());
	thrust::device_ptr<CompactRenderPoint> bufferPointerDevice = thrust::device_pointer_cast(mapping.getDeviceOutput());

	pbo.map();
	size_t pboBufferLength = pbo.getSizeInBytes() / sizeof(ThreeCompact10BitUInts);
	ThreeCompact10BitUInts* pboBufferPointerRaw = thrust::raw_pointer_cast(pbo.getDeviceOutput());
	thrust::device_ptr<ThreeCompact10BitUInts> pboBufferPointerDevice = thrust::device_pointer_cast(pbo.getDeviceOutput());

	// Point to the partial extraction buffer
	ExtractedPoint* mortenSortedPointsRaw = thrust::raw_pointer_cast(mortonSortedPointsBuffer->data());
	uint32_t* isVertexOusideIsoBufferRaw = thrust::raw_pointer_cast(areVerticiesOutsideIsoBuffer->data());

	thrust::device_ptr<ExtractedPoint> mortenSortedPointsDevice = thrust::device_pointer_cast(mortonSortedPointsBuffer->data());
	thrust::device_ptr<uint32_t> isVertexOusideIsoBufferDevice = thrust::device_pointer_cast(areVerticiesOutsideIsoBuffer->data());

	for (int i = 0; i < gridDimension; i += parseDimension)
	{
		for (int j = 0; j < gridDimension; j += parseDimension)
		{
			for (int k = 0; k < gridDimension; k += parseDimension)
			{
				extractPointsInMortonOrder << <extractInMortonOrderBlockDim, extractInMortonOrderThreadDim >> >(mortenSortedPointsRaw, &sdf, isVertexOusideIsoBufferRaw, gridDimension, parseDimension, i, j, k);
				cudaDeviceSynchronize();
				thrust::stable_partition(thrust::device, mortonSortedPointsBuffer->begin(), mortonSortedPointsBuffer->end(), mortonSortedPointsBuffer->begin(), is_extracted_not_zero());
			}
		}
	}

	mapping.unmap();
	pbo.unmap();


	return 0;
}
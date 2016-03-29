#include "SDFHilbertExtractor.cuh"

#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "cuda_runtime.h"

#include "NumericBoolean.cuh"

#include "SDFDevice.cuh"
#include "Morton30.cuh"


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

	uint32_t index = Morton30::encode(x, y, z);
	
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
	d_output[index].morton = Morton30::encode(offsetX , offsetY, offsetZ);
	d_output[index].morton *= shouldGeneratePoint;
}

__global__ void 
clusterPoints(CompactMortonPoint* renderPointBuffer, WorldPositionMorton* offsetBuffer, ExtractedPoint* sortedPoints, size_t overlapSize, CompactMortonPoint* renderPointBufferEnd, WorldPositionMorton* offsetBufferEnd, ExtractedPoint* sortedPointsEnd)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t renderPointBufferWriteIndex = x * 64;
	uint32_t startingIndex = renderPointBufferWriteIndex - (overlapSize * x);

	if (((offsetBuffer + x) >= offsetBufferEnd))
	{
		int d = x + 5;
		return;
	}

	if ( ((sortedPoints + startingIndex + 64) >= sortedPointsEnd) )
	{
		int d = x + 5;
		return;
	}

	if (((renderPointBuffer + renderPointBufferWriteIndex) >= renderPointBufferEnd))
	{
		int d = x + 5;
		return;
	}

	if (sortedPoints[startingIndex].morton == 0)
	{
		return;
	}

	uint64_t baseMorton = sortedPoints[startingIndex].morton;

	for (uint32_t i = 0; i < 64; ++i)
	{
		uint64_t upperMorton = sortedPoints[startingIndex + i].morton;
		uint64_t highestDifferent = Morton30::highestOrderBitDifferent(baseMorton, upperMorton);
		// Make sure the highest bit different is less than x is 64
		NumericBoolean isWithingBounds = numericLessThan_uint32_t(highestDifferent, 0x40000);
		// Subtract the base morton from the upper morton so relative offset can be stored
		uint32_t mortonOffset = (upperMorton - baseMorton);

		// unpack the normals
		uint32_t normalX, normalY, normalZ;
		sortedPoints[startingIndex + i].normals.unpack(normalX, normalY, normalZ);

		renderPointBuffer[renderPointBufferWriteIndex + i].pack(mortonOffset, 1, normalY, normalZ);// ONE BECAUSE SELECTING THE NEXT BUFFER INDEX RELIES ON A NORMALIZED NORMALS
		renderPointBuffer[renderPointBufferWriteIndex + i].compactData = renderPointBuffer[renderPointBufferWriteIndex + i].compactData * isWithingBounds + baseMorton * numericNegate_uint32_t(isWithingBounds);
	}

	offsetBuffer[x] = baseMorton;
}

__global__ void
findEndOfWrittenCompactMortons(CompactMortonPoint* bufferBegin, size_t bufferLength, CompactMortonPoint* out_bufferDataEnd)
{
	uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x);
	// One higher than index x is being checked
	if (x >= (bufferLength - 1))
	{
		return;
	}
	
	bool lowerHasValue = bufferBegin[x].compactData != 0;
	bool upperHasValue = bufferBegin[x + 1].compactData != 0;
	
	if (lowerHasValue && !upperHasValue)
	{
		out_bufferDataEnd = &(bufferBegin[x]);
	}
}

__global__ void
findEndOfWrittenWorldMortons(WorldPositionMorton* bufferBegin, size_t bufferLength, WorldPositionMorton* out_bufferDataEnd)
{
	uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x);
	// One higher than index x is being checked
	if (x >= (bufferLength - 1))
	{
		return;
	}

	bool lowerHasValue = bufferBegin[x] != 0;
	bool upperHasValue = bufferBegin[x + 1] != 0;

	if (lowerHasValue && !upperHasValue)
	{
		out_bufferDataEnd = &(bufferBegin[x]);
	}
}

///TODO
__global__ void
countCompactMortons(CompactMortonPoint* bufferBegin, size_t bufferLength, uint32_t* out_size)
{
	uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x);
	// One higher than index x is being checked
	if (x >= (bufferLength - 1))
	{
		return;
	}

	bool lowerHasValue = bufferBegin[x].compactData != 0;
	bool upperHasValue = bufferBegin[x + 1].compactData != 0;

	if (lowerHasValue && !upperHasValue)
	{
		*out_size = x; // out_bufferDataEnd = &(bufferBegin[x]);
	}
}


struct is_extracted_not_zero
{
	__host__ __device__
		bool operator()(const ExtractedPoint& point)
	{
		return point.morton != 0;
	}
};

struct is_morton_point_not_zero
{
	__host__ __device__
		bool operator()(const CompactMortonPoint& point)
	{
		return point.compactData != 0;
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
	extractInMortonOrderBlockDim((parseDimension + 7) / 8, (parseDimension + 7) / 8, (parseDimension + 7) / 8),
	extractInMortonOrderThreadDim(8, 8, 8),
	mortonSortedPointThreadSize(256),
	mortonSortedPointsBuffer(new thrust::device_vector< ExtractedPoint >(parseDimension * parseDimension * parseDimension)),
	mortonSortedPointsCompactBuffer(new thrust::device_vector< ExtractedPoint >(parseDimension * parseDimension * parseDimension)),
	areVerticiesOutsideIsoBuffer(new thrust::device_vector< uint32_t >((parseDimension + 2) * (parseDimension + 2) * (parseDimension + 2)))
{
	mortonSortedPointBlockSize = (mortonSortedPointsBuffer->size() + 255) / 256;
	cudaMalloc((void**)&device_sizeBucket, sizeof(uint32_t));
}

size_t
SDFHilbertExtractor::extract(SDFDevice& sdf, CudaGLBufferMapping<CompactMortonPoint>& mapping, CudaGLBufferMapping<WorldPositionMorton>& pbo, uint32_t overlapSize)
{
	mapping.map();
	size_t bufferLength = mapping.getSizeInBytes() / sizeof(CompactMortonPoint);
	CompactMortonPoint* bufferPointerRaw = thrust::raw_pointer_cast(mapping.getDeviceOutput());
	CompactMortonPoint* bufferPointerEndRaw = thrust::raw_pointer_cast(mapping.getDeviceOutput()) + bufferLength;
	thrust::device_ptr<CompactMortonPoint> bufferPointerDevice = thrust::device_pointer_cast(mapping.getDeviceOutput());
	uint32_t compactMortonBlockSize = ((bufferLength + 255) / 256), compactMortonThreadSize = 256;
	//thrust::fill(bufferPointerDevice, bufferPointerDevice + bufferLength, CompactMortonPoint());

	pbo.map();
	size_t pboBufferLength = pbo.getSizeInBytes() / sizeof(WorldPositionMorton);
	WorldPositionMorton* pboBufferPointerRaw = thrust::raw_pointer_cast(pbo.getDeviceOutput());
	WorldPositionMorton* pboBufferPointerEndRaw = thrust::raw_pointer_cast(pbo.getDeviceOutput()) + pboBufferLength;
	thrust::device_ptr<WorldPositionMorton> pboBufferPointerDevice = thrust::device_pointer_cast(pbo.getDeviceOutput());
	uint32_t worldPositionBlockSize = ((pboBufferLength + 255) / 256), worldPositionThreadSize = 256;
	//thrust::fill(pboBufferPointerDevice, pboBufferPointerDevice + pboBufferLength, 0);

	ExtractedPoint* mortenSortedPointsRaw = thrust::raw_pointer_cast(mortonSortedPointsBuffer->data());
	ExtractedPoint* mortenSortedPointsEndRaw = thrust::raw_pointer_cast(mortonSortedPointsBuffer->data()) + mortonSortedPointsBuffer->size();
	uint32_t* isVertexOusideIsoBufferRaw = thrust::raw_pointer_cast(areVerticiesOutsideIsoBuffer->data());

	ExtractedPoint* mortonSortedBegin = thrust::raw_pointer_cast(mortonSortedPointsCompactBuffer->data());
	ExtractedPoint* mortonSortedEnd = mortonSortedBegin + mortonSortedPointsCompactBuffer->size();

	CompactMortonPoint* endOfWrittenCompactMortons = bufferPointerRaw;
	WorldPositionMorton* endOfWrittenWorldMortons = pboBufferPointerRaw;

	for (int i = 0; i < gridDimension; i += parseDimension)
	{
		for (int j = 0; j < gridDimension; j += parseDimension)
		{
			for (int k = 0; k < gridDimension; k += parseDimension)
			{
				areVerticesOutsideIsosurface << <extractInMortonOrderBlockDim, extractInMortonOrderThreadDim >> > (isVertexOusideIsoBufferRaw, &sdf, gridDimension, parseDimension, i, j, k);
				extractPointsInMortonOrder << <extractInMortonOrderBlockDim, extractInMortonOrderThreadDim >> >(mortenSortedPointsRaw, &sdf, isVertexOusideIsoBufferRaw, gridDimension, parseDimension, i, j, k);
				//int numberCreated = thrust::count_if(mortonSortedPointsBuffer->begin(), mortonSortedPointsBuffer->end(), is_extracted_not_zero());
				thrust::copy_if(thrust::device, mortonSortedPointsBuffer->begin(), mortonSortedPointsBuffer->end(), mortonSortedPointsCompactBuffer->begin(), is_extracted_not_zero());
				clusterPoints << < mortonSortedPointBlockSize, mortonSortedPointThreadSize >> >(endOfWrittenCompactMortons, endOfWrittenWorldMortons, mortonSortedBegin, overlapSize, bufferPointerEndRaw, pboBufferPointerEndRaw, mortonSortedEnd);

				findEndOfWrittenCompactMortons << <compactMortonBlockSize, compactMortonThreadSize>> >(bufferPointerRaw, bufferLength, endOfWrittenCompactMortons);
				findEndOfWrittenWorldMortons << <worldPositionBlockSize, worldPositionThreadSize >> >(pboBufferPointerRaw, pboBufferLength, endOfWrittenWorldMortons);
			}
		}
	}

	uint32_t size = 0;
	countCompactMortons << <compactMortonBlockSize, compactMortonThreadSize >> >(bufferPointerRaw, bufferLength, device_sizeBucket);
	//int numberCreated = thrust::count_if(bufferPointerDevice, bufferPointerDevice + bufferLength, is_morton_point_not_zero());
	cudaMemcpy(&size, device_sizeBucket, sizeof(int), cudaMemcpyDeviceToHost);

	mapping.unmap();
	pbo.unmap();


	return size;
}
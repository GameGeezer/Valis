#include "SDFHilbertExtractor.cuh"

#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "cuda_runtime.h"

#include "NumericBoolean.cuh"

#include "SDFDevice.cuh"
#include "Morton30.cuh"

#include "Nova.cuh"

#include "SignedDistanceField.cuh"
#include "ByteArray.cuh"

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

	uint32_t offsetX = x + dimensionOffsetX;
	uint32_t offsetY = y + dimensionOffsetY;
	uint32_t offsetZ = z + dimensionOffsetZ;

	// Check to see if x, y, or z exceeds the bounds of the local grid
	// Check to see if x, y, ot z exceed the bounds of the entire grid
	if (x >= parseDimension || y >= parseDimension || z >= parseDimension || offsetX >= gridDimension || offsetY >= gridDimension || offsetZ >= gridDimension)
	{
		return;
	}

	uint32_t index = findIndex(x, y, z, parseDimension);

	float divisionsAsFloat = ((float)gridDimension);
	// normalized x, y, and z
	float normalizeX = (((float)offsetX) - 0.5f) / divisionsAsFloat;
	float normalizeY = (((float)offsetY) - 0.5f) / divisionsAsFloat;
	float normalizeZ = (((float)offsetZ) - 0.5f) / divisionsAsFloat;

	// How far the cell is from the sdf
	float distance = sdf->distanceFromPoint(glm::vec3(normalizeX, normalizeY, normalizeZ));

	NumericBoolean isDistancePositive = numericGreaterThan_uint32_t(distance, 0);

	d_output[index] = isDistancePositive;
}

__global__ void 
extractPointsInMortonOrder(ExtractedPoint *d_output, SDFDevice *sdf, uint32_t *vertexPlacement, uint32_t gridDimension, uint32_t parseDimension, uint32_t dimensionOffsetX, uint32_t dimensionOffsetY, uint32_t dimensionOffsetZ)
{

	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	uint32_t offsetX = x + dimensionOffsetX;
	uint32_t offsetY = y + dimensionOffsetY;
	uint32_t offsetZ = z + dimensionOffsetZ;

	// Check to see if x, y, or z exceeds the bounds of the local grid
	// Check to see if x, y, ot z exceed the bounds of the entire grid
	if (x >= parseDimension || y >= parseDimension || z >= parseDimension || offsetX >= gridDimension || offsetY >= gridDimension || offsetZ >= gridDimension)
	{
		return;
	}
	
	float divisionsAsFloat = ((float)gridDimension);

	// normalized x, y, and z
	float normalizeX = ((float)offsetX) / divisionsAsFloat;
	float normalizeY = ((float)offsetY) / divisionsAsFloat;
	float normalizeZ = ((float)offsetZ) / divisionsAsFloat;

	// How far the cell is from the sdf
	float distance = sdf->distanceFromPoint(glm::vec3(normalizeX, normalizeY, normalizeZ));

	uint32_t index = Morton30::encode(x, y, z);

	float cellDimension = 1.0f / divisionsAsFloat;

	NumericBoolean shouldGeneratePoint = numericLessThan_float(distance, cellDimension) * numericGreaterThan_float(distance, 0);
	
	NumericBoolean bottomLeftBack = vertexPlacement[findIndex(x, y, z, parseDimension)];
	NumericBoolean bottomRightBack = vertexPlacement[findIndex(x + 1, y, z, parseDimension)];
	NumericBoolean topLeftBack = vertexPlacement[findIndex(x, y + 1, z, parseDimension)];
	NumericBoolean bottomLeftForward = vertexPlacement[findIndex(x, y, z + 1, parseDimension)];
	NumericBoolean topRightBack = vertexPlacement[findIndex(x + 1, y + 1, z, parseDimension)];
	NumericBoolean bottomRightForward = vertexPlacement[findIndex(x + 1, y, z + 1, parseDimension)];
	NumericBoolean topLeftForward = vertexPlacement[findIndex(x, y + 1, z + 1, parseDimension)];
	NumericBoolean topRightForward = vertexPlacement[findIndex(x + 1, y + 1, z + 1, parseDimension)];

	glm::ivec3 bottomLeftBackVec = glm::ivec3(bottomLeftBack, bottomLeftBack, bottomLeftBack);
	glm::ivec3 bottomRightBackVec = glm::ivec3(-bottomRightBack, bottomRightBack, bottomRightBack);
	glm::ivec3 topLeftBackVec = glm::ivec3(topLeftBack, -topLeftBack, topLeftBack);
	glm::ivec3 bottomLeftForwardVec = glm::ivec3(bottomLeftForward, bottomLeftForward, -bottomLeftForward);
	glm::ivec3 topRightBackVec = glm::ivec3(-topRightBack, -topRightBack, topRightBack);
	glm::ivec3 bottomRightForwardVec = glm::ivec3(-bottomRightForward, bottomRightForward, -bottomRightForward);
	glm::ivec3 topLeftForwardVec = glm::ivec3(topLeftForward, -topLeftForward, -topLeftForward);
	glm::ivec3 topRightForwardVec = glm::ivec3(-topRightForward, -topRightForward, -topRightForward);
	
	glm::ivec3 normalVec = bottomLeftBackVec + bottomRightBackVec + topLeftBackVec + bottomLeftForwardVec + topRightBackVec + bottomRightForwardVec + topLeftForwardVec + topRightForwardVec;
	normalVec += glm::ivec3(4, 4, 4);

	d_output[index].normals.pack(normalVec.x, normalVec.y, normalVec.z);
	d_output[index].normals.compactData *= shouldGeneratePoint;
	d_output[index].morton = Morton30::encode(offsetX , offsetY, offsetZ);
	d_output[index].morton *= shouldGeneratePoint;
}

__global__ void
sdfExtractPointsInMortonOrder(ExtractedPoint *d_output, ByteArrayChunk *sdfData, uint32_t gridResolution, uint32_t parseDimension, uint32_t dimensionOffsetX, uint32_t dimensionOffsetY, uint32_t dimensionOffsetZ)
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

	uint32_t index = x + y * parseDimension + z * parseDimension * parseDimension;

	//d_output[index].normals.pack(normalVec.x, normalVec.y, normalVec.z);
	d_output[index].normals.compactData *= shouldGeneratePoint;
	d_output[index].morton = Morton30::encode(offsetX, offsetY, offsetZ);
	d_output[index].morton *= shouldGeneratePoint;
}

__global__ void
clusterPoints(CompactMortonPoint* renderPointBuffer, WorldPositionMorton* offsetBuffer, uint32_t* indexBuffer, ExtractedPoint* sortedPoints, size_t overlapSize, uint32_t* compactMortonStart, uint32_t* worldMortonStart, uint32_t* indexStart, uint32_t compactMortonBufSize, uint32_t worldMortonBufSize, ExtractedPoint* sortedPointsEnd, uint32_t indexBufferSize)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t renderPointBufferWriteIndex = (x * 64) + *compactMortonStart;
	uint32_t indexBufferWriteIndex = (x * 192) + *indexStart;
	uint32_t startingIndex = renderPointBufferWriteIndex - (overlapSize * x);
	uint32_t worldMortonIndex = *worldMortonStart + x;

	if ((worldMortonIndex >= worldMortonBufSize) || ((sortedPoints + startingIndex + 64) >= sortedPointsEnd) || (renderPointBufferWriteIndex >= compactMortonBufSize))
	{
		return;
	}

	uint32_t baseMorton = sortedPoints[startingIndex].morton;

	for (uint32_t i = 0; i < 64; ++i)
	{
		uint32_t upperMorton = sortedPoints[startingIndex + i].morton;
		uint32_t highestDifferent = Morton30::highestOrderBitDifferent(baseMorton, upperMorton);
		// Make sure the highest bit different is less than x is 64
		NumericBoolean isWithinBounds = numericLessThan_uint32_t(highestDifferent, 0x40000);
		// Subtract the base morton from the upper morton so relative offset can be stored
		uint32_t mortonOffset = (upperMorton - baseMorton);

		// unpack the normals
		uint32_t normalX, normalY, normalZ;
		sortedPoints[startingIndex + i].normals.unpack(normalX, normalY, normalZ);

		renderPointBuffer[renderPointBufferWriteIndex  + i].pack(mortonOffset, normalX, normalY, normalZ);
		renderPointBuffer[renderPointBufferWriteIndex + i].compactData = renderPointBuffer[renderPointBufferWriteIndex + i].compactData * isWithinBounds + baseMorton * numericNegate_uint32_t(isWithinBounds);
	}

	uint32_t previousMorton = sortedPoints[startingIndex].morton;
	uint32_t triangleCount = 0;
	uint32_t lastIndex = 0;
	for (uint32_t i = 0; i < 192; ++i)
	{
		uint32_t iRemainder = (i % 3);
		uint32_t indexOffset = ((i / 3) + iRemainder);
		uint32_t newMorton = sortedPoints[startingIndex + indexOffset].morton;
		uint32_t highestDifferent = Morton30::highestOrderBitDifferent(previousMorton, newMorton);

		NumericBoolean isLegal = numericLessThan_uint32_t(highestDifferent, 0x8);
		// If the index is legal we can add it to the triangle as an index
		indexBuffer[indexBufferWriteIndex + lastIndex + triangleCount] = renderPointBufferWriteIndex + indexOffset;
		triangleCount += isLegal;
		// Have we built a triangle?
		NumericBoolean triangleBuilt = numericGreaterThan_uint32_t(triangleCount, 2);
		
		// If the triangle has been built or is illegal reset the counter
		triangleCount = isLegal * numericNegate_uint32_t(triangleBuilt) * triangleCount;
		// If the triangle has been built move to the next three indicie indexes
		lastIndex += 3 * triangleBuilt;
		// Skip This index set if illegal (2 - iRemainder is the num left in the cluster 012 123 234 345 ect. is a cluster)
		i += numericNegate_uint32_t(isLegal) * (2 - iRemainder);
		uint32_t newIndexOffset = ((i / 3) + (i % 3));
		previousMorton = sortedPoints[startingIndex + newIndexOffset].morton;
	}
	
	offsetBuffer[worldMortonIndex] = baseMorton;
}

__global__ void
countIndices(uint32_t* bufferBegin, size_t bufferLength, uint32_t* out_size)
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
		*out_size = x;
	}
}

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
		*out_size = x;
	}
}

__global__ void
countWorldMortons(WorldPositionMorton* bufferBegin, size_t bufferLength, uint32_t* out_size)
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
		*out_size = x;
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

	cudaMalloc((void**)&device_compactMortonSizeBucket, sizeof(uint32_t));
	cudaMalloc((void**)&device_worldMortonSizeBucket, sizeof(uint32_t));
	cudaMalloc((void**)&device_indexSizeBucket, sizeof(uint32_t));
}

size_t
SDFHilbertExtractor::extract(SDFDevice& sdf, Nova &nova, uint32_t overlapSize)
{
	nova.map();
	nova.clean();

	ExtractedPoint* mortenSortedPointsRaw = thrust::raw_pointer_cast(mortonSortedPointsBuffer->data());
	ExtractedPoint* mortenSortedPointsEndRaw = thrust::raw_pointer_cast(mortonSortedPointsBuffer->data()) + mortonSortedPointsBuffer->size();
	uint32_t* isVertexOusideIsoBufferRaw = thrust::raw_pointer_cast(areVerticiesOutsideIsoBuffer->data());

	ExtractedPoint* mortonSortedBegin = thrust::raw_pointer_cast(mortonSortedPointsCompactBuffer->data());
	ExtractedPoint* mortonSortedEnd = mortonSortedBegin + mortonSortedPointsCompactBuffer->size();

	cudaMemset(device_compactMortonSizeBucket, 0, sizeof(uint32_t));
	cudaMemset(device_worldMortonSizeBucket, 0, sizeof(uint32_t));
	cudaMemset(device_indexSizeBucket, 0, sizeof(uint32_t));

	for (int i = 0; i < gridDimension; i += parseDimension)
	{
		for (int j = 0; j < gridDimension; j += parseDimension)
		{
			for (int k = 0; k < gridDimension; k += parseDimension)
			{
				
				//areVerticesOutsideIsosurface << <extractInMortonOrderBlockDim, extractInMortonOrderThreadDim >> > (isVertexOusideIsoBufferRaw, &sdf, gridDimension, parseDimension, i, j, k);
				//extractPointsInMortonOrder << <extractInMortonOrderBlockDim, extractInMortonOrderThreadDim >> >(mortenSortedPointsRaw, &sdf, isVertexOusideIsoBufferRaw, gridDimension, parseDimension, i, j, k);
				sdfExtractPointsInMortonOrder << <extractInMortonOrderBlockDim, extractInMortonOrderThreadDim >> >(mortenSortedPointsRaw, nova.getMaterialDevicePointer(), gridDimension, parseDimension, i, j, k);
				thrust::fill(mortonSortedPointsCompactBuffer->begin(), mortonSortedPointsCompactBuffer->end(), ExtractedPoint());
				thrust::copy_if(thrust::device, mortonSortedPointsBuffer->begin(), mortonSortedPointsBuffer->end(), mortonSortedPointsCompactBuffer->begin(), is_extracted_not_zero());
				clusterPoints << < mortonSortedPointBlockSize, mortonSortedPointThreadSize >> >(nova.getRawVBO(), nova.getRawPBO(), nova.getRawIBO(), mortonSortedBegin, overlapSize, device_compactMortonSizeBucket, device_worldMortonSizeBucket, device_indexSizeBucket, nova.getLengthVBO(), nova.getLengthPBO(), mortonSortedEnd, nova.getLengthIBO());
				
				countCompactMortons << <nova.getBlockSizeVBO(), NOVA_PARSE_BLOCK_SIZE >> >(nova.getRawVBO(), nova.getLengthVBO(), device_compactMortonSizeBucket);
				countWorldMortons << <nova.getBlockSizePBO(), NOVA_PARSE_BLOCK_SIZE >> >(nova.getRawPBO(), nova.getLengthPBO(), device_worldMortonSizeBucket);
				countIndices << <nova.getBlockSizeIBO(), NOVA_PARSE_BLOCK_SIZE >> >(nova.getRawIBO(), nova.getLengthIBO(), device_indexSizeBucket);
			}
		}
	}

	uint32_t size = 0;
	countCompactMortons << <nova.getBlockSizeVBO(), 256 >> >(nova.getRawVBO(), nova.getLengthVBO(), device_compactMortonSizeBucket);
	cudaMemcpy(&size, device_compactMortonSizeBucket, sizeof(uint32_t), cudaMemcpyDeviceToHost);

	nova.unmap();

	return size * 3;
}
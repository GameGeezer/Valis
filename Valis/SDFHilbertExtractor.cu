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

	// Decide whether to generate a point
	

	
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
clusterPoints(CompactMortonPoint* renderPointBuffer, WorldPositionMorton* offsetBuffer, ExtractedPoint* sortedPoints, size_t overlapSize, uint32_t* compactMortonStart, uint32_t* worldMortonStart, uint32_t compactMortonBufSize, uint32_t worldMortonBufSize, ExtractedPoint* sortedPointsEnd)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t renderPointBufferWriteIndex = (x * 64) + *compactMortonStart;
	uint32_t startingIndex = renderPointBufferWriteIndex - (overlapSize * x);
	uint32_t worldMortonIndex = *worldMortonStart + x;

	if (worldMortonIndex >= worldMortonBufSize)
	{
		int d = x + 5;
		return;
	}

	if ( ((sortedPoints + startingIndex + 64) >= sortedPointsEnd) )
	{
		int d = x + 5;
		return;
	}
	
	if (renderPointBufferWriteIndex >= compactMortonBufSize)
	{
		int d = x + 5;
		return;
	}

	


	uint32_t baseMorton = sortedPoints[startingIndex].morton;

	for (uint32_t i = 0; i < 64; ++i)
	{
		uint64_t upperMorton = sortedPoints[startingIndex + i].morton;
		uint64_t highestDifferent = Morton30::highestOrderBitDifferent(baseMorton, upperMorton);
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

	offsetBuffer[worldMortonIndex] = baseMorton;
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
		*out_size = x; // out_bufferDataEnd = &(bufferBegin[x]);
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
	cudaMalloc((void**)&device_compactMortonSizeBucket, sizeof(uint32_t));
	cudaMalloc((void**)&device_worldMortonSizeBucket, sizeof(uint32_t));
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
	thrust::fill(bufferPointerDevice, bufferPointerDevice + bufferLength, CompactMortonPoint());

	pbo.map();
	size_t pboBufferLength = pbo.getSizeInBytes() / sizeof(WorldPositionMorton);
	WorldPositionMorton* pboBufferPointerRaw = thrust::raw_pointer_cast(pbo.getDeviceOutput());
	WorldPositionMorton* pboBufferPointerEndRaw = thrust::raw_pointer_cast(pbo.getDeviceOutput()) + pboBufferLength;
	thrust::device_ptr<WorldPositionMorton> pboBufferPointerDevice = thrust::device_pointer_cast(pbo.getDeviceOutput());
	uint32_t worldPositionBlockSize = ((pboBufferLength + 255) / 256), worldPositionThreadSize = 256;
	thrust::fill(pboBufferPointerDevice, pboBufferPointerDevice + pboBufferLength, 0);

	ExtractedPoint* mortenSortedPointsRaw = thrust::raw_pointer_cast(mortonSortedPointsBuffer->data());
	ExtractedPoint* mortenSortedPointsEndRaw = thrust::raw_pointer_cast(mortonSortedPointsBuffer->data()) + mortonSortedPointsBuffer->size();
	uint32_t* isVertexOusideIsoBufferRaw = thrust::raw_pointer_cast(areVerticiesOutsideIsoBuffer->data());

	ExtractedPoint* mortonSortedBegin = thrust::raw_pointer_cast(mortonSortedPointsCompactBuffer->data());
	ExtractedPoint* mortonSortedEnd = mortonSortedBegin + mortonSortedPointsCompactBuffer->size();

	cudaMemset(device_compactMortonSizeBucket, 0, sizeof(uint32_t));
	cudaMemset(device_worldMortonSizeBucket, 0, sizeof(uint32_t));


	for (int i = 0; i < gridDimension; i += parseDimension)
	{
		for (int j = 0; j < gridDimension; j += parseDimension)
		{
			for (int k = 0; k < gridDimension; k += parseDimension)
			{
				//areVerticesOutsideIsosurface << <extractInMortonOrderBlockDim, extractInMortonOrderThreadDim >> > (isVertexOusideIsoBufferRaw, &sdf, gridDimension, parseDimension, i, j, k);
				extractPointsInMortonOrder << <extractInMortonOrderBlockDim, extractInMortonOrderThreadDim >> >(mortenSortedPointsRaw, &sdf, isVertexOusideIsoBufferRaw, gridDimension, parseDimension, i, j, k);
				//int numberCreated = thrust::count_if(mortonSortedPointsBuffer->begin(), mortonSortedPointsBuffer->end(), is_extracted_not_zero());
				thrust::fill(mortonSortedPointsCompactBuffer->begin(), mortonSortedPointsCompactBuffer->end(), ExtractedPoint());
				thrust::copy_if(thrust::device, mortonSortedPointsBuffer->begin(), mortonSortedPointsBuffer->end(), mortonSortedPointsCompactBuffer->begin(), is_extracted_not_zero());
				clusterPoints << < mortonSortedPointBlockSize, mortonSortedPointThreadSize >> >(bufferPointerRaw, pboBufferPointerRaw, mortonSortedBegin, overlapSize, device_compactMortonSizeBucket, device_worldMortonSizeBucket, bufferLength, pboBufferLength, mortonSortedEnd);

				countCompactMortons << <compactMortonBlockSize, compactMortonThreadSize >> >(bufferPointerRaw, bufferLength, device_compactMortonSizeBucket);
				countWorldMortons << <worldPositionBlockSize, worldPositionThreadSize >> >(pboBufferPointerRaw, pboBufferLength, device_worldMortonSizeBucket);
				//uint32_t size32 = 0;
				//cudaMemcpy(&size32, device_compactMortonSizeBucket, sizeof(uint32_t), cudaMemcpyDeviceToHost);
				//cudaMemcpy(&size32, device_worldMortonSizeBucket, sizeof(uint32_t), cudaMemcpyDeviceToHost);
			}
		}
	}

	uint32_t size = 0;
	countCompactMortons << <compactMortonBlockSize, compactMortonThreadSize >> >(bufferPointerRaw, bufferLength, device_compactMortonSizeBucket);
	//int numberCreated = thrust::count_if(bufferPointerDevice, bufferPointerDevice + bufferLength, is_morton_point_not_zero());
	cudaMemcpy(&size, device_compactMortonSizeBucket, sizeof(uint32_t), cudaMemcpyDeviceToHost);

	mapping.unmap();
	pbo.unmap();


	return size;
}
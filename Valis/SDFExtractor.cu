#include "SDFExtractor.cuh"

#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sort.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"

#include "NumericBoolean.cuh"
#include "RenderPoint.cuh"
#include "SDFDevice.cuh"


__global__ void extractPointCloudAsBitArray(ExtractionBlock *d_output, SDFDevice *sdf, uint32_t clusterDim)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	uint32_t gridDimension = (clusterDim * 4);

	if (x >= gridDimension || y >= gridDimension || z >= gridDimension)
	{
		return;
	}

	// The index of the cell in relation to 4 x 4 x 4 block of bits it's contained in
	uint32_t localX = x & 3;
	uint32_t localY = y & 3;
	uint32_t localZ = z & 3;

	uint32_t bitToFlip = localX + localY * 4 + localZ * 16;

	// Which cluster the cell is in
	uint32_t clusterX = x / 4;
	uint32_t clusterY = y / 4;
	uint32_t clusterZ = z / 4;

	uint32_t clusterIndex = clusterX + clusterY * clusterDim + clusterZ * clusterDim * clusterDim;

	float divisionsAsFloat = ((float)gridDimension);

	// normalized x, y, and z
	float normalizeX = ((float)x) / divisionsAsFloat;
	float normalizeY = ((float)y) / divisionsAsFloat;
	float normalizeZ = ((float)z) / divisionsAsFloat;

	// How far the cell is from the sdf
	float distance = sdf->distanceFromPoint(sdf->primitives, sdf->modifications, sdf->modificationCount, glm::vec3(normalizeX, normalizeY, normalizeZ));

	// Decide whether to generate a point
	float cellDimension = 1.0f / divisionsAsFloat;

	NumericBoolean shouldGeneratePoint = numericLessThan_float(distance, cellDimension) * numericGreaterThan_float(distance, 0);

	NumericBoolean writeFirst = numericLessThan_uint32_t(bitToFlip, 32);
	NumericBoolean writeSecond = numericNegate_uint32_t(writeFirst);

	bitToFlip = bitToFlip * writeFirst + (bitToFlip - 32) * writeSecond;

	uint32_t bitToOrWith = (1 << bitToFlip) *shouldGeneratePoint;
	uint32_t orFirst = bitToOrWith * writeFirst;
	uint32_t orSecond = bitToOrWith * writeSecond;
	atomicOr(&(d_output[clusterIndex].first), orFirst);
	atomicOr(&(d_output[clusterIndex].second), orSecond);
}

__global__ void createCloudFromBuffers(RenderPoint* d_output, ExtractionBlock *coverageBuffer, ExtractionBlock *materialBuffer, uint32_t subsectionClusterDim, uint32_t totalClusterDim, uint32_t clusterBufferSize, int dimensionOffsetX, int dimensionOffsetY, int dimensionOffsetZ)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	// Check to see if x, y, or z exceeds the bounds of the local grid
	if (x >= subsectionClusterDim * 4 || y >= subsectionClusterDim * 4 || z >= subsectionClusterDim * 4)
	{
		return;
	}


	uint32_t offsetX = x + dimensionOffsetX;
	uint32_t offsetY = y + dimensionOffsetY;
	uint32_t offsetZ = z + dimensionOffsetZ;

	// Check to see if x, y, ot z exceed the bounds of the entire grid
	if (offsetX >= totalClusterDim * 4 || offsetY >= totalClusterDim * 4 || offsetZ >= totalClusterDim * 4)
	{
		return;
	}

	int outputIndex = x + y * totalClusterDim * 4 + z * totalClusterDim * totalClusterDim * 16;

	if (outputIndex >= clusterBufferSize)
	{
		return;
	}
	
	// x, y, and z relative to the cluster 0, 0 , 0
	uint32_t localX = x & 3;
	uint32_t localY = y & 3;
	uint32_t localZ = z & 3;

	uint32_t bitToCheck = localX + localY * 4 + localZ * 16;

	// Which cluster the cell is in
	uint32_t clusterX = offsetX / 4;
	uint32_t clusterY = offsetY / 4;
	uint32_t clusterZ = offsetZ / 4;

	// The cluster index relative to the entire grid
	uint32_t clusterIndex = clusterX + clusterY * totalClusterDim + clusterZ * totalClusterDim * totalClusterDim;

	ExtractionBlock surfaceCoverage = coverageBuffer[clusterIndex];
	ExtractionBlock materialCoverage = materialBuffer[clusterIndex];

	NumericBoolean checkFirst = numericLessThan_uint32_t(bitToCheck, 32);
	NumericBoolean checkSecond = numericNegate_uint32_t(checkFirst);

	bitToCheck = bitToCheck * checkFirst + (bitToCheck - 32) * checkSecond;

	uint32_t bitToAndWith = (1 << bitToCheck);

	uint32_t andCoverageFirst = (surfaceCoverage.first & materialCoverage.first) & bitToAndWith;
	uint32_t andCoverageSecond = (surfaceCoverage.second & materialCoverage.second) & bitToAndWith;

	NumericBoolean foundFirst = numericGreaterThan_uint32_t(andCoverageFirst * checkFirst, 0);
	NumericBoolean foundSecond = numericGreaterThan_uint32_t(andCoverageSecond * checkSecond, 0);

	NumericBoolean materialCoverageOverlap = numericGreaterThan_uint32_t(foundFirst + foundSecond, 0);

	uint32_t gridDimension = (totalClusterDim * 4);

	float divisionsAsFloat = ((float)gridDimension);

	float normalizeX = ((float)offsetX) / divisionsAsFloat;
	float normalizeY = ((float)offsetY) / divisionsAsFloat;
	float normalizeZ = ((float)offsetZ) / divisionsAsFloat;

	d_output[outputIndex].positionX = normalizeX * materialCoverageOverlap;
	d_output[outputIndex].positionY = normalizeY * materialCoverageOverlap;
	d_output[outputIndex].positionZ = normalizeZ * materialCoverageOverlap;
	
}


SDFExtractor::SDFExtractor(uint32_t clusterDensity, uint32_t extractionClusterDensity) : clusterDensity(clusterDensity), extractionClusterDensity(extractionClusterDensity)
{
	pointCoverageBuffer = new thrust::device_vector< ExtractionBlock >(clusterDensity * clusterDensity * clusterDensity);
	materialCoverageBuffer = new thrust::device_vector< ExtractionBlock >(clusterDensity * clusterDensity * clusterDensity);
	partialExtractionBuffer = new thrust::device_vector< RenderPoint >(extractionClusterDensity * extractionClusterDensity * extractionClusterDensity * 64);
}

struct is_not_zero
{
	__host__ __device__
	bool operator()(const RenderPoint& point)
	{
		return point.positionX != 0 && point.positionY != 0 && point.positionZ != 0;
	}
};

struct is_not_zero_extract
{
	__host__ __device__
		bool operator()(const ExtractionBlock& point)
	{
		return point.first != 0 && point.second != 0;
	}
};

struct is_not_zero_uint32_t
{
	__host__ __device__
	bool operator()(const uint32_t& point)
	{
		return point != 0;
	}
};

struct shiftRenderPointsLeft
{
	__host__ __device__
	bool operator()(const RenderPoint& point1, const RenderPoint& point2)
	{
		return (point1.positionX + point1.positionY + point1.positionZ) >  (point2.positionX + point2.positionY + point2.positionZ);
	}
};

thrust::host_vector< RenderPoint >*
SDFExtractor::extract(SDFDevice& sdf)
{

	thrust::host_vector< RenderPoint >* extractedPoints = new thrust::host_vector< RenderPoint >();

	thrust::fill(pointCoverageBuffer->begin(), pointCoverageBuffer->end(), ExtractionBlock());

	ExtractionBlock* pointCoverageStart = thrust::raw_pointer_cast(pointCoverageBuffer->data());

	dim3 blocksBitAr(clusterDensity / 2, clusterDensity / 2, clusterDensity / 2);
	dim3 threadsBitAr(8, 8, 8);
	extractPointCloudAsBitArray << <blocksBitAr, threadsBitAr >> >(pointCoverageStart, &sdf, clusterDensity);

	//thrust::host_vector< ExtractionBlock > checkExtract = *pointCoverageBuffer;
	//int count3 = thrust::count_if(checkExtract.begin(), checkExtract.end(), is_not_zero_extract());
	//int z = checkExtract.size();
	RenderPoint* partialExtractionStart = thrust::raw_pointer_cast(partialExtractionBuffer->data());

	dim3 partialExtractionBlocks(extractionClusterDensity / 2, extractionClusterDensity / 2, extractionClusterDensity / 2);
	dim3 partialExtractionThreads(8, 8, 8);
	int totalCreated = 0;
	for (int i = 0; i < clusterDensity; i += extractionClusterDensity)
	{
		for (int j = 0; j < clusterDensity; j += extractionClusterDensity)
		{
			for (int k = 0; k < clusterDensity; k += extractionClusterDensity)
			{
				createCloudFromBuffers << <partialExtractionBlocks, partialExtractionThreads >> > (partialExtractionStart, pointCoverageStart, pointCoverageStart, extractionClusterDensity, clusterDensity, partialExtractionBuffer->size(), i * 4, j * 4, k * 4);
				thrust::sort(partialExtractionBuffer->begin(), partialExtractionBuffer->end(), shiftRenderPointsLeft());
				thrust::host_vector< RenderPoint > checkExtract = *partialExtractionBuffer;
				int numberCreated = thrust::count_if(partialExtractionBuffer->begin(), partialExtractionBuffer->end(), is_not_zero());
				extractedPoints->resize(totalCreated + numberCreated);
				cudaMemcpy(thrust::raw_pointer_cast(extractedPoints->data()) + totalCreated, partialExtractionStart, numberCreated * sizeof(RenderPoint), cudaMemcpyDeviceToHost);
				totalCreated += numberCreated;
			}
		}
	}
	
	return extractedPoints;
}
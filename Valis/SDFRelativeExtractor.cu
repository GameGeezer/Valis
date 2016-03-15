#include "SDFRelativeExtractor.cuh"

#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"

#include "NumericBoolean.cuh"
#include "RenderPoint.cuh"
#include "SDFDevice.cuh"

#include "CudaGLBufferMapping.cuh"

#include "VBO.cuh"
#include "PBO.cuh"

#include "CompactRenderPoint.cuh"
#include "CompactLocation.cuh"


__global__ void extractPointCloudAsBitArrayRelative(CompactRenderPoint *d_output, SDFDevice *sdf, uint32_t gridDimension, uint32_t parseDimension, uint32_t dimensionOffsetX, uint32_t dimensionOffsetY, uint32_t dimensionOffsetZ)
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

	uint32_t localX = x & 63;
	uint32_t localY = y & 63;
	uint32_t localZ = z & 63;

	uint32_t index = localX + localY * parseDimension + localZ * parseDimension * parseDimension;

	float divisionsAsFloat = ((float)gridDimension);

	// normalized x, y, and z
	float normalizeX = ((float)x) / divisionsAsFloat;
	float normalizeY = ((float)y) / divisionsAsFloat;
	float normalizeZ = ((float)z) / divisionsAsFloat;

	// How far the cell is from the sdf
	float distance = sdf->distanceFromPoint(glm::vec3(normalizeX, normalizeY, normalizeZ));

	// Decide whether to generate a point
	float cellDimension = 1.0f / divisionsAsFloat;

	NumericBoolean shouldGeneratePoint = numericLessThan_float(distance, cellDimension) * numericGreaterThan_float(distance, 0);

	d_output[index].pack(localX * shouldGeneratePoint, localY * shouldGeneratePoint, localZ * shouldGeneratePoint, 0, 0, 0);
}
/*
__global__ void createCloudFromBuffersRelative(CompactRenderPoint* d_output, ExtractionBlock *coverageBuffer, ExtractionBlock *materialBuffer, uint32_t subsectionClusterDim, uint32_t totalClusterDim, uint32_t clusterBufferSize, int dimensionOffsetX, int dimensionOffsetY, int dimensionOffsetZ)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	// Check to see if x, y, or z exceeds the bounds of the local grid
	uint32_t subGridDimension = (subsectionClusterDim * 4);
	if (x >= subGridDimension || y >= subGridDimension || z >= subGridDimension)
	{
		return;
	}

	uint32_t offsetX = x + dimensionOffsetX;
	uint32_t offsetY = y + dimensionOffsetY;
	uint32_t offsetZ = z + dimensionOffsetZ;

	// Check to see if x, y, ot z exceed the bounds of the entire grid
	uint32_t gridDimension = (totalClusterDim * 4);
	if (offsetX >= gridDimension || offsetY >= gridDimension || offsetZ >= gridDimension)
	{
		return;
	}

	// Make sure we don't write out of bounds
	int outputIndex = x + y * subGridDimension + z * subGridDimension * subGridDimension;
	if (outputIndex >= clusterBufferSize)
	{
		return;// Ideally this should never be hit, look into what it is.
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

	float divisionsAsFloat = ((float)gridDimension);

	float normalizeX = ((float)offsetX) / divisionsAsFloat;
	float normalizeY = ((float)offsetY) / divisionsAsFloat;
	float normalizeZ = ((float)offsetZ) / divisionsAsFloat;

	uint32_t writeX = x & 63;
	uint32_t writeY = y & 63;
	uint32_t writeZ = z & 63;

	d_output[outputIndex].pack(writeZ * materialCoverageOverlap, writeY * materialCoverageOverlap, writeZ * materialCoverageOverlap, 0 * materialCoverageOverlap, 0 * materialCoverageOverlap, 1 * materialCoverageOverlap);
}
*/
__global__ void copyLocalRelative(RenderPoint* d_output, RenderPoint *coverageBuffer, uint32_t blockWidth, uint32_t pointsToCopy)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	uint32_t index = x + y * blockWidth;

	if (x >= pointsToCopy)
	{
		return;
	}

	d_output[index] = coverageBuffer[index];
}


SDFRelativeExtractor::SDFRelativeExtractor(uint32_t clusterDensity, uint32_t extractionClusterDensity) :
clusterDensity(clusterDensity),
extractionClusterDensity(extractionClusterDensity),
coverageExtractBlockDim(clusterDensity / 2, clusterDensity / 2, clusterDensity / 2),
partialExtractionBlockDim(extractionClusterDensity / 2, extractionClusterDensity / 2, extractionClusterDensity / 2),
parseThreadsDim(8, 8, 8),
partialExtractionBuffer(new thrust::device_vector< CompactRenderPoint >(extractionClusterDensity * extractionClusterDensity * extractionClusterDensity * 64))
{

}

SDFRelativeExtractor::~SDFRelativeExtractor()
{
	delete partialExtractionBuffer;
}

struct is_not_zeroRelative
{
	__host__ __device__
		bool operator()(const CompactRenderPoint& point)
	{
		return point.compactData != 0;
	}
};

struct is_less_Relative
{
	__host__ __device__
		bool operator()(const CompactRenderPoint& point1, const CompactRenderPoint& point2)
	{
		return point1.compactData > point2.compactData;
	}
};


struct is_not_zero_extractRelative
{
	__host__ __device__
		bool operator()(const ExtractionBlock& point)
	{
		return point.first != 0 && point.second != 0;
	}
};

struct is_not_zero_uint32_tRelative
{
	__host__ __device__
		bool operator()(const uint32_t& point)
	{
		return point != 0;
	}
};


size_t
SDFRelativeExtractor::extract(SDFDevice& sdf, CudaGLBufferMapping<CompactRenderPoint>& mapping, PBO& pbo)
{
	mapping.map();
	size_t bufferLength = mapping.getSizeInBytes() / sizeof(CompactRenderPoint);
	CompactRenderPoint* bufferPointerRaw = thrust::raw_pointer_cast(mapping.getDeviceOutput());
	thrust::device_ptr<CompactRenderPoint> bufferPointerDevice = thrust::device_pointer_cast(mapping.getDeviceOutput());
	
	// Point to the partial extraction buffer
	CompactRenderPoint* partialExtractionRaw = thrust::raw_pointer_cast(partialExtractionBuffer->data());
	
	int maxExtractedElements = extractionClusterDensity * extractionClusterDensity * extractionClusterDensity * 64;
	// How many points have been created thus far
	thrust::host_vector<CompactLocation> locationBuffer;
	size_t totalCreated = 0;
	for (int i = 0; i < clusterDensity; i += extractionClusterDensity)
	{
		for (int j = 0; j < clusterDensity; j += extractionClusterDensity)
		{
			for (int k = 0; k < clusterDensity; k += extractionClusterDensity)
			{
				
				uint32_t offsetX = i * 4;
				uint32_t offsetY = j * 4;
				uint32_t offsetZ = k * 4;
				extractPointCloudAsBitArrayRelative << <coverageExtractBlockDim, parseThreadsDim >> >(bufferPointerRaw + totalCreated, &sdf, clusterDensity * 4, extractionClusterDensity * 4, offsetX, offsetY, offsetZ);
				//createCloudFromBuffersRelative << <partialExtractionBlockDim, parseThreadsDim >> > (partialExtractionRaw, pointCoverageRaw, pointCoverageRaw, extractionClusterDensity, clusterDensity, partialExtractionBuffer->size(), offsetX, offsetY, offsetZ);
				
				//Improve performance by eliminating this copy to the CPU
				int numberCreated = thrust::count_if(bufferPointerDevice + totalCreated, bufferPointerDevice + totalCreated + maxExtractedElements, is_not_zeroRelative());
				// Move all the newly created points to the left
				thrust::sort(thrust::device, bufferPointerRaw + totalCreated, bufferPointerRaw + totalCreated + maxExtractedElements, is_less_Relative());
				// multiple of 64 must be buffered
				int extraSpaceToBuffer = numberCreated & 63;
				// Fill in the extra space with the last point
				//thrust::fill(bufferPointerRaw + totalCreated + numberCreated, bufferPointerRaw + totalCreated + numberCreated + extraSpaceToBuffer, CompactRenderPoint());

				//uint32_t offsetY = j * 4;
				CompactLocation relativeLocation;
				relativeLocation.pack(offsetX, offsetY, offsetZ);
				for (int i = 0; i < (numberCreated / 64) + 63; ++i)
				{
					locationBuffer.push_back(relativeLocation);
				}

				//thrust::copy_if(partialExtractionBuffer->begin(), partialExtractionBuffer->end(), bufferPointer + totalCreated, is_not_zeroRelative());

				totalCreated += (numberCreated + extraSpaceToBuffer);
				/*
				*/
			}
		}
	}

	mapping.unmap();



	return totalCreated;
}
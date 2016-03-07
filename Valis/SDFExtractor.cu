#include "SDFExtractor.cuh"

#include "Color.cuh"
#include "NumericBoolean.cuh"
#include "SDSphere.cuh"
#include "SDTorus.cuh"
#include "RenderPoint.cuh"
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "SDFHost.cuh"
#include "SDFDevice.cuh"
#include "PlaceSDPrimitive.cuh"


//const float SQRT2 = 1.41421f;

__global__ void
extractPointCloudSphere(RenderPoint *d_output, SDFDevice* sdf,  int length, int gridDivisions)
{
	float x = blockIdx.x * blockDim.x + threadIdx.x;
	float y = blockIdx.y * blockDim.y + threadIdx.y;
	float z = blockIdx.z * blockDim.z + threadIdx.z;

	int index = x + gridDivisions * y + gridDivisions * gridDivisions * z;

	if (index >= length)
	{
		return;
	}

	if (x > gridDivisions || y  > gridDivisions || z  > gridDivisions)
	{
		return;
	}

	float divisionsAsFloat = ((float) gridDivisions);

	x /= divisionsAsFloat;
	y /= divisionsAsFloat;
	z /= divisionsAsFloat;

	/*
	SDSphere sdSphere(0.2f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDTorus sdTorus(0.31f, 0.1f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDTorus sdTorus2(0.4f, 0.1f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDSphere sdSphere2(0.1f, glm::vec3(0.5f, 0.5f, 0.8f));
	SDSphere sdSphere3(0.05f, glm::vec3(0.5f, 0.5f, 0.7f));
	SDSphere sdSphere4(0.05f, glm::vec3(0.5f, 0.5f, 0.6f));
	SDSphere sdSphere5(0.025f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDSphere sdSphere6(0.05f, glm::vec3(0.5f, 0.5f, 0.4f));
	SDSphere sdSphere7(0.05f, glm::vec3(0.5f, 0.5f, 0.3f));
	SDSphere sdSphere8(0.1f, glm::vec3(0.5f, 0.5f, 0.2f));


	float distance1 = sdSphere.distanceFromPoint(glm::vec3(x, y, z));
	float distance2 = sdTorus.distanceFromPoint(glm::vec3(x, y, z));
	float distance3 = sdTorus2.distanceFromPoint(glm::vec3(x, y, z));
	float distance4 = sdSphere2.distanceFromPoint(glm::vec3(x, y, z));
	float distance5 = sdSphere3.distanceFromPoint(glm::vec3(x, y, z));
	float distance6 = sdSphere4.distanceFromPoint(glm::vec3(x, y, z));
	float distance7 = sdSphere5.distanceFromPoint(glm::vec3(x, y, z));
	float distance8 = sdSphere6.distanceFromPoint(glm::vec3(x, y, z));
	float distance9 = sdSphere7.distanceFromPoint(glm::vec3(x, y, z));
	float distance10 = sdSphere8.distanceFromPoint(glm::vec3(x, y, z));
	*/
	float distance = sdf->distanceFromPoint(sdf->primitives, sdf->modifications, sdf->modificationCount, glm::vec3(x, y, z)); //fminf(distance1, distance2);
	//distance = fmaxf(distance, -distance4);
	//distance = fmaxf(distance, -distance5);
	//distance = fmaxf(distance, -distance6);
	//distance = fmaxf(distance, -distance6);
	//distance = fmaxf(distance, -distance7);
	//distance = fmaxf(distance, -distance8);
	//distance = fmaxf(distance, -distance9);
	//distance = fmaxf(distance, -distance10);
	//distance = fmaxf(distance, -distance3);
	
	float cellDimension = 1.0f / divisionsAsFloat;

	NumericBoolean shouldGeneratePoint = numericLessThan_float(distance, cellDimension) * numericGreaterThan_float(distance, 0);

	int color = Color(1, 0, 0, 0).device_toInt();

	d_output[index].positionX = x * shouldGeneratePoint;
	d_output[index].positionY = y * shouldGeneratePoint;
	d_output[index].positionZ = z * shouldGeneratePoint;
	//d_output[index].color = color * shouldGeneratePoint;
}


__global__ void extractPointCloudAsBitArray(ExtractionBlock *d_output, uint32_t clusterDim)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	uint32_t gridDimension = (clusterDim * 4);

	if (x >= gridDimension || y >= gridDimension || z >= gridDimension)
	{
		return;
	}

	uint32_t localX = x % 3;
	uint32_t localY = y % 3;
	uint32_t localZ = z % 3;

	uint32_t bitToFlip = localX + localY * 4 + localZ * 16;

	uint32_t clusterX = x / 4;
	uint32_t clusterY = y / 4;
	uint32_t clusterZ = z / 4;

	//Find the distance from the SDF
	float divisionsAsFloat = ((float) gridDimension); // bad remove

	float normalizeX = ((float) x) / divisionsAsFloat;
	float normalizeY = ((float) y) / divisionsAsFloat;
	float normalizeZ = ((float) z) / divisionsAsFloat;

	//SDSphere sdSphere(0.5f, glm::vec3(0.5f, 0.5f, 0.5f));

	float distance = 1;// sdSphere.distanceFromPoint(glm::vec3(normalizeX, normalizeY, normalizeZ));

	// Decide whether to generate a point
	float cellDimension = 1.0f / divisionsAsFloat;

	NumericBoolean shouldGeneratePoint = numericLessThan_float(distance, cellDimension) * numericGreaterThan_float(distance, 0);

	NumericBoolean writeFirst = numericLessThan_uint32_t(bitToFlip, 32);
	
	uint32_t clusterIndex = clusterX + clusterY * clusterDim + clusterZ * clusterDim * clusterDim;
//	uint32_t* writeAddress = (&(d_output[clusterIndex].first)) * clusterIndex + (&(d_output[clusterIndex].second)) * numericNegate_uint32_t(clusterIndex);
	uint32_t bitToOrWith = (1 << bitToFlip) * shouldGeneratePoint;
	//atomicOr(&(d_output[clusterIndex]), bitToOrWith);
}
/*
__global__ void createCloudFromBuffers(RenderPoint* d_output, uint32_t *coverageBuffer, uint32_t *materialBuffer, int gridDivisions, int extractDimensions, int dimensionOffsetX, int dimensionOffsetY, int dimensionOffsetZ)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= extractDimensions || y >= extractDimensions || z >= extractDimensions)
	{
		return;
	}

	int outputIndex = x + extractDimensions * y + extractDimensions * extractDimensions * z;

	x += dimensionOffsetX;
	y += dimensionOffsetY;
	z += dimensionOffsetZ;

	uint32_t intsPerRow = gridDivisions / 32;

	uint32_t index = (x + 1) / 32 + y * intsPerRow + z * intsPerRow * intsPerRow;

	uint32_t relativeX = x & 31; // mod 32

	uint32_t coverageBit = coverageBuffer[index] & (1 << relativeX);
	uint32_t materialBit = materialBuffer[index] & (1 << relativeX);

	NumericBoolean materialCoverageOverlap = coverageBit > 0;

	float divisionsAsFloat = ((float)gridDivisions);

	float normalizeX = ((float)x) / divisionsAsFloat;
	float normalizeY = ((float)y) / divisionsAsFloat;
	float normalizeZ = ((float)z) / divisionsAsFloat;

	d_output[outputIndex].positionX = normalizeX * materialCoverageOverlap;
	d_output[outputIndex].positionY = normalizeY * materialCoverageOverlap;
	d_output[outputIndex].positionZ = normalizeZ * materialCoverageOverlap;
}
*/

__global__ void myKernel()
{
	printf("Hello, world from the device!\n");
}



SDFExtractor::SDFExtractor()
{
	extractedPoints = new thrust::device_vector< RenderPoint >(300 * 300 * 300);

	pointCoverageBuffer = new thrust::device_vector< ExtractionBlock >(gridResolution * gridResolution * gridResolution / 32);
	materialCoverageBuffer = new thrust::device_vector< uint32_t >(gridResolution * gridResolution * gridResolution / 32);
	partialExtractionBuffer = new thrust::device_vector< RenderPoint >(partialExtractionSize * partialExtractionSize * partialExtractionSize);
}

struct is_not_zero
{
	__host__ __device__
	bool operator()(const RenderPoint& point)
	{
		return point.positionX != 0 && point.positionY != 0 && point.positionZ != 0;
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

thrust::device_vector< RenderPoint >*
SDFExtractor::extract()
{
	/*
	thrust::fill(pointCoverageBuffer->begin(), pointCoverageBuffer->end(), 0);
	ExtractionBlock* coverageStart = thrust::raw_pointer_cast(pointCoverageBuffer->data());
	//int vecLength = pointCoverageBuffer->size();
	dim3 blocksBitAr(100, 100, 100);
	dim3 threadsBitAr(4, 4, 4);
	extractPointCloudAsBitArray << <blocksBitAr, threadsBitAr >> >(coverageStart, 100);

	//thrust::device_vector<RenderPoint>* compactedPoints = new thrust::device_vector< RenderPoint >(pointsCreated);

	thrust::device_vector< RenderPoint >* createdPoints = new thrust::device_vector< RenderPoint >(10000000);
	RenderPoint* partialExtractionStart = thrust::raw_pointer_cast(partialExtractionBuffer->data());
	dim3 partialExtractionBlocks(25, 25, 25);
	dim3 partialExtractionThreads(8, 8, 8);
	int numberOfPointsCreated = 0;
	for (int i = 0; i < gridResolution; i += partialExtractionSize)
	{
		for (int j = 0; j < gridResolution; j += partialExtractionSize)
		{
			for (int k = 0; k < gridResolution; k += partialExtractionSize)
			{

				//thrust::fill(partialExtractionBuffer->begin(), partialExtractionBuffer->end(), zeroRenderPoint);
				/*
				createCloudFromBuffers << <partialExtractionBlocks, partialExtractionThreads >> > (partialExtractionStart, coverageStart, coverageStart, gridResolution, partialExtractionSize, i, j, k);

				int newPointsCreated = thrust::count_if(partialExtractionBuffer->begin(), partialExtractionBuffer->end(), is_not_zero());
				if ((newPointsCreated + numberOfPointsCreated) > createdPoints->capacity())
				{
					createdPoints->resize(newPointsCreated + numberOfPointsCreated);
				}

				thrust::copy_if(partialExtractionBuffer->begin(), partialExtractionBuffer->end(), createdPoints->begin() + numberOfPointsCreated, is_not_zero());
				numberOfPointsCreated += newPointsCreated;
				
			}
		}
		
	}
	*/
	SDSphere sdSphere(0.5f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDTorus sdTorus(0.31f, 0.1f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDModification* place = new PlaceSDPrimitive();
	SDFHost* testSDF = new SDFHost(&sdSphere);
	testSDF->modify(&sdTorus, place);
	SDFDevice* testSDFDevice = testSDF->copyToDevice();
	
	RenderPoint* vecStart = thrust::raw_pointer_cast(extractedPoints->data());
	int vecLength = extractedPoints->size();
	dim3 blocks(40, 40, 40);
	dim3 threads(8, 8, 8);
	extractPointCloudSphere << <blocks, threads >> >(vecStart, testSDFDevice, vecLength, 300);


	int index = 200 + 400 * 200;

	int pointsCreated = thrust::count_if(extractedPoints->begin(), extractedPoints->end(), is_not_zero());
	thrust::device_vector<RenderPoint>* compactedPoints = new thrust::device_vector< RenderPoint >(pointsCreated);
	thrust::copy_if(extractedPoints->begin(), extractedPoints->end(), compactedPoints->begin(), is_not_zero());
	
	return compactedPoints;
}
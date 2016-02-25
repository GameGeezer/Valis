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

//const float SQRT2 = 1.41421f;

__global__ void
extractPointCloudSphere(RenderPoint *d_output, int length, int gridDivisions)
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

	SDSphere sdSphere(0.3f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDTorus sdTorus(0.3f, 0.25f, glm::vec3(0.5f, 0.5f, 0.5f));

	float distance1 = sdSphere.distanceFromPoint(glm::vec3(x, y, z));
	float distance2 = sdTorus.distanceFromPoint(glm::vec3(x, y, z));

	float distance = fmaxf(distance1, -distance2);
	
	float cellDimension = 1.0f / divisionsAsFloat;

	NumericBoolean shouldGeneratePoint = numericLessThan_float(distance, cellDimension) * numericGreaterThan_float(distance, 0);

	int color = Color(1, 0, 0, 0).device_toInt();

	d_output[index].positionX = x * shouldGeneratePoint;
	d_output[index].positionY = y * shouldGeneratePoint;
	d_output[index].positionZ = z * shouldGeneratePoint;
	//d_output[index].color = color * shouldGeneratePoint;
}

__global__ void myKernel()
{
	printf("Hello, world from the device!\n");
}



SDFExtractor::SDFExtractor()
{
	extractedPoints = new thrust::device_vector< RenderPoint >(200 * 200 * 200);
}

struct is_not_zero
{
	__host__ __device__
	bool operator()(const RenderPoint& point)
	{
		return point.positionX != 0 && point.positionY != 0 && point.positionZ != 0;
	}
};

thrust::device_vector< RenderPoint >*
SDFExtractor::extract()
{
	RenderPoint* vecStart = thrust::raw_pointer_cast(extractedPoints->data());
	int vecLength = extractedPoints->size();
	dim3 blocks(32, 32, 32);
	dim3 threads(8, 8, 8);
	extractPointCloudSphere << <blocks, threads >> >(vecStart, vecLength, 200);


	int index = 200 + 400 * 200;
	//thrust::host_vector< RenderPoint > offGPU = *extractedPoints;
	//RenderPoint r = offGPU[index];

	int pointsCreated = thrust::count_if(extractedPoints->begin(), extractedPoints->end(), is_not_zero());
	thrust::device_vector<RenderPoint>* compactedPoints = new thrust::device_vector< RenderPoint >(pointsCreated);
	thrust::copy_if(extractedPoints->begin(), extractedPoints->end(), compactedPoints->begin(), is_not_zero());
	//thrust::host_vector< RenderPoint > offGPU = *compactedPoints;
	//RenderPoint r = offGPU[0];
//	myKernel << <1, 10 >> >();
	//int x = 3;
	return compactedPoints;
}
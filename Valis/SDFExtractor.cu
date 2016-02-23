#include "SDFExtractor.cuh"

/*
#include "Color.cuh"
#include "CudaGLBufferMapping.cuh"
#include "Camera.cuh"
#include <glm/mat4x4.hpp>
#include "DistanceFunctions.h"
#include "NumericBoolean.cuh"
#include "SDSphere.cuh"
#include "RenderPoint.cuh"


__global__ void
extractPointCloudSphere(RenderPoint *d_output, glm::vec3 gridDimensions)
{
	float x = blockIdx.x * blockDim.x + threadIdx.x;
	float y = blockIdx.y * blockDim.y + threadIdx.y;
	float z = blockIdx.z * blockDim.z + threadIdx.z;

	int index = x + gridDimensions.x * y + gridDimensions.x * gridDimensions.y * z;

	if (x > gridDimensions.x || y  > gridDimensions.y || z  > gridDimensions.z)
	{
		return;
	}

	x /= gridDimensions.x;
	y /= gridDimensions.y;
	z /= gridDimensions.z;

	SDSphere sdSphere(1, glm::vec3(1, 1, 1));

	float distance = sdSphere.distanceFromPoint(glm::vec3(x, y, z));

	d_output[index].setPosition(1, 1, 1);
}

__global__ void
extractPointCloudSphere(int *d_output, int imageW, int imageH, glm::vec3 gridDimensions, glm::vec3 subsectionDimensions)
{
	float x = (blockIdx.x * blockDim.x + threadIdx.x) * subsectionDimensions.x;
	float y = (blockIdx.y * blockDim.y + threadIdx.y) * subsectionDimensions.y;
	float z = (blockIdx.z * blockDim.z + threadIdx.z) * subsectionDimensions.z;

	if ((x + subsectionDimensions.x) > gridDimensions.x || (y + subsectionDimensions.y)  > gridDimensions.y || (z + subsectionDimensions.z)  > gridDimensions.z)
	{
		return;
	}

	x /= gridDimensions.x;
	y /= gridDimensions.y;
	z /= gridDimensions.z;

	float dx = x / gridDimensions.x;
	float dy = y / gridDimensions.y;
	float dz = z / gridDimensions.z;

	SDSphere sdSphere(1, glm::vec3(1, 1, 1));

	for (int i = 0; i < subsectionDimensions.x; ++i)
	{
		for (int j = 0; j < subsectionDimensions.y; ++j)
		{
			for (int k = 0; k < subsectionDimensions.z; ++k)
			{

			}
		}
	}

	float distance = sdSphere.distanceFromPoint(glm::vec3(x, y, z));

	if ((x < imageW) && (y < imageH))
	{
		Color color(0.1f, 0.25f, 1, 1);
		// In our sample tex is always valid, but for something like your own
		// sparse texturing you would need to make sure to handle the zero case.

		// write output color
		int i = y * imageW + x;
		d_output[i] = color.device_toInt();
	}
}



SDFExtractor::SDFExtractor()
{
	extractedPoints = new thrust::device_vector< float >(400 * 400 * 400);
}

void
SDFExtractor::extract()
{
	
}

*/
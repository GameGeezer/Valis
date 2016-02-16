#include "SDFPointCloudExtractor.h"

#include "cuda_runtime.h"

#include "device_launch_parameters.h"

/*
__global__ void
extractCloud(int *d_output, int imageW, int imageH)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x >= imageW) && (y >= imageH))
	{
	}
}
*/

SDFPointCloudExtractor::SDFPointCloudExtractor()
{

}

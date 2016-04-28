#include "ByteArray.cuh"

#include <thrust/fill.h>
#include <thrust/copy.h>

#include "CudaHelper.cuh"

__global__ void
copyByteArrayInto(ByteArrayChunk *from, ByteArrayChunk *to, size_t size)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= size)
	{
		return;
	}

	to[x] = from[x];
}

__global__ void
zeroByteArray(ByteArrayChunk *d_output, size_t size)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= size)
	{
		return;
	}

	d_output[x].first = 0;
	d_output[x].second = 0;
}

ByteArray::ByteArray(size_t size) : 
	size((size + 3) / 4)
{
	device_data = new thrust::device_vector<ByteArrayChunk>(this->size);
	rawDataPointer = thrust::raw_pointer_cast(device_data->data());
}


void
ByteArray::copyInto(ByteArray& other)
{
	copyByteArrayInto <<< (size + 255) / 256, 256 >>> (rawDataPointer, other.rawDataPointer, size);
}

ByteArrayChunk*
ByteArray::getDevicePointer()
{
	return rawDataPointer;
}

void
ByteArray::zero()
{
	zeroByteArray << < (size + 255) / 256, 256 >> > (rawDataPointer, size);
}




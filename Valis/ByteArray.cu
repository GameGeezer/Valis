#include "ByteArray.cuh"

#include <thrust/fill.h>
#include <thrust/copy.h>

#include "CudaHelper.cuh"


ByteArray::ByteArray(size_t size) : 
	size((size + 3) / 4)
{
	device_data = new thrust::device_vector<ByteArrayChunk>(this->size);
	rawDataPointer = thrust::raw_pointer_cast(device_data->data());
}


void
ByteArray::copyInto(ByteArray& other)
{
	//assertCUDA(cudaMemcpy(other.device_data, device_data, size, cudaMemcpyDeviceToDevice));
	thrust::copy(device_data->begin(), device_data->end(), other.device_data->begin());
}

ByteArrayChunk*
ByteArray::getDevicePointer()
{
	return rawDataPointer;
}

void
ByteArray::zero()
{
	thrust::fill(device_data->begin(), device_data->end(), ByteArrayChunk());
}




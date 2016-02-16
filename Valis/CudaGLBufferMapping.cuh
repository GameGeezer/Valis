#ifndef VALIS_CUDAGLBUFFERMAPPING_H
#define VALIS_CUDAGLBUFFERMAPPING_H

#include "CudaHelper.cuh"
#include "PBO.cuh"

#include <cuda_gl_interop.h>

class CudaGLBufferMapping
{
public:

	
	CudaGLBufferMapping(PBO& pbo)
	{
		assertCUDA(cudaGraphicsGLRegisterBuffer(&handle, pbo.getHandle(), cudaGraphicsMapFlagsWriteDiscard));
	}

	inline void
	map()
	{
		assertCUDA(cudaGraphicsMapResources(1, &handle, 0));
		assertCUDA(cudaGraphicsResourceGetMappedPointer((void **)&device_output, &sizeBytes, handle));
	}

	inline void
	unmap()
	{
		assertCUDA(cudaGraphicsUnmapResources(1, &handle, 0));
	}

	inline size_t
	getSizeInBtyes()
	{
		return sizeBytes;
	}

	inline int*
	getDeviceOutput()
	{
		return device_output;
	}

private:
	struct cudaGraphicsResource *handle;
	int *device_output;
	size_t sizeBytes;
};

#endif //VALIS_CUDAGLBUFFERMAPPING_H
#ifndef VALIS_CUDAGLBUFFERMAPPING_H
#define VALIS_CUDAGLBUFFERMAPPING_H

#include "CudaHelper.cuh"
#include "PBO.cuh"
#include "VBO.cuh"
#include "IBO.cuh"

#include <cuda_gl_interop.h>

template <class ViewType>
class CudaGLBufferMapping
{
public:

	CudaGLBufferMapping(GLuint bufferHandle, cudaGraphicsMapFlags flags)
	{
		assertCUDA(cudaGraphicsGLRegisterBuffer(&handle, bufferHandle, flags));
	}

	~CudaGLBufferMapping()
	{
		assertCUDA(cudaGraphicsUnregisterResource(handle));
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
	getSizeInBytes()
	{
		return sizeBytes;
	}

	inline ViewType*
	getDeviceOutput()
	{
		return device_output;
	}

private:
	struct cudaGraphicsResource *handle;
	ViewType *device_output;
	size_t sizeBytes;
};

#endif 
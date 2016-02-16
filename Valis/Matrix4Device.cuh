#ifndef VALIS_MATRIX4_DEVICE_H
#define VALIS_MATRIX4_DEVICE_H

/*
#include "Matrix4.cuh"

class Matrix4Device
{

public:

	Matrix4Device(Matrix4& hostMatrix) : hostPtr(&hostMatrix)
	{
		cudaMalloc((void **)&devicePtr, sizeof(Matrix4));

		updateOnDevice();
	}

	~Matrix4Device()
	{
		cudaFree(devicePtr);
	}

	inline void
	updateOnDevice()
	{
		cudaMemcpy(devicePtr, hostPtr, sizeof(Matrix4), cudaMemcpyHostToDevice);
	}

private:
	Matrix4 *hostPtr, *devicePtr;
};

*/
#endif
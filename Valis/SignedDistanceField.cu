#include "SignedDistanceField.cuh"

#include <glm\glm.hpp>
#include <glm\vec4.hpp>

#include "ByteArray.cuh"

#include "SDSphere.cuh"
#include "SDModification.cuh"

#include "NumericBoolean.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__device__ __inline__ float
sdfDistanceFromSphere(glm::vec4 point, glm::mat4 transform, float radius)
{
	point = transform * point;
	return GLMUtil::length(glm::vec3(point)) - radius;
}

__global__ void 
placeSphereMaterialDistanceField(ByteArrayChunk *d_output, SDSphere *primitive, uint32_t material, uint32_t gridResolution)
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x > gridResolution || y > gridResolution || z > gridResolution)
	{
		return;
	}

	uint32_t index = x + y * gridResolution + z * gridResolution * gridResolution;

	float cellDimension = 1.0f / ((float)gridResolution);

	// normalized x, y, and z
	float normalizeX = ((float)x) * cellDimension;
	float normalizeY = ((float)y) * cellDimension;
	float normalizeZ = ((float)z) * cellDimension;

	// How far the cell is from the sdf
	float distance = sdfDistanceFromSphere(glm::vec4(normalizeX, normalizeY, normalizeZ, 1), primitive->transform, primitive->radius);

	NumericBoolean isOutside = numericGreaterThan_float(distance, 0);
	NumericBoolean isInside = numericNegate_uint32_t(isOutside);


	uint32_t indexCurrentValue = byteArray_getValueAtIndex(d_output, index);
	NumericBoolean insideWholeShape = numericEqual_uint32_t(indexCurrentValue, SDF_INSIDE_SURFACE);
	NumericBoolean notInsideWholeShape = numericNegate_uint32_t(insideWholeShape);

	NumericBoolean shouldGeneratePoint = numericLessThan_float(distance, cellDimension) * isOutside;
	NumericBoolean shouldNotGeneratePoint = numericNegate_uint32_t(shouldGeneratePoint);

	// TODO don't write if inside the surface
	uint32_t ifPointIsLegal = material * notInsideWholeShape + SDF_INSIDE_SURFACE * insideWholeShape;
	uint32_t ifPointIsIllegal = indexCurrentValue * isOutside + SDF_INSIDE_SURFACE * isInside;
	uint32_t valueToWrite = ifPointIsIllegal * shouldNotGeneratePoint + ifPointIsLegal *  shouldGeneratePoint;

	byteArray_setValueAtIndex(d_output, index, valueToWrite);
}

SignedDistanceField::SignedDistanceField(uint32_t gridResolution) :
	gridResolution(gridResolution),
	materialBlockSize((gridResolution + (SDF_THREAD_BLOCK_DIM - 1)) / SDF_THREAD_BLOCK_DIM, (gridResolution + (SDF_THREAD_BLOCK_DIM - 1)) / SDF_THREAD_BLOCK_DIM, (gridResolution + (SDF_THREAD_BLOCK_DIM - 1)) / SDF_THREAD_BLOCK_DIM),
	materialThreadSize(SDF_THREAD_BLOCK_DIM, SDF_THREAD_BLOCK_DIM, SDF_THREAD_BLOCK_DIM),
	materialGrid(*(new ByteArray(gridResolution * gridResolution * gridResolution))),
	normalGrid(*(new ByteArray(gridResolution * gridResolution * gridResolution)))
{
	materialGrid.zero();
	normalGrid.zero();
}

void
SignedDistanceField::place(DistancePrimitive& primitive, uint32_t material)
{
	SDSphere* devicePrimitive = (SDSphere*)primitive.copyToDevice();
	placeSphereMaterialDistanceField << <materialBlockSize, materialThreadSize >> >(materialGrid.getDevicePointer(), devicePrimitive, material, gridResolution);
}

void
SignedDistanceField::copyInto(SignedDistanceField& other)
{
	materialGrid.copyInto(other.materialGrid);
	//normalGrid.copyInto(other.normalGrid);
}

ByteArrayChunk*
SignedDistanceField::getMaterialDevicePointer()
{
	return materialGrid.getDevicePointer();
}
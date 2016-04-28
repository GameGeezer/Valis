#include "SignedDistanceField.cuh"

#include <glm\glm.hpp>
#include <glm\vec4.hpp>

#include "ByteArray.cuh"

#include "SDSphere.cuh"
#include "SDTorus.cuh"
#include "SDCube.cuh"
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

__device__ __inline__ float
sdfDistanceFromTorus(glm::vec4 point, glm::mat4 transform, glm::vec2 dimensions)
{
	point = transform * point;
	glm::vec2 q = glm::vec2(GLMUtil::length(glm::vec2(point.x, point.y)) - dimensions.x, point.z);
	return GLMUtil::length(q) - dimensions.y;
}

__device__ __inline__ float
sdfDistanceFromCube(glm::vec4 point, glm::mat4 transform, glm::vec3 corner)
{
	point = transform * point;
	glm::vec3 d = glm::vec3(abs(point.x), abs(point.y), abs(point.z)) - corner;
	return fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.0) + GLMUtil::length(glm::vec3(fmaxf(d.x, 0), fmaxf(d.y, 0), fmaxf(d.z, 0)));
}

__device__ __inline__ float
placePoint(ByteArrayChunk *materialGrid, ByteArrayChunk *surfaceGrid, float distance, float cellDimension, uint32_t index, uint32_t material)
{
	NumericBoolean isOutside = numericGreaterThan_float(distance, 0);
	NumericBoolean isInside = numericNegate_uint32_t(isOutside);

	uint32_t indexCurrentValue = byteArray_getValueAtIndex(surfaceGrid, index);
	NumericBoolean insideWholeShape = numericEqual_uint32_t(indexCurrentValue, SDF_INSIDE_SURFACE);
	NumericBoolean notInsideWholeShape = numericNegate_uint32_t(insideWholeShape);

	NumericBoolean shouldGeneratePointSurface = numericGreaterThan_float(distance, -cellDimension) * isInside;
	NumericBoolean shouldNotGeneratePointSurface = numericNegate_uint32_t(shouldGeneratePointSurface);

	uint32_t ifPointIsLegalSurface = SDF_ON_SURFACE * notInsideWholeShape + SDF_INSIDE_SURFACE * insideWholeShape;
	uint32_t ifPointIsIllegalSurface = indexCurrentValue * isOutside + SDF_INSIDE_SURFACE * isInside;
	uint32_t valueToWriteSurface = ifPointIsIllegalSurface * shouldNotGeneratePointSurface + ifPointIsLegalSurface *  shouldGeneratePointSurface;

	byteArray_setValueAtIndex(surfaceGrid, index, valueToWriteSurface);

	uint32_t materialCurrentValue = byteArray_getValueAtIndex(materialGrid, index);
	uint32_t valueToWriteMaterial = materialCurrentValue * isOutside + material *  isInside;

	byteArray_setValueAtIndex(materialGrid, index, valueToWriteMaterial);
}

__global__ void 
placeSphereMaterialDistanceField(ByteArrayChunk *materialGrid, ByteArrayChunk *surfaceGrid, SDSphere *primitive, uint32_t material, uint32_t gridResolution)
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

	placePoint(materialGrid, surfaceGrid, distance, cellDimension, index, material);
}

__global__ void
placeTorusMaterialDistanceField(ByteArrayChunk *materialGrid, ByteArrayChunk *surfaceGrid, SDTorus *primitive, uint32_t material, uint32_t gridResolution)
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
	float distance = sdfDistanceFromTorus(glm::vec4(normalizeX, normalizeY, normalizeZ, 1), primitive->transform, primitive->dimensions);

	placePoint(materialGrid, surfaceGrid, distance, cellDimension, index, material);
}

__global__ void
placeCubeMaterialDistanceField(ByteArrayChunk *materialGrid, ByteArrayChunk *surfaceGrid, SDCube *primitive, uint32_t material, uint32_t gridResolution)
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
	float distance = sdfDistanceFromCube(glm::vec4(normalizeX, normalizeY, normalizeZ, 1), primitive->transform, primitive->corner);

	placePoint(materialGrid, surfaceGrid, distance, cellDimension, index, material);
}

__device__ __inline__ float
carvePoint(ByteArrayChunk *materialGrid, ByteArrayChunk *surfaceGrid, float distance, float cellDimension, uint32_t index, uint32_t material)
{
	NumericBoolean isOutside = numericGreaterThan_float(distance, 0);
	NumericBoolean isInside = numericNegate_uint32_t(isOutside);

	uint32_t indexCurrentValue = byteArray_getValueAtIndex(surfaceGrid, index);
	NumericBoolean insideWholeShape = numericEqual_uint32_t(indexCurrentValue, SDF_INSIDE_SURFACE);
	NumericBoolean notInsideWholeShape = numericNegate_uint32_t(insideWholeShape);

	NumericBoolean shouldGeneratePointSurface = numericGreaterThan_float(distance, -cellDimension) * isInside;
	NumericBoolean shouldNotGeneratePointSurface = numericNegate_uint32_t(shouldGeneratePointSurface);

	uint32_t ifPointIsLegalSurface = indexCurrentValue * notInsideWholeShape + SDF_ON_SURFACE * insideWholeShape;
	uint32_t ifPointIsIllegalSurface = indexCurrentValue * isOutside + SDF_OUTSIDE_SURFACE * isInside;
	uint32_t valueToWriteSurface = ifPointIsIllegalSurface * shouldNotGeneratePointSurface + ifPointIsLegalSurface *  shouldGeneratePointSurface;

	byteArray_setValueAtIndex(surfaceGrid, index, valueToWriteSurface);

	//uint32_t materialCurrentValue = byteArray_getValueAtIndex(materialGrid, index);
	//uint32_t valueToWriteMaterial = materialCurrentValue * isOutside + materialCurrentValue * isInside * shouldNotGeneratePointSurface;

	//byteArray_setValueAtIndex(materialGrid, index, valueToWriteMaterial);
}

__global__ void
carveSphereDistanceField(ByteArrayChunk *materialGrid, ByteArrayChunk *surfaceGrid, SDSphere *primitive, uint32_t material, uint32_t gridResolution)
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

	carvePoint(materialGrid, surfaceGrid, distance, cellDimension, index, material);
}

__global__ void
carveTorusDistanceField(ByteArrayChunk *materialGrid, ByteArrayChunk *surfaceGrid, SDTorus *primitive, uint32_t material, uint32_t gridResolution)
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
	float distance = sdfDistanceFromTorus(glm::vec4(normalizeX, normalizeY, normalizeZ, 1), primitive->transform, primitive->dimensions);

	carvePoint(materialGrid, surfaceGrid, distance, cellDimension, index, material);
}

__global__ void
carveCubeDistanceField(ByteArrayChunk *materialGrid, ByteArrayChunk *surfaceGrid, SDCube *primitive, uint32_t material, uint32_t gridResolution)
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
	float distance = sdfDistanceFromCube(glm::vec4(normalizeX, normalizeY, normalizeZ, 1), primitive->transform, primitive->corner);

	carvePoint(materialGrid, surfaceGrid, distance, cellDimension, index, material);
}

SignedDistanceField::SignedDistanceField(uint32_t gridResolution) :
	gridResolution(gridResolution),
	materialBlockSize((gridResolution + (SDF_THREAD_BLOCK_DIM - 1)) / SDF_THREAD_BLOCK_DIM, (gridResolution + (SDF_THREAD_BLOCK_DIM - 1)) / SDF_THREAD_BLOCK_DIM, (gridResolution + (SDF_THREAD_BLOCK_DIM - 1)) / SDF_THREAD_BLOCK_DIM),
	materialThreadSize(SDF_THREAD_BLOCK_DIM, SDF_THREAD_BLOCK_DIM, SDF_THREAD_BLOCK_DIM),
	materialGrid(*(new ByteArray(gridResolution * gridResolution * gridResolution))),
	surfaceGrid(*(new ByteArray(gridResolution * gridResolution * gridResolution)))
{
	materialGrid.zero();
	surfaceGrid.zero();
}

void
SignedDistanceField::place(DistancePrimitive& primitive, uint32_t material)
{
	if (SDSphere* v = dynamic_cast<SDSphere*>(&primitive))
	{
		SDSphere* devicePrimitive = (SDSphere*)primitive.copyToDevice();
		placeSphereMaterialDistanceField << <materialBlockSize, materialThreadSize >> >(materialGrid.getDevicePointer(), surfaceGrid.getDevicePointer(), devicePrimitive, material, gridResolution);
		assertCUDA(cudaFree(devicePrimitive));
	}
	else if (SDTorus* v = dynamic_cast<SDTorus*>(&primitive))
	{
		SDTorus* devicePrimitive = (SDTorus*)primitive.copyToDevice();
		placeTorusMaterialDistanceField << <materialBlockSize, materialThreadSize >> >(materialGrid.getDevicePointer(), surfaceGrid.getDevicePointer(), devicePrimitive, material, gridResolution);
		assertCUDA(cudaFree(devicePrimitive));
	}
	else if (SDCube* v = dynamic_cast<SDCube*>(&primitive))
	{
		SDCube* devicePrimitive = (SDCube*)primitive.copyToDevice();
		placeCubeMaterialDistanceField << <materialBlockSize, materialThreadSize >> >(materialGrid.getDevicePointer(), surfaceGrid.getDevicePointer(), devicePrimitive, material, gridResolution);
		assertCUDA(cudaFree(devicePrimitive));
	}
}

void
SignedDistanceField::carve(DistancePrimitive& primitive)
{
	if (SDSphere* v = dynamic_cast<SDSphere*>(&primitive))
	{
		SDSphere* devicePrimitive = (SDSphere*)primitive.copyToDevice();
		carveSphereDistanceField << <materialBlockSize, materialThreadSize >> >(materialGrid.getDevicePointer(), surfaceGrid.getDevicePointer(), devicePrimitive, SDF_OUTSIDE_SURFACE, gridResolution);
		assertCUDA(cudaFree(devicePrimitive));
	}
	else if (SDTorus* v = dynamic_cast<SDTorus*>(&primitive))
	{
		SDTorus* devicePrimitive = (SDTorus*)primitive.copyToDevice();
		carveTorusDistanceField << <materialBlockSize, materialThreadSize >> >(materialGrid.getDevicePointer(), surfaceGrid.getDevicePointer(), devicePrimitive, SDF_OUTSIDE_SURFACE, gridResolution);
		assertCUDA(cudaFree(devicePrimitive));
	}
	else if (SDCube* v = dynamic_cast<SDCube*>(&primitive))
	{
		SDCube* devicePrimitive = (SDCube*)primitive.copyToDevice();
		carveCubeDistanceField << <materialBlockSize, materialThreadSize >> >(materialGrid.getDevicePointer(), surfaceGrid.getDevicePointer(), devicePrimitive, SDF_OUTSIDE_SURFACE, gridResolution);
		assertCUDA(cudaFree(devicePrimitive));
	}
}

void
SignedDistanceField::copyInto(SignedDistanceField& other)
{
	materialGrid.copyInto(other.materialGrid);
	surfaceGrid.copyInto(other.surfaceGrid);
}

ByteArrayChunk*
SignedDistanceField::getMaterialDevicePointer()
{
	return materialGrid.getDevicePointer();
}

ByteArrayChunk*
SignedDistanceField::getSurfaceDevicePointer()
{
	return surfaceGrid.getDevicePointer();
}
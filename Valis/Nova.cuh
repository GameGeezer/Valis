#ifndef VALIS_NOVA_CUH
#define VALIS_NOVA_CUH

#include <stdint.h>
#include <thrust/device_vector.h>

#include "CudaGLBufferMapping.cuh"
#include "CompactMortonPoint.cuh"

#define NOVA_PARSE_BLOCK_SIZE 256

class VBO;
class PBO;
class IBO;
class SignedDistanceField;
class DistancePrimitive;
class SDModification;
class ByteArrayChunk;

class Nova
{
public:

	thrust::device_ptr<CompactMortonPoint> vboPointerDevice;
	CompactMortonPoint *vboPointerRaw;

	thrust::device_ptr<uint32_t> iboPointerDevice;
	uint32_t *iboPointerRaw;

	thrust::device_ptr<uint32_t> pboPointerDevice;
	uint32_t *pboPointerRaw;

	uint32_t vboBufferLength, iboBufferLength, pboBufferLength, parseVBOBlockSize, parseIBOBlockSize, parsePBOBlockSize;

	Nova(VBO &vbo, PBO &pbo, IBO &ibo, uint32_t gridResolution);

	~Nova();

	void
	place(DistancePrimitive& primitive, uint32_t material);

	void
	carve(DistancePrimitive& primitive);

	void
	revertEdits();

	void
	finalizeEdits();

	void
	map();

	void
	unmap();

	void
	clean();

	thrust::device_ptr<CompactMortonPoint>*
	getDeviceVBO();

	thrust::device_ptr<uint32_t>*
	getDevicePBO();

	thrust::device_ptr<uint32_t>*
	getDeviceIBO();

	CompactMortonPoint*
	getRawVBO();

	uint32_t*
	getRawPBO();

	uint32_t*
	getRawIBO();

	uint32_t
	getLengthVBO();

	uint32_t
	getLengthPBO();

	uint32_t
	getLengthIBO();

	uint32_t
	getBlockSizeVBO();

	uint32_t
	getBlockSizePBO();

	uint32_t
	getBlockSizeIBO();

	ByteArrayChunk*
	getMaterialDevicePointer();

	ByteArrayChunk*
	getSurfaceDevicePointer();

private:

	VBO &vbo;
	PBO &pbo;
	IBO &ibo;

	SignedDistanceField &baseField, &editedField;

	CudaGLBufferMapping<CompactMortonPoint> &compactVerticesVBO;
	CudaGLBufferMapping<uint32_t> &vertexOffsetsPBO;
	CudaGLBufferMapping<uint32_t> &indicesIBO;
};

#endif
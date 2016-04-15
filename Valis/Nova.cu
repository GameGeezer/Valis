#include "Nova.cuh"

#include "VBO.cuh"
#include "PBO.cuh"
#include "IBO.cuh"

Nova::Nova(VBO &vbo, PBO &pbo, IBO &ibo) :
	vbo(vbo),
	pbo(pbo),
	ibo(ibo),
	indicesIBO(*(new CudaGLBufferMapping<uint32_t>(ibo.getHandle(), cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone))),
	vertexOffsetsPBO(*(new CudaGLBufferMapping<uint32_t>(pbo.getHandle(), cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone))),
	compactVerticesVBO(*(new CudaGLBufferMapping<CompactMortonPoint>(vbo.getHandle(), cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone)))
{
	compactVerticesVBO.map();
	vertexOffsetsPBO.map();
	indicesIBO.map();

	vboBufferLength = compactVerticesVBO.getSizeInBytes() / sizeof(CompactMortonPoint);
	vboPointerRaw = thrust::raw_pointer_cast(compactVerticesVBO.getDeviceOutput());
	vboPointerDevice = thrust::device_pointer_cast(compactVerticesVBO.getDeviceOutput());
	parseVBOBlockSize = (vboBufferLength + (NOVA_PARSE_BLOCK_SIZE - 1)) / NOVA_PARSE_BLOCK_SIZE;

	pboBufferLength = vertexOffsetsPBO.getSizeInBytes() / sizeof(uint32_t);
	pboPointerRaw = thrust::raw_pointer_cast(vertexOffsetsPBO.getDeviceOutput());
	pboPointerDevice = thrust::device_pointer_cast(vertexOffsetsPBO.getDeviceOutput());
	parsePBOBlockSize = (pboBufferLength + (NOVA_PARSE_BLOCK_SIZE - 1)) / NOVA_PARSE_BLOCK_SIZE;

	iboBufferLength = indicesIBO.getSizeInBytes() / sizeof(uint32_t);
	iboPointerRaw = thrust::raw_pointer_cast(indicesIBO.getDeviceOutput());
	iboPointerDevice = thrust::device_pointer_cast(indicesIBO.getDeviceOutput());
	parseIBOBlockSize = (iboBufferLength + (NOVA_PARSE_BLOCK_SIZE - 1)) / NOVA_PARSE_BLOCK_SIZE;

	compactVerticesVBO.unmap();
	vertexOffsetsPBO.unmap();
	indicesIBO.unmap();
}

void
Nova::startEdit()
{
	compactVerticesVBO.map();
	vertexOffsetsPBO.map();
	indicesIBO.map();
}

void
Nova::endEdit()
{
	compactVerticesVBO.unmap();
	vertexOffsetsPBO.unmap();
	indicesIBO.unmap();
}

void
Nova::clean()
{
	thrust::fill(vboPointerDevice, vboPointerDevice + vboBufferLength, CompactMortonPoint());
	thrust::fill(pboPointerDevice, pboPointerDevice + pboBufferLength, 0);
	thrust::fill(iboPointerDevice, iboPointerDevice + iboBufferLength, 0);
}

thrust::device_ptr<CompactMortonPoint>*
Nova::getDeviceVBO()
{
	return &vboPointerDevice;
}

thrust::device_ptr<uint32_t>*
Nova::getDevicePBO()
{
	return &pboPointerDevice;
}

thrust::device_ptr<uint32_t>*
Nova::getDeviceIBO()
{
	return &iboPointerDevice;
}

CompactMortonPoint*
Nova::getRawVBO()
{
	return vboPointerRaw;
}

uint32_t*
Nova::getRawPBO()
{
	return pboPointerRaw;
}

uint32_t*
Nova::getRawIBO()
{
	return iboPointerRaw;
}

uint32_t
Nova::getLengthVBO()
{
	return vboBufferLength;
}

uint32_t
Nova::getLengthPBO()
{
	return pboBufferLength;
}

uint32_t
Nova::getLengthIBO()
{
	return iboBufferLength;
}

uint32_t
Nova::getBlockSizeVBO()
{
	return parseVBOBlockSize;
}

uint32_t
Nova::getBlockSizePBO()
{
	return parsePBOBlockSize;
}

uint32_t
Nova::getBlockSizeIBO()
{
	return parseIBOBlockSize;
}
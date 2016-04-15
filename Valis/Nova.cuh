#ifndef VALIS_NOVA_CUH
#define VALIS_NOVA_CUH

#include <glm/mat4x4.hpp>
#include <stdint.h>

#include "CudaGLBufferMapping.cuh";
#include "CompactMortonPoint.cuh";

class Nova
{
public:

	Nova(uint32_t resolution);

	~Nova();

private:
	CudaGLBufferMapping<CompactMortonPoint>* compactVerticesVBO;
	CudaGLBufferMapping<uint32_t>* vertexOffsetsPBO;
	CudaGLBufferMapping<uint32_t>* indicesIBO;
};

#endif
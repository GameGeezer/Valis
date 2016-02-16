#ifndef VALIS_SDFRENDERER
#define VALIS_SDFRENDERER

#include <glm\mat4x4.hpp>

class CudaGLBufferMapping;

class SDFRenderer
{
public:

	void
	renderToMapping(CudaGLBufferMapping& mapping, dim3 windowGridSize, dim3 windowBlockSize, glm::mat4& inverseViewProjection);
};

#endif
#ifndef VALIS_VBO_CUH
#define VALIS_VBO_CUH

#include "GLLibraries.h"

#include "BufferedObjectUsage.cuh"

using namespace std;

class VBO
{
public:

	VBO(void* data, size_t size, BufferedObjectUsage usage);

	VBO(size_t size, BufferedObjectUsage usage);

	~VBO();

	void
	bind();

	void
	unbind();

	GLuint
	getHandle();

private:
	GLuint handle;
};

#endif
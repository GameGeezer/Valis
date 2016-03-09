#include "IBO.cuh"

IBO::IBO(void* data, size_t size, BufferedObjectUsage usage)
{
	glGenBuffers(1, &handle);

	glBindBuffer(GL_ARRAY_BUFFER, handle);

	glBufferData(GL_ARRAY_BUFFER, size * sizeof(int), data, usage);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

IBO::IBO(size_t size, BufferedObjectUsage usage)
{
	glGenBuffers(1, &handle);

	glBindBuffer(GL_ARRAY_BUFFER, handle);

	glBufferData(GL_ARRAY_BUFFER, size * sizeof(int), NULL, usage);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

IBO::~IBO()
{
	glDeleteBuffers(1, &handle);
}

inline void
IBO::bind()
{
	glBindBuffer(GL_ARRAY_BUFFER, handle);
}

inline void
IBO::unbind()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

inline GLuint
IBO::getHandle()
{
	return handle;
}
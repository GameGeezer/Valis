#include "IBO.cuh"

IBO::IBO(void* data, size_t size, BufferedObjectUsage usage)
{
	glGenBuffers(1, &handle);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle);

	glBufferData(GL_ELEMENT_ARRAY_BUFFER, size * sizeof(int), data, usage);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

IBO::IBO(size_t size, BufferedObjectUsage usage)
{
	glGenBuffers(1, &handle);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle);

	glBufferData(GL_ELEMENT_ARRAY_BUFFER, size * sizeof(int), NULL, usage);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

IBO::~IBO()
{
	glDeleteBuffers(1, &handle);
}

void
IBO::bind()
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle);
}

void
IBO::unbind()
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

GLuint
IBO::getHandle()
{
	return handle;
}
#include "VBO.cuh"

VBO::VBO(void* data, size_t size, BufferedObjectUsage usage)
{
	glGenBuffers(1, &handle);

	glBindBuffer(GL_ARRAY_BUFFER, handle);

	glBufferData(GL_ARRAY_BUFFER, size * sizeof(float), data, usage);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

VBO::VBO(size_t size, BufferedObjectUsage usage)
{
	glGenBuffers(1, &handle);

	glBindBuffer(GL_ARRAY_BUFFER, handle);

	glBufferData(GL_ARRAY_BUFFER, size * sizeof(float), NULL, usage);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

VBO::~VBO()
{
	glDeleteBuffers(1, &handle);
}

void
VBO::bind()
{
	glBindBuffer(GL_ARRAY_BUFFER, handle);
}

void
VBO::unbind()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

GLuint
VBO::getHandle()
{
	return handle;
}
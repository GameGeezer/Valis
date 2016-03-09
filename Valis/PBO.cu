#include "PBO.cuh"

PBO::PBO(int bytesToAllocate) : sizeBytes(bytesToAllocate)
{
	glGenBuffers(1, &handle);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, handle);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(GLubyte) * bytesToAllocate, 0, GL_STREAM_DRAW_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

PBO::~PBO()
{
	glDeleteBuffers(1, &handle);
}

void
PBO::bind()
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, handle);
}

void
PBO::unbind()
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

GLuint
PBO::getHandle()
{
	return handle;
}
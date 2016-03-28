#include "Texture1D.cuh"

Texture1D::Texture1D(int width) : width(width)
{
	glGenTextures(1, &handle);
	glBindTexture(GL_TEXTURE_1D, handle);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_R32UI, width, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Don't sample the border color.
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_1D, 0);
}

Texture1D::~Texture1D()
{
	glDeleteTextures(1, &handle);
}

void
Texture1D::bind()
{
	glBindTexture(GL_TEXTURE_1D, handle);
}

void
Texture1D::unbind()
{
	glBindTexture(GL_TEXTURE_1D, 0);
}

GLuint
Texture1D::getHandle()
{
	return handle;
}
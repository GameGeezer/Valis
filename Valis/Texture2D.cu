#include "Texture2D.cuh"

Texture2D::Texture2D(int width, int height) : width(width), height(height)
{
	glGenTextures(1, &handle);
	glBindTexture(GL_TEXTURE_2D, handle);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

Texture2D::~Texture2D()
{
	glDeleteTextures(1, &handle);
}

void
Texture2D::bind()
{
	glBindTexture(GL_TEXTURE_2D, handle);
}

void
Texture2D::unbind()
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

GLuint
Texture2D::getHandle()
{
	return handle;
}
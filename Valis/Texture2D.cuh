#ifndef VALIS_TEXTURE2D_H
#define VALIS_TEXTURE2D_H

#include "GLLibraries.h"

class Texture2D
{
public:

	const int width, height;

	Texture2D(int width, int height) : width(width), height(height)
	{
		glGenTextures(1, &handle);
		glBindTexture(GL_TEXTURE_2D, handle);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	~Texture2D()
	{

	}

	inline void
	bind()
	{
		glBindTexture(GL_TEXTURE_2D, handle);
	}

	inline void
	unbind()
	{
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	inline GLuint
	getHandle()
	{
		return handle;
	}

private:
	GLuint handle;
};

#endif //VALIS_TEXTURE2D_H
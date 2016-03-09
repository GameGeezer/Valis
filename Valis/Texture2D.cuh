#ifndef VALIS_TEXTURE2D_H
#define VALIS_TEXTURE2D_H

#include "GLLibraries.h"

class Texture2D
{
public:

	const int width, height;

	Texture2D(int width, int height);

	~Texture2D();

	void
	bind();

	void
	unbind();

	GLuint
	getHandle();

private:
	GLuint handle;
};

#endif //VALIS_TEXTURE2D_H
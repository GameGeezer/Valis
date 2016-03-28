#ifndef VALIS_TEXTURE2D_H
#define VALIS_TEXTURE2D_H

#include "GLLibraries.h"

class Texture1D
{
public:

	const int width;

	Texture1D(int width);

	~Texture1D();

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
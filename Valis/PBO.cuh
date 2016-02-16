#ifndef VALIS_PBO_H
#define VALIS_PBO_H

#include "GLLibraries.h"

using namespace std;

class PBO
{
public:

	const size_t sizeBytes;

	PBO(int bytesToAllocate) : sizeBytes(bytesToAllocate)
	{
		glGenBuffers(1, &handle);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, handle);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(GLubyte) * bytesToAllocate, 0, GL_STREAM_DRAW_ARB);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}

	~PBO()
	{
		glDeleteBuffers(1, &handle);
	}

	inline void
	bind()
	{
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, handle);
	}

	inline void
	unbind()
	{
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	}

	inline GLuint
	getHandle()
	{
		return handle;
	}

private:
	GLuint handle;
};

#endif //VALIS_PBO_H
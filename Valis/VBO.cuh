#ifndef VALIS_VBO_CUH
#define VALIS_VBO_CUH

#include <vector>

#include "GLLibraries.h"

using namespace std;

class VBO
{
public:

	VBO(void* data, size_t size)
	{
		glGenBuffers(1, &handle);

		glBindBuffer(GL_ARRAY_BUFFER, handle);

		glBufferData(GL_ARRAY_BUFFER, size * sizeof(float), data, GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	~VBO()
	{
		glDeleteBuffers(1, &handle);
	}

	inline void
	bind()
	{
		glBindBuffer(GL_ARRAY_BUFFER, handle);
	}

	inline void
	unbind()
	{
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	inline GLuint
	getHandle()
	{
		return handle;
	}

private:
	GLuint handle;
};

#endif
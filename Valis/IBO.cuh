#ifndef RAYCAST_IBO_H
#define RAYCAST_IBO_H

#include <vector>

#include "GLLibraries.h"

using namespace std;

class IBO
{
public:

	IBO(vector<int>* data)
	{
		glGenBuffers(1, &handle);

		glBindBuffer(GL_ARRAY_BUFFER, handle);

		glBufferData(GL_ARRAY_BUFFER, data->size() * sizeof(int), data, GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	~IBO()
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


#endif //RAYCAST_IBO_H
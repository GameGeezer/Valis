#ifndef RAYCAST_IBO_H
#define RAYCAST_IBO_H

#include "GLLibraries.h"

#include "BufferedObjectUsage.cuh"

using namespace std;

class IBO
{
public:

	IBO(void* data, size_t size, BufferedObjectUsage usage);

	IBO(size_t size, BufferedObjectUsage usage);

	~IBO();

	void
	bind();

	void
	unbind();

	GLuint
	getHandle();

private:
	GLuint handle;
};


#endif //RAYCAST_IBO_H
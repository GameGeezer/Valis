#ifndef VALIS_PBO_H
#define VALIS_PBO_H

#include "GLLibraries.h"

using namespace std;

class PBO
{
public:

	const size_t sizeBytes;

	PBO(int bytesToAllocate);

	~PBO();

	void
	bind();

	void
	unbind();

	GLuint
	getHandle();

private:
	GLuint handle;
};

#endif //VALIS_PBO_H
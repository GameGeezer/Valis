#ifndef VALIS_VAO_CUH
#define VALIS_VAO_CUH

#include <stdint.h>
#include <map>

#include "GLLibraries.h"

class Descriptor;
class VBO;

using namespace std;

class VAO
{
public:

	VAO(VBO& vbo, size_t size);
	
	void
	init();

	void
	draw();

	void
	destroy();

	void
	addVertexAttribute(uint32_t index, Descriptor& descriptor);

private:
	GLuint handle;
	size_t size;
	map<int, Descriptor*> descriptors;
	VBO* vbo;
};

#endif
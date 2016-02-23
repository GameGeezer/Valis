#ifndef VALIS_VAO_CUH
#define VALIS_VAO_CUH

#include <stdint.h>

class Descriptor;

class VAO
{
public:

	VAO();
	
	void
	init();

	void
	draw();

	void
	destroy();

	void
	addVertexAttribute(uint32_t index, Descriptor& descriptor);
};

#endif
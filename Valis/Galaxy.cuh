#ifndef VALIS_GALAXY_CUH
#define VALIS_GALAXY_CUH

class VBO;

class Galaxy
{
public:
	
	Galaxy();

	~Galaxy();

private:
	VBO* pointsVBO;
};

#endif
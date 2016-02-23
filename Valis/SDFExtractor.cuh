#ifndef VALIS_SDFEXTRACTOR
#define VALIS_SDFEXTRACTOR

#include <thrust/device_vector.h>

class RenderPoint;

class SDFExtractor
{
public:

	SDFExtractor();

	void
	extract();

private:
	thrust::device_vector< RenderPoint >* extractedPoints;
};


#endif
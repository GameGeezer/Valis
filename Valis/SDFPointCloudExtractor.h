
#ifndef VALIS_SDF_POINT_CLOUD_EXTRACTOR_H
#define VALIS_SDF_POINT_CLOUD_EXTRACTOR_H

#include <glm\vec3.hpp>

#include "SignedDistanceFunction.cuh"

class SDFPointCloudExtractor
{
public:
	SDFPointCloudExtractor(glm::vec3 blockSize, glm::vec3 numberOfBlocks);

	void
	extract(SignedDistanceFunction& sdf);

private:
	glm::vec3 blockSize, numberOfBlocks;
};

#endif
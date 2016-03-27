#ifndef TEST_RELATIVE_SCREEN_CUH
#define TEST_RELATIVE_SCREEN_CUH


#include "Screen.cuh"
#include "cuda_runtime.h"

#include "CompactMortonPoint.cuh"
#include "ThreeCompact10BitUInts.cuh"
#include "CudaGLBufferMapping.cuh"

class ShaderProgram;
class Camera;
class Player;
class SDFHilbertExtractor;
class IBO;
class SDFDevice;

typedef ThreeCompact10BitUInts CompactLocation;

class TestRelativeScreen : public Screen
{
	__host__ void
	onCreate() override;

	__host__ void
	onPause() override;

	__host__ void
	onLeave() override;

	__host__ void
	onResume() override;

	__host__ void
	onUpdate(int delta) override;

	__host__ void
	onResize(int width, int height) override;

	__host__ void
	onDestroy() override;
private:
	ShaderProgram* shader;
	Player* player;
	SDFHilbertExtractor* extractor;
	SDFDevice* testSDFDevice;
	CudaGLBufferMapping<CompactMortonPoint>* mapping;
	CudaGLBufferMapping<uint64_t>* pboMapping;
	PBO* pbo;
	IBO* ibo;
	int pointCount;
};


#endif
#ifndef TEST_RELATIVE_SCREEN_CUH
#define TEST_RELATIVE_SCREEN_CUH


#include "Screen.cuh"
#include "cuda_runtime.h"

#include "CompactRenderPoint.cuh"
#include "CudaGLBufferMapping.cuh"

class ShaderProgram;
class Camera;
class Player;
class SDFRelativeExtractor;
class VBO;
class SDFDevice;

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
	SDFRelativeExtractor* extractor;
	SDFDevice* testSDFDevice;
	CudaGLBufferMapping<CompactRenderPoint>* mapping;
	VBO* vbo;
	int pointCount;
};


#endif
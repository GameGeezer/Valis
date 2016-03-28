#ifndef VALIS_TEST_SCREEN_H
#define VALIS_TEST_SCREEN_H

#include "Screen.cuh"
#include "cuda_runtime.h"

#include "CompactRenderPoint.cuh"
#include "ThreeCompact10BitUInts.cuh"
#include "CudaGLBufferMapping.cuh"


#include "Screen.cuh"
#include "cuda_runtime.h"

#include "CompactRenderPoint.cuh"
#include "ThreeCompact10BitUInts.cuh"
#include "CudaGLBufferMapping.cuh"

class ShaderProgram;
class Camera;
class Player;
class SDFRelativeExtractor;
class IBO;
class SDFDevice;
class Texture1D;

class TestScreen : public Screen
{
	void
		onCreate() override;

	void
		onPause() override;

	void
		onLeave() override;

	void
		onResume() override;

	void
		onUpdate(int delta) override;

	void
		onResize(int width, int height) override;

	void
		onDestroy() override;
private:
	ShaderProgram* shader;
	Player* player;
	SDFRelativeExtractor* extractor;
	SDFDevice* testSDFDevice;
	CudaGLBufferMapping<CompactRenderPoint>* mapping;
	CudaGLBufferMapping<ThreeCompact10BitUInts>* pboMapping;
	PBO* pbo;
	IBO* ibo;
	Texture1D* pboTexture;
	int pointCount;
};

#endif
#ifndef VALIS_TEST_SCREEN_H
#define VALIS_TEST_SCREEN_H

#include "Screen.cuh"
#include "cuda_runtime.h"


class ShaderProgram;
class Camera;
class Player;
class SDFExtractor;
class VBO;
class SDFDevice;

class TestScreen : public Screen
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
	SDFExtractor* extractor;
	SDFDevice* testSDFDevice;
	VBO* vbo;
	int pointCount;
};

#endif
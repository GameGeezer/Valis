#ifndef VALIS_TEST_SCREEN_H
#define VALIS_TEST_SCREEN_H

#include "Screen.cuh"
#include "GLLibraries.h"

class PBO;
class Texture2D;
class CudaGLBufferMapping;
class dim3;
class SDFRenderer;
class ShaderProgram;
class Camera;
class Player;

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
	GLuint vertexbuffer;
	SDFRenderer* renderer;
	PBO *pbo;
	Texture2D *texture;
	CudaGLBufferMapping *mapping;
	dim3 *windowBlockSize;
	dim3 *windowGridSize;
	ShaderProgram* shader;
	Player* player;
};

#endif
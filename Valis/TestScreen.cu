#include "TestScreen.cuh"

#include <map>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext.hpp>

#include "SDFExtractor.cuh"

#include <iostream>

#include "cuda_runtime.h"

#include "SDFRenderer.cuh"
#include "PBO.cuh"
#include "Texture2D.cuh"
#include "CudaGLBufferMapping.cuh"
#include "Matrix4Device.cuh"
#include "ShaderProgram.cuh"
#include "Camera.cuh"
#include "Player.cuh"
#include "SignedDistanceFunction.cuh"
#include "SDSphere.cuh"
#include "Descriptor.cuh"
#include "SDFExtractor.cuh"


void
TestScreen::onCreate()
{
	renderer = new SDFRenderer();
	pbo = new PBO(4 * 640 * 480);
	texture = new Texture2D(640, 480);
	mapping = new CudaGLBufferMapping(*pbo);
	windowBlockSize = new dim3(16, 16, 1);
	windowGridSize = new dim3(640 / windowBlockSize->x, 480 / windowBlockSize->y);

	Camera* camera = new Camera(640, 680, 0.1f, 1000.0f, 45.0f);
	camera->translate(0, 0, 1);

	sphereSdf = new SignedDistanceFunction();
	sphereSdf->addFunction(*(new SDSphere(1, glm::vec3(0, 0, 0))));

	extractor = new SDFExtractor();
	
	player = new Player(*camera);

}

void
TestScreen::onPause()
{

}

void
TestScreen::onLeave()
{

}

void
TestScreen::onResume()
{

}

void
TestScreen::onUpdate(int delta)
{
	extractor->extract();

	glm::mat4 invViewProjection;
	player->camera->constructInverseViewProjection(invViewProjection);
	renderer->renderToMapping(*mapping, *windowGridSize, *windowBlockSize, invViewProjection);
	pbo->bind();

	glDrawPixels(texture->width, texture->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture->width, texture->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	pbo->unbind();
}

void
TestScreen::onResize(int width, int height)
{

}

void
TestScreen::onDestroy()
{

}
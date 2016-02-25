#include "TestScreen.cuh"

#include <map>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext.hpp>

#include "SDFExtractor.cuh"

#include <iostream>

#include "cuda_runtime.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
#include "VAO.cuh"
#include "RenderPoint.cuh"
#include "VBO.cuh"

#include <iostream>
#include <fstream>
#include <sstream>


void
TestScreen::onCreate()
{
	renderer = new SDFRenderer();
	pbo = new PBO(4 * 640 * 480);
	texture = new Texture2D(640, 480);
	mapping = new CudaGLBufferMapping(*pbo);
	windowBlockSize = new dim3(16, 16, 1);
	windowGridSize = new dim3(640 / windowBlockSize->x, 480 / windowBlockSize->y);
	
	ifstream myfile("BasicShader.vert");
	std::stringstream buffer;
	buffer << myfile.rdbuf();
	string vertShader = buffer.str();

	ifstream myfile2("BasicShader.frag");
	std::stringstream buffer2;
	buffer2 << myfile2.rdbuf();
	string fragShader = buffer2.str();

	map<int, char *> attributes;
	attributes.insert(pair<int, char*>(0, "in_Position"));
	shader = new ShaderProgram(vertShader.c_str(), fragShader.c_str(), attributes);
	Camera* camera = new Camera(640, 680, 0.1f, 100.0f, 45.0f);
	camera->translate(0, 0, 2);

	sphereSdf = new SignedDistanceFunction();
	sphereSdf->addFunction(*(new SDSphere(1, glm::vec3(0, 0, 0))));

	extractor = new SDFExtractor();
	thrust::device_vector< RenderPoint >* spherePoints = extractor->extract();
	thrust::host_vector< RenderPoint > hostPoints = *spherePoints;
	pointCount = hostPoints.size();
	vbo = new VBO(&(hostPoints[0]), hostPoints.size() * 3);

	//vao = new VAO(*vbo, pointCount / 3);
	//vao->addVertexAttribute(0, Descriptor(3, GL_FLOAT, false, sizeof(RenderPoint), 0));
	//vao->init();
	
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
	
	glm::mat4 invViewProjection;
	player->camera->constructInverseViewProjection(invViewProjection);
	renderer->renderToMapping(*mapping, *windowGridSize, *windowBlockSize, invViewProjection);
	pbo->bind();

	//glDrawPixels(texture->width, texture->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture->width, texture->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	pbo->unbind();

	glm::mat4 viewProjection;
	player->camera->constructViewProjection(viewProjection);

	shader->bind();

	GLint projectionLocation = shader->getUniformLocation("projectionMatrix");
	shader->setUnifromMatrix4f(projectionLocation, viewProjection);

	vbo->bind();
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(RenderPoint), (void*)(sizeof(float) * 0));
	//glVertexPointer(3, GL_FLOAT, sizeof(RenderPoint), (void*)(sizeof(float) * 0));
	//glVertexAttribPointer(0, sizeof(RenderPoint), GL_FLOAT, false, 0, 0);
	glDrawArrays(GL_POINTS, 0, pointCount);
	glDisableClientState(GL_VERTEX_ARRAY);
	vbo->unbind();
	//vao->draw();
	shader->unbind();
}

void
TestScreen::onResize(int width, int height)
{

}

void
TestScreen::onDestroy()
{

}
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

#include "ShaderProgram.cuh"
#include "Camera.cuh"
#include "Player.cuh"
#include "SDSphere.cuh"
#include "Descriptor.cuh"
#include "SDFExtractor.cuh"
#include "RenderPoint.cuh"
#include "VBO.cuh"

#include "SDFHost.cuh"
#include "SDFDevice.cuh"
#include "PlaceSDPrimitive.cuh"

#include <iostream>
#include <fstream>
#include <sstream>


void
TestScreen::onCreate()
{
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

	extractor = new SDFExtractor(200, 50);
	SDSphere sdSphere(0.25f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDTorus sdTorus(0.31f, 0.1f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDModification* place = new PlaceSDPrimitive();
	SDFHost* testSDF = new SDFHost(&sdSphere);
	testSDF->modify(&sdTorus, place);
	SDFDevice* testSDFDevice = testSDF->copyToDevice();

	//thrust::host_vector< RenderPoint >& hostPoints = *(extractor->extract(*testSDFDevice));
	//pointCount = hostPoints.size();
	//vbo = new VBO(&(hostPoints[0]), hostPoints.size() * 3);
	vbo = new VBO(6000000);
	pointCount = extractor->extractDynamic(*testSDFDevice, *vbo);
	
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
	player->update(delta);
	glm::mat4 invViewProjection;
	player->camera->constructInverseViewProjection(invViewProjection);

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
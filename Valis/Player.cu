#include "Player.cuh"

#include "GLLibraries.h"

#include <glm\glm.hpp>
#include <glm\vec4.hpp>


#include "Camera.cuh"
#include "Application.cuh"

#include "SDFHost.cuh"
#include "SDFDevice.cuh"
#include "PlaceSDPrimitive.cuh"
#include "CarveSDPrimitive.cuh"
#include "BlendSDModification.cuh"
#include "SDModification.cuh"

#include "VBO.cuh"
#include "IBO.cuh"
#include "PBO.cuh"

#include "Nova.cuh"

Player::Player(Camera& camera) : camera(&camera), scale(glm::vec3(1, 1, 1))
{
	Application::KEYBOARD->addListener(*this);
	Application::MOUSE_CLICK->addListener(*this);
	Application::MOUSE_MOVE->addListener(*this);

	SDSphere sdSphere(0.25f, glm::vec3(1, 1, 1), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0, 1, 0), 0);
	SDCube sdCube(glm::vec3(0.1f, 0.1f, 0.1f), glm::vec3(1, 1, 1), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0, 1, 0), 0);
	ibo = new IBO(1000000, BufferedObjectUsage::DYNAMIC_DRAW);
	vbo = new VBO(500000, BufferedObjectUsage::DYNAMIC_DRAW);
	pbo = new PBO(160000);

	testNova = new Nova(*vbo, *pbo, *ibo, 128);
	testNova->place(sdCube, 3);
	testNova->finalizeEdits();

	rotation = glm::vec3(0, 1, 0);
}

void
Player::update(int delta)
{
	if (isDPressed)
		camera->moveRight(0.04f);
	if (isAPressed)
		camera->moveLeft(0.04f);
	if (isWPressed)
		camera->moveForward(0.04f);
	if (isSPressed)
		camera->moveBackward(0.04f);

	if (isOPressed)
		scale -= 0.05f;
	if (isPPressed)
		scale += 0.05f;

	if (isQPressed)
		distanceFromCamera -= 0.05;

	if (isEPressed)
		distanceFromCamera += 0.05;

	glm::vec3 position = glm::vec3(glm::inverse(*(camera->getView())) * glm::vec4(0, 0, -distanceFromCamera, 1));
	orientation = glm::lookAt(position, position + camera->getDirection(), glm::vec3(0, 1, 0));

	testNova->revertEdits();

	if (brushType == 1)
	{
		SDSphere sdSphere(0.1f, scale, orientation);
		testNova->place(sdSphere, materialType);
	}
	else if (brushType == 2)
	{
		SDTorus sdTorus(0.1f, 0.025f, scale, orientation);
		testNova->place(sdTorus, materialType);
	}
	else if(brushType == 3)
	{
		SDCube sdCube(glm::vec3(0.1f, 0.1f, 0.1f), scale, orientation);
		testNova->place(sdCube, materialType);
	}
	else if (brushType == 4)
	{
		SDSphere sdSphere(0.1f, scale, orientation);
		testNova->carve(sdSphere);
	}
	else if (brushType == 5)
	{
		SDTorus sdTorus(0.1f, 0.025f, scale, orientation);
		testNova->carve(sdTorus);
	}
	else if (brushType == 6)
	{
		SDCube sdCube(glm::vec3(0.1f, 0.1f, 0.1f), scale, orientation);
		testNova->carve(sdCube);
	}

	if (isRightMousePressed)
	{
		testNova->finalizeEdits();
	}
}

void
Player::onKeyRelease(int keyCode)
{
	if (keyCode == GLFW_KEY_D)
		isDPressed = false;
	if (keyCode == GLFW_KEY_A)
		isAPressed = false;
	if (keyCode == GLFW_KEY_W)
		isWPressed = false;
	if (keyCode == GLFW_KEY_S)
		isSPressed = false;

	if (keyCode == GLFW_KEY_K)
		isKPressed = false;
	if (keyCode == GLFW_KEY_L)
		isLPressed = false;

	if (keyCode == GLFW_KEY_O)
		isOPressed = false;
	if (keyCode == GLFW_KEY_P)
		isPPressed = false;

	if (keyCode == GLFW_KEY_Q)
		isQPressed = false;
	if (keyCode == GLFW_KEY_E)
		isEPressed = false;
}

void
Player::onKeyPress(int keyCode)
{
	if (keyCode == GLFW_KEY_D)
		isDPressed = true;
	if (keyCode == GLFW_KEY_A)
		isAPressed = true;
	if (keyCode == GLFW_KEY_W)
		isWPressed = true;
	if (keyCode == GLFW_KEY_S)
		isSPressed = true;

	if (keyCode == GLFW_KEY_K)
		isKPressed = true;
	if (keyCode == GLFW_KEY_L)
		isLPressed = true;

	if (keyCode == GLFW_KEY_O)
		isOPressed = true;
	if (keyCode == GLFW_KEY_P)
		isPPressed = true;

	if (keyCode == GLFW_KEY_Q)
		isQPressed = true;
	if (keyCode == GLFW_KEY_E)
		isEPressed = true;

	if (keyCode == GLFW_KEY_J)
		materialType = 4;
	if (keyCode == GLFW_KEY_L)
		materialType = 3;
	if (keyCode == GLFW_KEY_K)
		materialType = 2;
	if (keyCode == GLFW_KEY_L)
		materialType = 3;

	if (keyCode == GLFW_KEY_1)
		brushType = 1;
	if (keyCode == GLFW_KEY_2)
		brushType = 2;
	if (keyCode == GLFW_KEY_3)
		brushType = 3;
	if (keyCode == GLFW_KEY_4)
		brushType = 4;
	if (keyCode == GLFW_KEY_5)
		brushType = 5;
	if (keyCode == GLFW_KEY_6)
		brushType = 6;
	if (keyCode == GLFW_KEY_7)
		brushType = 7;

}

void
Player::onKeyRepeat(int keyCode)
{

}

void
Player::onMouseRelease(MouseButton button, float posX, float posY)
{
	if (button == MouseButton::LEFT)
	{
		isLeftMousePressed = false;
	}
	if (button == MouseButton::RIGHT)
	{
		isRightMousePressed = false;
	}

	
}

void
Player::onMousePress(MouseButton button, float posX, float posY)
{
	lastMousePosition = glm::vec2(posX, posY);
	lastMouseClickPosition = glm::vec2(posX, posY);

	if (button == MouseButton::LEFT)
	{
		isLeftMousePressed = true;
	}

	if (button == MouseButton::RIGHT)
	{
		isRightMousePressed = true;
	}
}

void
Player::onMouseMove(float posX, float posY)
{
	if (isLeftMousePressed)
	{
		camera->rotate( 0.01f * (lastMousePosition.x - posX), glm::vec3(0, 1, 0));
		camera->rotateLocalX(0.01f * (lastMousePosition.y - posY));
	}

	lastMousePosition = glm::vec2(posX, posY);
}
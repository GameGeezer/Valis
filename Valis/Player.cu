#include "Player.cuh"

#include "GLLibraries.h"

#include "Camera.cuh"
#include "Application.cuh"

Player::Player(Camera& camera) : camera(&camera)
{
	Application::KEYBOARD->addListener(*this);
	Application::MOUSE_CLICK->addListener(*this);
	Application::MOUSE_MOVE->addListener(*this);
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
}

void
Player::onKeyRepeat(int keyCode)
{

}

void
Player::onMouseRelease(MouseButton button, float posX, float posY)
{
	isMousePressed = false;
}

void
Player::onMousePress(MouseButton button, float posX, float posY)
{
	lastMousePosition = glm::vec2(posX, posY);
	lastMouseClickPosition = glm::vec2(posX, posY);
	isMousePressed = true;
}

void
Player::onMouseMove(float posX, float posY)
{
	if (isMousePressed)
	{
		camera->rotate( 0.01f * (lastMousePosition.x - posX), glm::vec3(0, 1, 0));
		camera->rotateLocalX(0.01f * (lastMousePosition.y - posY));
	}

	lastMousePosition = glm::vec2(posX, posY);
}
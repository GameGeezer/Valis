#include "Player.cuh"

#include "GLLibraries.h"

#include "Camera.cuh"
#include "Application.cuh"

Player::Player(Camera& camera) : camera(&camera)
{
	Application::KEYBOARD->addListener(*this);
}

void
Player::onKeyRelease(int keyCode)
{

}

void
Player::onKeyPress(int keyCode)
{

}

void
Player::onKeyRepeat(int keyCode)
{
	if (keyCode == GLFW_KEY_D)
		camera->moveRight(0.01f);
	if (keyCode == GLFW_KEY_A)
		camera->moveLeft(0.01f);
	if (keyCode == GLFW_KEY_W)
		camera->moveForward(0.01f);
	if (keyCode == GLFW_KEY_S)
		camera->moveBackward(0.01f);
	if (keyCode == GLFW_KEY_Q)
		//camera->translate(0, 0, 0.01f);
		camera->rotate(0.01f, glm::vec3(0, 1,0));
	if (keyCode == GLFW_KEY_E)
		//camera->translate(0, 0, -0.01f);
		camera->rotate(-0.01f, glm::vec3(0, 1, 0));
}

void
Player::onMouseRelease(MouseButton button, float posX, float posY)
{

}

void
Player::onMousePress(MouseButton button, float posX, float posY)
{

}
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
		camera->translate(10, 0, 0);
	if (keyCode == GLFW_KEY_A)
		camera->translate(-10, 0, 0);
	if (keyCode == GLFW_KEY_W)
		camera->translate(0, 10, 0);
	if (keyCode == GLFW_KEY_S)
		camera->translate(0, -10, 0);
	if (keyCode == GLFW_KEY_Q)
		camera->translate(0, 0, 10);
	if (keyCode == GLFW_KEY_E)
		camera->translate(0, 0, -10);
}
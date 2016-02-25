#ifndef VALIS_APPLICATION_H
#define VALIS_APPLICATION_H

#include <string>
#include <chrono>
#include <thread>
#include <iostream>

#include "KeyboardCallback.cuh"
#include "MouseClickCallback.cuh"

class Game;
class GLFWwindow;

using namespace std;
using namespace std::chrono;

class Application
{
public:
	static KeyboardCallback* KEYBOARD;
	static MouseClickCallback* MOUSE_CLICK;

	Application(Game& game, string windowTitle, int windowWidth, int windowHeight);
	~Application();

	void
	start();

private:
	int windowWidth, windowHeight;
	Game* game;
	string windowTitle;
	GLFWwindow* window;
	milliseconds lastFrameTime;

	void
	init();

	void
	loop();
};



#endif
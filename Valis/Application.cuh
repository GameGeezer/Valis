#ifndef VALIS_APPLICATION_H
#define VALIS_APPLICATION_H

#include <string>

#include "KeyboardCallback.cuh"

class Game;
class GLFWwindow;
class KeyboardListener;

using namespace std;

class Application
{
public:
	static KeyboardCallback* KEYBOARD;

	Application(Game& game, string windowTitle, int windowWidth, int windowHeight);
	~Application();

	void
	start();

private:
	int windowWidth, windowHeight;
	Game* game;
	string windowTitle;
	GLFWwindow* window;

	void
	init();

	void
	loop();
};



#endif
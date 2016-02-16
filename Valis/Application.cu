#include "Application.cuh"
#include "Game.cuh"
#include "GLLibraries.h"
#include "KeyboardCallback.cuh"

KeyboardCallback* Application::KEYBOARD = new KeyboardCallback();

static void error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	Application::KEYBOARD->invoke(key, scancode, action, mods);
}

Application::Application(Game& game, string windowTitle, int windowWidth, int windowHeight) : game(&game), windowTitle(windowTitle), windowWidth(windowWidth), windowHeight(windowHeight)
{

}

Application::~Application()
{

}

void
Application::start()
{
	this->init();

	this->game->onCreate();

	this->loop();
}

void
Application::init()
{
	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
	{
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(windowWidth, windowHeight, windowTitle.c_str(), NULL, NULL);

	if (!window)
	{
		glfwTerminate();

		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);

	glfwSwapInterval(1);

	glfwSetKeyCallback(window, key_callback);

	glewInit();
}

void
Application::loop()
{
	while (!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		game->update(0);

		glfwSwapBuffers(window);

		glfwPollEvents();
	}

	glfwDestroyWindow(window);

	glfwTerminate();
}
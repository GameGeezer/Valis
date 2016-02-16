#include <GLFW/glfw3.h>
#include <vector>
#include <algorithm>

#include "KeyboardCallback.cuh"
#include "KeyboardListener.cuh"

KeyboardCallback::KeyboardCallback()
{

}

void
KeyboardCallback::invoke(int keyCode, int scanCode, int action, int mods)
{
	switch (action)
	{
	case GLFW_REPEAT:
		for (auto listener = listeners.begin(); listener != listeners.end(); ++listener)
		{
			(*listener)->onKeyRepeat(keyCode);
		}
		break;
	case GLFW_PRESS:
		for (auto listener = listeners.begin(); listener != listeners.end(); ++listener)
		{
			(*listener)->onKeyPress(keyCode);
		}
		break;
	case GLFW_RELEASE:
		for (auto listener = listeners.begin(); listener != listeners.end(); ++listener)
		{
			(*listener)->onKeyRelease(keyCode);
		}
		break;
	}
}

void
KeyboardCallback::addListener(KeyboardListener& listener)
{
	listeners.push_back(&listener);
}

void
KeyboardCallback::removeListener(KeyboardListener& listener)
{
	listeners.erase(std::remove(listeners.begin(), listeners.end(), &listener), listeners.end());
}

void
KeyboardCallback::clearListeners()
{
	listeners.clear();
}

#include <vector>
#include <algorithm>

#include "MouseClickCallback.cuh"
#include "MouseClickListener.cuh"

MouseClickCallback::MouseClickCallback()
{

}

void
MouseClickCallback::invoke(int button, int action, int mods, float posX, float posY)
{
	switch (action)
	{
	case GLFW_PRESS:
		for (auto listener = listeners.begin(); listener != listeners.end(); ++listener)
		{
			(*listener)->onMousePress((MouseButton)button, posX, posY);
		}
		break;
	case GLFW_RELEASE:
		for (auto listener = listeners.begin(); listener != listeners.end(); ++listener)
		{
			(*listener)->onMouseRelease((MouseButton)button, posX, posY);
		}
		break;
	}
}

void
MouseClickCallback::addListener(MouseClickListener& listener)
{
	listeners.push_back(&listener);
}

void
MouseClickCallback::removeListener(MouseClickListener& listener)
{
	listeners.erase(std::remove(listeners.begin(), listeners.end(), &listener), listeners.end());
}

void
MouseClickCallback::clearListeners()
{
	listeners.clear();
}
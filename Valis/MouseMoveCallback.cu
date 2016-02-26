
#include <vector>
#include <algorithm>

#include "MouseMoveCallback.cuh"
#include "MouseMoveListener.cuh"

MouseMoveCallback::MouseMoveCallback()
{

}

void
MouseMoveCallback::invoke(float posX, float posY)
{
	for (auto listener = listeners.begin(); listener != listeners.end(); ++listener)
	{
		(*listener)->onMouseMove(posX, posY);
	}
}

void
MouseMoveCallback::addListener(MouseMoveListener& listener)
{
	listeners.push_back(&listener);
}

void
MouseMoveCallback::removeListener(MouseMoveListener& listener)
{
	listeners.erase(std::remove(listeners.begin(), listeners.end(), &listener), listeners.end());
}

void
MouseMoveCallback::clearListeners()
{
	listeners.clear();
}
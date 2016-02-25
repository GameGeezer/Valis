#ifndef VALIS_MOUSE_CALLBACK_CUH
#define VALIS_MOUSE_CALLBACK_CUH

#include <vector>

class MouseClickListener;
class GLFWwindow;

using namespace std;

class MouseClickCallback
{
public:

	MouseClickCallback();

	void
	invoke(int button, int action, int mods, float posX, float posY);

	void
	addListener(MouseClickListener& listener);

	void
	removeListener(MouseClickListener& listener);

	void
	clearListeners();

private:
	vector<MouseClickListener*> listeners;
};


#endif //VALIS_KEYBOARDCALLBACK_H
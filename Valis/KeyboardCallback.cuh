#ifndef VALIS_KEYBOARDCALLBACK_H
#define VALIS_KEYBOARDCALLBACK_H

#include <vector>

class KeyboardListener;
class GLFWwindow;

using namespace std;

class KeyboardCallback
{
public:

	KeyboardCallback();

	void
	invoke(int keyCode, int scanCode, int action, int mods);

	void
	addListener(KeyboardListener& listener);

	void
	removeListener(KeyboardListener& listener);

	void
	clearListeners();

private:
	vector<KeyboardListener*> listeners;
};


#endif //VALIS_KEYBOARDCALLBACK_H
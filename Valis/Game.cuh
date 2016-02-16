#ifndef VALIS_GAME_H
#define VALIS_GAME_H

class Screen;

class Game
{
	friend class Application;

public:
	
	Game(Screen& initialScreen);

	void
	setScreen(Screen& screen);

protected:

	void
	onCreate();

	void
	update(int delta);

private:

	bool isCreated;

	Screen* currentScreen;
};

#endif
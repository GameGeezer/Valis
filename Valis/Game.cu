#include "Game.cuh"

#include "Screen.cuh"

Game::Game(Screen& initialScreen)
{
	currentScreen = &initialScreen;
}

void
Game::onCreate()
{
	currentScreen->onCreate();

	currentScreen->isCreated = true;

	currentScreen->onResume();

	isCreated = true;
}

void
Game::update(int delta)
{
	currentScreen->onUpdate(delta);
}

void
Game::setScreen(Screen& screen)
{
	// Create the new screen if it hasn't been
	if (!screen.isCreated && isCreated)
	{
		screen.onCreate();

		screen.isCreated = true;
	}

	// Resume the new screen
	if (isCreated)
	{
		currentScreen->onLeave();

		screen.onResume();
	}

	currentScreen = &screen;
}
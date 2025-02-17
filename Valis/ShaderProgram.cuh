#ifndef VALIS_SHADERPROGRAM_H
#define VALIS_SHADERPROGRAM_H

#include <GLFW/glfw3.h>
#include <map>
#include <string>
#include <glm/mat4x4.hpp>

using namespace std;

class ShaderProgram
{
public:

	ShaderProgram(const char *vertexShader, const char *fragmentShader, map<int, char *>& attributes);

	ShaderProgram(const char *vertexShader, const char *fragmentShader, const char *tessControlShader, const char *tessEvalShader, map<int, char *>& attributes);

	~ShaderProgram();

	void
	bind();

	void
	unbind();

	void
	destroy();

	GLuint
	getUniformLocation(const char *uniformName);

	GLuint
	getAttributeLocation(const char *uniformName);

	void
	setUnifromMatrix4f(int location, glm::mat4& matrix);

	void
	setUniformf(int location, float value);

	void
	setUniform1i(int location, int value);

private:

	GLuint handle;

	static GLuint
	compileShader(GLuint handle,  const char *shader, GLuint shaderType);
};

#endif //VALIS_SHADERPROGRAM_H
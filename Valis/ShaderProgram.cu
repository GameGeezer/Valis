#include <GL/glew.h>

#include <glm/gtc/type_ptr.hpp>

#include "ShaderProgram.cuh"

ShaderProgram::ShaderProgram(const char *vertexShader, const char *fragmentShader, map<int, char *>& attributes)
{
	handle = glCreateProgram();

	GLuint vertexHandle = compileShader(handle, vertexShader, GL_VERTEX_SHADER);
	GLuint fragmentHandle = compileShader(handle, fragmentShader, GL_FRAGMENT_SHADER);

	glAttachShader(handle, vertexHandle);
	glAttachShader(handle, fragmentHandle);

	for (auto iter = attributes.begin(); iter != attributes.end(); ++iter)
	{
		glBindAttribLocation(handle, iter->first, iter->second);
	}

	glLinkProgram(handle);

	GLint params = -1;
	glGetProgramiv(handle, GL_LINK_STATUS, &params);
	if (params == 0)
	{

	}

	glDetachShader(handle, vertexHandle);
	glDetachShader(handle, fragmentHandle);
	glDeleteShader(vertexHandle);
	glDeleteShader(fragmentHandle);
}

ShaderProgram::ShaderProgram(const char *vertexShader, const char *fragmentShader, const char *tessControlShader, const char *tessEvalShader, map<int, char *>& attributes)
{
	handle = glCreateProgram();

	GLuint vertexHandle = compileShader(handle, vertexShader, GL_VERTEX_SHADER);
	GLuint fragmentHandle = compileShader(handle, fragmentShader, GL_FRAGMENT_SHADER);
	GLuint tessControlHandle = compileShader(handle, tessControlShader, GL_TESS_CONTROL_SHADER);
	GLuint tessEvalHandle = compileShader(handle, tessEvalShader, GL_TESS_EVALUATION_SHADER);

	glAttachShader(handle, vertexHandle);
	glAttachShader(handle, fragmentHandle);
	glAttachShader(handle, tessControlHandle);
	glAttachShader(handle, tessEvalHandle);

	for (auto iter = attributes.begin(); iter != attributes.end(); ++iter)
	{
		glBindAttribLocation(handle, iter->first, iter->second);
	}

	glLinkProgram(handle);

	GLint params = -1;
	glGetProgramiv(handle, GL_LINK_STATUS, &params);
	if (params == GL_FALSE)
	{
		glDeleteProgram(handle);
		handle = -1;
	}

	glDetachShader(handle, vertexHandle);
	glDetachShader(handle, fragmentHandle);
	glDetachShader(handle, tessControlHandle);
	glDetachShader(handle, tessEvalHandle);
	glDeleteShader(vertexHandle);
	glDeleteShader(fragmentHandle);
	glDeleteShader(tessControlHandle);
	glDeleteShader(tessEvalHandle);
}

ShaderProgram::~ShaderProgram()
{
	glDeleteProgram(handle);
}

void
ShaderProgram::bind()
{
	glUseProgram(handle);
}

void
ShaderProgram::unbind()
{
	glUseProgram(0);
}

GLuint
ShaderProgram::getUniformLocation(const char *uniformName)
{
	return glGetUniformLocation(handle, uniformName);
}

GLuint
ShaderProgram::getAttributeLocation(const char *uniformName)
{
	return glGetAttribLocation(handle, uniformName);
}

void
ShaderProgram::setUnifromMatrix4f(int location, glm::mat4& matrix)
{
	glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
}

void
ShaderProgram::setUniformf(int location, float value)
{
	glUniform1f(location, value);
}

void
ShaderProgram::setUniform1i(int location, int value)
{
	glUniform1i(location, value);
}

GLuint
ShaderProgram::compileShader(GLuint handle, const char *shader, GLuint shaderType)
{
	GLuint shaderHandle = glCreateShader(shaderType);

	glShaderSource(shaderHandle, 1, &shader, NULL);

	glCompileShader(handle);
	// Handle compilation errors
	GLint compileStatus;
	glGetShaderiv(handle, GL_COMPILE_STATUS, &compileStatus);
	if (compileStatus == GL_FALSE)
	{
		switch (shaderType)
		{
		case GL_VERTEX_SHADER:
			break;
		case GL_FRAGMENT_SHADER:
			break;
		case GL_TESS_CONTROL_SHADER:
			break;
		case GL_TESS_EVALUATION_SHADER:
			break;
		}
	}

	return shaderHandle;
}
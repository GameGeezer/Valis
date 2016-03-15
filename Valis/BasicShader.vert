uniform mat4 projectionMatrix;

attribute vec4 in_Position;

void main()
{
	//projectionMatrix * gl_ModelViewMatrix *
	vec4 t =  projectionMatrix * in_Position;
	gl_Position = t;
	gl_PointSize  = 0.61803398f;
}
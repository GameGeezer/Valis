uniform mat4 projectionMatrix;

attribute vec4 in_Position;

void main()
{
	//projectionMatrix * gl_ModelViewMatrix *
	vec4 t =  projectionMatrix * in_Position;
	gl_Position = t * vec4(10, 10, 10, 10);
	gl_PointSize  = 1.61803398f;
}
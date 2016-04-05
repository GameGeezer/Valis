#version 450

//in vec3 pass_Normal;

void main()
{

	float lambertCoef = max(dot(vec3(1,0,0), vec3(0.5f, 0.0f, 0.4f)), 0.0);
        
    vec3 diffuse      = vec3(0.7, 0.7, 0.7);
    vec3 ambientColor = vec3(0.6, 0.6, 0.6);
	
    vec3 lightWeighting = ambientColor + diffuse * lambertCoef;

    vec3 color = vec3(30.0f / 255.0f, 197.0f / 255.0f, 3.0f /255.0f) * lightWeighting;
	
    gl_FragColor = vec4(1.0, 0, 0, 1.0);
}
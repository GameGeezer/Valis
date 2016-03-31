#version 450

varying vec3 pass_Normal;

void main()
{

	float lambertCoef = max(dot(pass_Normal, vec3(0.5f, 0.0f, 0.4f)), 0.0);
        
    vec3 diffuse      = vec3(0.7, 0.7, 0.7);
    vec3 ambientColor = vec3(0.6, 0.6, 0.6);
	
    vec3 lightWeighting = ambientColor + diffuse * lambertCoef;

    vec3 color = vec3(1,0.5f,0.25f) * lightWeighting;
	
    gl_FragColor = vec4(color, 1.0);
}
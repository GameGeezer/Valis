#version 450

in vec3 pass_finalizedNormal;

void main()
{

	float lambertCoef = max(dot(pass_finalizedNormal, vec3(-1.0f, 0.0f, 0.0f)), 0.0);
        
    vec3 diffuse      = vec3(1, 1, 1);
    vec3 ambientColor = vec3(0.3, 0.3, 0.3);
	
    vec3 lightWeighting = ambientColor + diffuse * lambertCoef;

    vec3 color = vec3(30.0f / 255.0f, 197.0f / 255.0f, 3.0f /255.0f) * lightWeighting;
	
    gl_FragColor = vec4(color, 1);
}
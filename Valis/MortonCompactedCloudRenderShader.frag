#version 450

in vec3 pass_finalizedNormal;
flat in uint pass_finalizedSpare;

void main()
{
	float firstBrush = float(pass_finalizedSpare == 2);
	float secondBrush = float(pass_finalizedSpare == 3);
	float thirdBrush = uint(pass_finalizedSpare == 4);

	vec3 colorChoice = vec3(255.0f / 255.0f, 255.0f / 255.0f,51.0f /255.0f) * firstBrush + vec3(30.0f / 255.0f, 150.0f / 255.0f, 250.0f /255.0f) * secondBrush + vec3(255.0f / 255.0f, 50.0f / 255.0f, 51.0f /255.0f) * thirdBrush;
	float lambertCoef = max(dot(pass_finalizedNormal, vec3(-1.0f, 0.0f, 0.0f)), 0.0);
        
    vec3 diffuse      = vec3(1, 1, 1);
    vec3 ambientColor = vec3(0.3, 0.3, 0.3);
	
    vec3 lightWeighting = ambientColor + diffuse * lambertCoef;

    //vec3 color = colorChoice * lightWeighting;
	
    gl_FragColor = vec4(colorChoice, 1);
}
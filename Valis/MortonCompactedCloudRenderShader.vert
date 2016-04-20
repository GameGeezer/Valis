#version 450

uniform usampler1D offsetTexture;

uniform mat4 projectionMatrix;

uniform float gridResolution;

uniform float offsetBufferSize;

attribute uint in_CompactData;

out vec3 pass_Normal;
out vec3 pass_Position;
out uint pass_Spare;

uint compactBy3(in uint value)
{
	value &= 0x09249249;
	value |= (value >> 2);
	value &= 0x030c30c3;
	value |= (value >> 4);
	value &= 0x0300f00f;
	value |= (value >> 8);
	value &= 0x030000ff;
	value |= (value >> 16);
	value &= 0x000003ff;

	return value;
}

float findNormalDirection(in uint value)
{
	float normal = 0.0f;
	//normal = 1.0f - 0.25f * float(value);
	
	normal += float(value == 0) * 1.0f;
	normal += float(value == 1) * 0.75f;
	normal += float(value == 2) * 0.5f;
	normal += float(value == 3) * 0.25f;
	normal += float(value == 4) * 0.0f;
	normal += float(value == 5) * -0.25f;
	normal += float(value == 6) * -0.5f;
	normal += float(value == 7) * -0.75f;
	normal += float(value == 8) * -1.0f;
	
	return normal;
}

void main()
{
	float locationIndexX = (float(gl_VertexID) / 64.0f);
	uint compactOffsetLocation = texelFetch(offsetTexture, int(locationIndexX), 0).r;

	//uint compactOffsetLocation = offsetLocation.x | (offsetLocation.y << 8) | (offsetLocation.z << 16) | (offsetLocation.w << 24);

	uint maskedPosition = in_CompactData & 0x3FFFF;

	uint location = compactOffsetLocation + maskedPosition;

	uint raw_x = compactBy3(location);
	uint raw_y = compactBy3(location >> 1);
	uint raw_z = compactBy3(location >> 2);

	uint raw_nx = in_CompactData & 0x1C0000 >> 18;
	uint raw_ny = (in_CompactData & 0xE00000) >> 21;
	uint raw_nz = (in_CompactData & 0x7000000) >> 24;
	uint raw_spare = (in_CompactData & 0xF8000000) >> 27;

	float x = ((float(raw_x)) / gridResolution); //add the offset pased via texture
	float y = ((float(raw_y)) / gridResolution);
	float z = ((float(raw_z)) / gridResolution);

	float nx = findNormalDirection(raw_nx);
	float ny = findNormalDirection(raw_ny);
	float nz = findNormalDirection(raw_nz);

	//gl_Position = projectionMatrix * vec4(x, y, z, 1);
	pass_Position = vec3(x, y, z);
	pass_Normal = normalize(vec3(nx, ny, nz));
	pass_Spare = raw_spare;
	gl_PointSize  = 0.61803398f;
}
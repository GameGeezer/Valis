#version 450

uniform usampler2D offsetTexture;

uniform mat4 projectionMatrix;

uniform float gridResolution;

attribute uint in_CompactData;
attribute int gl_VertexID;

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

void main()
{
	float locationIndexX = (float(gl_VertexID) / 64.0f) / gridResolution;
	uvec4 offsetLocation = texture2D(offsetTexture, vec2(locationIndexX, 0));

	uint compactOffsetLocation = offsetLocation.x | (offsetLocation.y << 8) | (offsetLocation.z << 16) | (offsetLocation.w << 24);

	uint offsetX = compactOffsetLocation & 0x3FF;
	uint offsetY = (compactOffsetLocation & 0xFFC00) >> 10;
	uint offsetZ = (compactOffsetLocation & 0x3FF00000) >> 20;

	uint maskedPosition = in_CompactData & 0x3FFFF;

	uint raw_x  = compactBy3(maskedPosition);
	uint raw_y  = compactBy3(maskedPosition >> 1);
	uint raw_z  = compactBy3(maskedPosition >> 2);

	uint raw_nx = in_CompactData & 0x1C0000 >> 18;
	uint raw_ny = (in_CompactData & 0xE00000) >> 21;
	uint raw_nz = (in_CompactData & 0x7000000) >> 24;

	float x = ((float(raw_x) + float(offsetX)) / gridResolution); //add the offset pased via texture
	float y = ((float(raw_y) + float(offsetY)) / gridResolution);
	float z = ((float(raw_z) + float(offsetZ)) / gridResolution);

	float nx = float(raw_nx);
	float ny = float(raw_ny);
	float nz = float(raw_nz);

	gl_Position = projectionMatrix * vec4(x, y, z, 1);
	gl_PointSize  = 0.61803398f;
}
#version 450

uniform usampler1D offsetTexture;

uniform mat4 projectionMatrix;

uniform float gridResolution;

uniform float offsetBufferSize;

attribute uint in_CompactData;

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
	float locationIndexX = (float(gl_VertexID) / 64.0f);
	uint compactOffsetLocation = texelFetch(offsetTexture, int(locationIndexX), 0).r;

	//uint compactOffsetLocation = offsetLocation.x | (offsetLocation.y << 8) | (offsetLocation.z << 16) | (offsetLocation.w << 24);

	uint maskedPosition = in_CompactData & 0x3FFFF;

	uint location = compactOffsetLocation + maskedPosition;

	uint raw_x = compactBy3(location); //in_CompactData & 0x3F;
	uint raw_y = compactBy3(location >> 1); //(in_CompactData & 0xFC0) >> 6;
	uint raw_z = compactBy3(location >> 2); //(in_CompactData & 0x3F000) >> 12;

	uint raw_nx = in_CompactData & 0x1C0000 >> 18;
	uint raw_ny = (in_CompactData & 0xE00000) >> 21;
	uint raw_nz = (in_CompactData & 0x7000000) >> 24;

	float x = ((float(raw_x)) / gridResolution); //add the offset pased via texture
	float y = ((float(raw_y)) / gridResolution);
	float z = ((float(raw_z)) / gridResolution);

	float nx = float(raw_nx);
	float ny = float(raw_ny);
	float nz = float(raw_nz);

	gl_Position = projectionMatrix * vec4(x, y, z, 1);
	gl_PointSize  = 0.61803398f;
}
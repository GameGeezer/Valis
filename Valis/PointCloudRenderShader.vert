uniform mat4 projectionMatrix;

uniform float gridResolution

attribute uint in_CompactData;

varying pass_Normal;

void main()
{
	uint raw_x = in_CompactData & 0x3F;
	uint raw_y = in_CompactData & 0xFC0;
	uint raw_z = in_CompactData & 0x3F000;

	uint raw_nx = in_CompactData & 0x1C0000;
	uint raw_ny = in_CompactData & 0xE00000;
	uint raw_nz = in_CompactData & 0x7000000;

	float x = (uintBitsToFloat(x) / gridResolution); //add the offset pased via texture
	float y = (uintBitsToFloat(y) / gridResolution);
	float z = (uintBitsToFloat(z) / gridResolution);

	float nx = uintBitsToFloat(raw_nx);
	float ny = uintBitsToFloat(raw_ny);
	float nz = uintBitsToFloat(raw_nz);

	pass_Normal = projectionMatrix * vec4(nx, ny, nz, 1);

	gl_Position = projectionMatrix * vec4(x, y, z, 1);

	gl_PointSize  = 0.61803398f;
}
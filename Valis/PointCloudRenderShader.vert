uniform mat4 projectionMatrix;

uniform float gridResolution;

attribute uint in_CompactData;
attribute int gl_VertexID;

void main()
{
	uint raw_x = in_CompactData & 0x3F;
	uint raw_y = (in_CompactData & 0xFC0) >> 6;
	uint raw_z = (in_CompactData & 0x3F000) >> 12;

	uint raw_nx = in_CompactData & 0x1C0000 >> 18;
	uint raw_ny = in_CompactData & 0xE00000 >> 21;
	uint raw_nz = in_CompactData & 0x7000000 >> 24;

	float x = (float(raw_x) / gridResolution); //add the offset pased via texture
	float y = (float(raw_y) / gridResolution);
	float z = (float(raw_z) / gridResolution);

	float nx = float(raw_nx);
	float ny = float(raw_ny);
	float nz = float(raw_nz);

	gl_Position = projectionMatrix * vec4(x, y, z, 1);
	gl_PointSize  = 0.61803398f;
}
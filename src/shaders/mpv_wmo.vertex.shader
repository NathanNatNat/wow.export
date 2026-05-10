#version 300 es
precision highp float;
precision highp int;

// vertex attributes
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 4) in vec2 a_texcoord;
layout(location = 5) in vec2 a_texcoord2;
layout(location = 6) in vec4 a_color;
layout(location = 7) in vec4 a_color2;
layout(location = 8) in vec2 a_texcoord3;
layout(location = 9) in vec2 a_texcoord4;
layout(location = 10) in vec4 a_color3;

// uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform int u_vertex_shader;

// outputs
out vec2 v_texcoord;
out vec2 v_texcoord2;
out vec2 v_texcoord3;
out vec2 v_texcoord4;
out vec3 v_normal;
out vec3 v_position;
out vec4 v_color;
out vec4 v_color2;
out vec4 v_color3;

void main() {
	vec4 world_pos = u_model * vec4(a_position, 1.0);
	gl_Position = u_projection * u_view * world_pos;

	v_normal = normalize(mat3(u_model) * a_normal);
	v_position = world_pos.xyz;
	v_color = a_color;
	v_color2 = a_color2;
	v_color3 = a_color3;
	v_texcoord4 = a_texcoord4;

	// un-swizzle from viewer [X, Z, -Y] back to WoW [X, Y, Z] for UV computations
	vec3 wow_normal = vec3(a_normal.x, -a_normal.z, a_normal.y);

	switch (u_vertex_shader) {
		case 0: // MapObjDiffuse_T1
			v_texcoord = a_texcoord;
			v_texcoord2 = a_texcoord2;
			v_texcoord3 = a_texcoord3;
			break;
		case 1: // MapObjDiffuse_T1_Refl
			v_texcoord = a_texcoord;
			v_texcoord2 = reflect(normalize(vec3(1.0)), wow_normal).xy;
			v_texcoord3 = a_texcoord3;
			break;
		case 2: // MapObjDiffuse_T1_Env_T2
			v_texcoord = a_texcoord;
			v_texcoord2 = a_texcoord2;
			v_texcoord3 = a_texcoord3;
			break;
		case 3: // MapObjSpecular_T1
			v_texcoord = a_texcoord;
			v_texcoord2 = a_texcoord2;
			v_texcoord3 = a_texcoord3;
			break;
		case 4: // MapObjDiffuse_Comp
			v_texcoord = a_texcoord;
			v_texcoord2 = a_texcoord2;
			v_texcoord3 = a_texcoord3;
			break;
		case 5: // MapObjDiffuse_Comp_Refl
			v_texcoord = a_texcoord;
			v_texcoord2 = a_texcoord2;
			v_texcoord3 = reflect(normalize(vec3(1.0)), wow_normal).xy;
			break;
		case 6: // MapObjDiffuse_Comp_Terrain
			v_texcoord = a_texcoord;
			v_texcoord2 = vec2(a_position.x, -a_position.z) * -0.239999995;
			v_texcoord3 = a_texcoord3;
			break;
		case 7: // MapObjDiffuse_CompAlpha
			v_texcoord = a_texcoord;
			v_texcoord2 = vec2(a_position.x, -a_position.z) * -0.239999995;
			v_texcoord3 = a_texcoord3;
			break;
		case 8: // MapObjParallax
			v_texcoord = a_texcoord;
			v_texcoord2 = a_texcoord2;
			v_texcoord3 = a_texcoord3;
			break;
		default:
			v_texcoord = a_texcoord;
			v_texcoord2 = a_texcoord2;
			v_texcoord3 = a_texcoord3;
			break;
	}
}

#version 300 es
precision highp float;
precision highp sampler2DArray;

in vec3 v_normal;
in vec3 v_position;
in vec2 v_texcoord;
in vec4 v_color;

out vec4 frag_color;

uniform sampler2D u_tex0;
uniform sampler2D u_tex1;
uniform sampler2D u_tex2;
uniform sampler2D u_tex3;
uniform sampler2D u_tex4;
uniform sampler2D u_tex5;
uniform sampler2D u_tex6;
uniform sampler2D u_tex7;
uniform sampler2DArray u_alpha_maps;

uniform int u_layer_count;
uniform int u_alpha_offset;
uniform vec2 u_chunk_offset;

uniform int u_diffuse_slot[8];
uniform float u_layer_scale[8];

uniform vec3 u_camera_pos;

#include "mpv_light.inc.glsl"
#include "mpv_fog.inc.glsl"

vec4 sample_slot(int slot, vec2 uv) {
	switch (slot) {
		case 0: return texture(u_tex0, uv);
		case 1: return texture(u_tex1, uv);
		case 2: return texture(u_tex2, uv);
		case 3: return texture(u_tex3, uv);
		case 4: return texture(u_tex4, uv);
		case 5: return texture(u_tex5, uv);
		case 6: return texture(u_tex6, uv);
		case 7: return texture(u_tex7, uv);
	}
	return vec4(0.0);
}

void main() {
	vec2 chunk_uv = v_texcoord * 16.0 - u_chunk_offset;

	float alphas[8];
	alphas[0] = 1.0;
	for (int i = 1; i < 8; i++) {
		if (i < u_layer_count)
			alphas[i] = texture(u_alpha_maps, vec3(chunk_uv, float(u_alpha_offset + i - 1))).r;
		else
			alphas[i] = 0.0;
	}

	// simple alpha blending (no height weighting)
	vec2 tc0 = chunk_uv * (8.0 / u_layer_scale[0]);
	vec3 terrain_color = sample_slot(u_diffuse_slot[0], tc0).rgb;

	for (int i = 1; i < 8; i++) {
		if (i >= u_layer_count)
			break;

		vec2 tc = chunk_uv * (8.0 / u_layer_scale[i]);
		vec3 layer_color = sample_slot(u_diffuse_slot[i], tc).rgb;
		terrain_color = mix(terrain_color, layer_color, alphas[i]);
	}

	terrain_color *= v_color.rgb * 2.0;

	vec3 n = normalize(v_normal);
	vec3 color = calc_exterior_light(terrain_color, n);
	color = apply_fog(color, v_position, u_camera_pos);
	frag_color = vec4(color, 1.0);
}

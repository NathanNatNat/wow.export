#version 300 es
precision highp float;

in vec3 v_position;
in vec2 v_texcoord;
in float v_depth;

uniform sampler2D u_texture;
uniform int u_material_id;
uniform int u_liquid_flags;
uniform vec3 u_liquid_color;
uniform float u_float0;
uniform float u_float1;
uniform float u_time;

uniform vec3 u_camera_pos;

// scene-driven liquid colors from LightData/LightParams
uniform vec3 u_close_river_color;
uniform vec3 u_close_ocean_color;
uniform float u_river_shallow_alpha;
uniform float u_river_deep_alpha;
uniform float u_ocean_shallow_alpha;
uniform float u_ocean_deep_alpha;

out vec4 frag_color;

#include "mpv_light.inc.glsl"
#include "mpv_fog.inc.glsl"

const float PI = 3.14159265359;
const int LIQUID_FLAG_OCEAN = 1024;

vec2 rotate_z(vec2 v, float angle) {
	float s = sin(angle);
	float c = cos(angle);
	return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

vec2 get_scroll_offset(float time, vec2 speed) {
	vec2 result = vec2(0.0);
	if (abs(speed.x) > 0.001)
		result.x = mod(time, 1000.0 / speed.x) / (1000.0 / speed.x);
	if (abs(speed.y) > 0.001)
		result.y = mod(time, 1000.0 / speed.y) / (1000.0 / speed.y);
	return result;
}

void main() {
	// select base color and alpha from scene light data based on liquid flags
	vec3 base_color;
	float shallow_alpha;
	float deep_alpha;

	if ((u_liquid_flags & LIQUID_FLAG_OCEAN) != 0) {
		base_color = u_close_ocean_color;
		shallow_alpha = u_ocean_shallow_alpha;
		deep_alpha = u_ocean_deep_alpha;
	} else if (u_liquid_flags == 15) {
		base_color = u_close_river_color;
		shallow_alpha = u_river_shallow_alpha;
		deep_alpha = u_river_deep_alpha;
	} else {
		base_color = u_liquid_color;
		shallow_alpha = 0.7;
		deep_alpha = 0.7;
	}

	vec2 uv = v_texcoord;
	if (u_material_id == 1 || u_material_id == 3) {
		uv *= u_float0;
		uv = rotate_z(uv, u_float1 * PI / 180.0);
	} else if (u_material_id == 2 || u_material_id == 4) {
		uv += get_scroll_offset(u_time, vec2(u_float0, u_float1));
	}

	vec3 tex_color = texture(u_texture, uv).rgb;
	vec3 diffuse = base_color + tex_color;

	vec3 n = vec3(0.0, 1.0, 0.0);

	// magma is not affected by light
	if (u_material_id != 2 && u_material_id != 4)
		diffuse = calc_exterior_light(diffuse, n);

	diffuse = apply_fog(diffuse, v_position, u_camera_pos);

	float alpha = mix(shallow_alpha, deep_alpha, v_depth);
	frag_color = vec4(diffuse, alpha);
}

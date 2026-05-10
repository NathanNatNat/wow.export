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

uniform vec3 u_light_dir;
uniform vec3 u_sun_color;
uniform float u_sun_intensity;
uniform vec3 u_camera_pos;

out vec4 frag_color;

#include "mpv_fog.inc.glsl"

const float PI = 3.14159265359;
const vec3 SKY_COLOR = vec3(0.4, 0.5, 0.7);
const vec3 GROUND_COLOR = vec3(0.25, 0.2, 0.15);
const float AMBIENT = 0.25;
const float LIQUID_ALPHA_SHALLOW = 0.4;
const float LIQUID_ALPHA_DEEP = 0.8;

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
	vec3 base_color = u_liquid_color;

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

	if (u_material_id != 2 && u_material_id != 4) {
		float n_dot_l = dot(n, u_light_dir);
		float sky_factor = 0.5 + 0.5 * n.y;
		vec3 ambient = mix(GROUND_COLOR, SKY_COLOR, sky_factor) * AMBIENT;
		float diff = max(n_dot_l, 0.0);
		diffuse = diffuse * (ambient + u_sun_color * diff * u_sun_intensity);
	}

	diffuse = apply_fog(diffuse, v_position, u_camera_pos);

	float alpha = mix(LIQUID_ALPHA_SHALLOW, LIQUID_ALPHA_DEEP, v_depth);
	frag_color = vec4(diffuse, alpha);
}

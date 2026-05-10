#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_position;

uniform vec3 u_light_dir;
uniform vec3 u_sun_color;
uniform float u_sun_intensity;
uniform vec3 u_camera_pos;

out vec4 frag_color;

#include "mpv_fog.inc.glsl"

const vec3 SKY_AMBIENT = vec3(0.6, 0.65, 0.7);
const vec3 GROUND_AMBIENT = vec3(0.35, 0.3, 0.25);
const vec3 BASE_COLOR = vec3(0.6, 0.6, 0.6);

void main() {
	vec3 n = normalize(v_normal);
	float n_dot_l = dot(n, u_light_dir);
	float hemi = n_dot_l * 0.5 + 0.5;
	vec3 ambient = mix(GROUND_AMBIENT, SKY_AMBIENT, hemi);
	float direct = max(n_dot_l, 0.0);
	float intensity = u_sun_intensity / 100.0;

	vec3 lit = BASE_COLOR * (ambient + u_sun_color * direct * intensity);
	lit = apply_fog(lit, v_position, u_camera_pos);

	frag_color = vec4(lit, 1.0);
}

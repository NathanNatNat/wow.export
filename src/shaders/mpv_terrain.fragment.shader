#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_position;

uniform vec3 u_terrain_color;
uniform vec3 u_camera_pos;

out vec4 frag_color;

#include "mpv_light.inc.glsl"
#include "mpv_fog.inc.glsl"

void main() {
	vec3 n = normalize(v_normal);
	vec3 color = calc_exterior_light(u_terrain_color, n);
	color = apply_fog(color, v_position, u_camera_pos);
	frag_color = vec4(color, 1.0);
}

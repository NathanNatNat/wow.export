#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;
out vec3 v_position;

void main() {
	vec4 world_pos = u_model * vec4(a_position, 1.0);
	gl_Position = u_projection * u_view * world_pos;

	v_normal = normalize(mat3(u_model) * a_normal);
	v_position = world_pos.xyz;
}

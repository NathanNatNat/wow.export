#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texcoord;
layout(location = 3) in vec4 a_color;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;
out vec3 v_position;
out vec2 v_texcoord;
out vec4 v_color;

void main() {
	gl_Position = u_projection * u_view * vec4(a_position, 1.0);
	v_normal = a_normal;
	v_position = a_position;
	v_texcoord = a_texcoord;
	v_color = a_color;
}

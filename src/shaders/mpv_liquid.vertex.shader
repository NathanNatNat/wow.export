#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texcoord;
layout(location = 2) in float a_depth;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_position;
out vec2 v_texcoord;
out float v_depth;

void main() {
	gl_Position = u_projection * u_view * vec4(a_position, 1.0);
	v_position = a_position;
	v_texcoord = a_texcoord;
	v_depth = a_depth;
}

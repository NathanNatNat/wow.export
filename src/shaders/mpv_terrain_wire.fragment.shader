#version 300 es
precision highp float;

uniform vec3 u_terrain_color;

out vec4 frag_color;

void main() {
	frag_color = vec4(u_terrain_color, 1.0);
}

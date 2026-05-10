#version 300 es
precision highp float;

layout(location = 0) in vec4 a_position;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_sky_colors[6];

out vec3 v_color;

void main() {
	// band index stored in w component
	int band = int(a_position.w);
	v_color = u_sky_colors[band];

	// scale to world size, camera-relative (view matrix strips translation)
	vec3 pos = a_position.xyz * 33.333;

	// strip translation from view matrix (sky follows camera)
	mat4 view_no_translate = u_view;
	view_no_translate[3] = vec4(0.0, 0.0, 0.0, 1.0);

	gl_Position = u_projection * view_no_translate * vec4(pos, 1.0);
}

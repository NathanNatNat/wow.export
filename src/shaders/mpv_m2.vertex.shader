#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 4) in vec2 a_texcoord;
layout(location = 5) in vec2 a_texcoord2;

// per-instance model matrix (4 columns)
layout(location = 6) in vec4 a_model_col0;
layout(location = 7) in vec4 a_model_col1;
layout(location = 8) in vec4 a_model_col2;
layout(location = 9) in vec4 a_model_col3;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;
out vec3 v_position;

void main() {
	mat4 model = mat4(a_model_col0, a_model_col1, a_model_col2, a_model_col3);
	vec4 world_pos = model * vec4(a_position, 1.0);
	gl_Position = u_projection * u_view * world_pos;

	// uniform scale: upper-3x3 preserves normal direction
	v_normal = normalize(mat3(model) * a_normal);
	v_position = world_pos.xyz;
}

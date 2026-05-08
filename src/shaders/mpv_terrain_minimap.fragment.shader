#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_position;
in vec2 v_texcoord;

uniform sampler2D u_minimap;

out vec4 frag_color;

const vec3 LIGHT_DIR = vec3(0.5051, 0.8081, 0.3031);
const vec3 SKY_COLOR = vec3(0.4, 0.5, 0.7);
const vec3 GROUND_COLOR = vec3(0.25, 0.2, 0.15);
const vec3 SUN_COLOR = vec3(1.0, 0.95, 0.85);
const float AMBIENT = 0.25;

void main() {
	vec3 n = normalize(v_normal);
	float n_dot_l = dot(n, LIGHT_DIR);

	float sky_factor = 0.5 + 0.5 * n.y;
	vec3 ambient = mix(GROUND_COLOR, SKY_COLOR, sky_factor) * AMBIENT;

	float diffuse = max(n_dot_l, 0.0);
	vec3 tex_color = texture(u_minimap, v_texcoord).rgb;
	vec3 color = tex_color * (ambient + SUN_COLOR * diffuse * 0.75);

	frag_color = vec4(color, 1.0);
}

#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_position;

uniform vec3 u_terrain_color;
uniform vec3 u_light_dir;
uniform vec3 u_sun_color;
uniform float u_sun_intensity;

out vec4 frag_color;

const vec3 SKY_COLOR = vec3(0.4, 0.5, 0.7);
const vec3 GROUND_COLOR = vec3(0.25, 0.2, 0.15);
const float AMBIENT = 0.25;

void main() {
	vec3 n = normalize(v_normal);
	float n_dot_l = dot(n, u_light_dir);

	// hemisphere ambient (sky vs ground)
	float sky_factor = 0.5 + 0.5 * n.y;
	vec3 ambient = mix(GROUND_COLOR, SKY_COLOR, sky_factor) * AMBIENT;

	// diffuse
	float diffuse = max(n_dot_l, 0.0);
	vec3 color = u_terrain_color * (ambient + u_sun_color * diffuse * u_sun_intensity);

	frag_color = vec4(color, 1.0);
}

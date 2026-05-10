#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_position;

uniform vec3 u_light_dir;
uniform vec3 u_sun_color;
uniform float u_sun_intensity;

uniform vec3 u_camera_pos;
uniform vec3 u_fog_color;
uniform float u_fog_start;
uniform float u_fog_end;

out vec4 frag_color;

const vec3 BASE_COLOR = vec3(0.6, 0.6, 0.6);
const vec3 SKY_COLOR = vec3(0.4, 0.5, 0.7);
const vec3 GROUND_COLOR = vec3(0.25, 0.2, 0.15);
const float AMBIENT = 0.25;

void main() {
	vec3 n = normalize(v_normal);
	float n_dot_l = dot(n, u_light_dir);

	float sky_factor = 0.5 + 0.5 * n.y;
	vec3 ambient = mix(GROUND_COLOR, SKY_COLOR, sky_factor) * AMBIENT;

	float diffuse = max(n_dot_l, 0.0);
	vec3 color = BASE_COLOR * (ambient + u_sun_color * diffuse * u_sun_intensity);

	float dist = distance(v_position, u_camera_pos);
	float fog = clamp((dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
	color = mix(color, u_fog_color, fog);

	frag_color = vec4(color, 1.0);
}

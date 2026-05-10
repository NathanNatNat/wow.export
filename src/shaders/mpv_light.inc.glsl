uniform vec3 u_light_dir;
uniform vec3 u_ambient_color;
uniform vec3 u_horizon_ambient_color;
uniform vec3 u_ground_ambient_color;
uniform vec3 u_direct_color;
uniform int u_lighting_enabled;

vec3 calc_exterior_light(vec3 base_color, vec3 normal) {
	if (u_lighting_enabled == 0)
		return base_color;

	float n_dot_up = normal.y;
	float sky_w = max(n_dot_up, 0.0);
	float ground_w = max(-n_dot_up, 0.0);
	float horiz_w = 1.0 - sky_w - ground_w;
	vec3 ambient = u_ambient_color * sky_w + u_horizon_ambient_color * horiz_w + u_ground_ambient_color * ground_w;
	float n_dot_l = max(dot(normal, u_light_dir), 0.0);
	return base_color * (ambient + u_direct_color * n_dot_l);
}

// shadow-aware overload: shadow attenuates direct light only
vec3 calc_exterior_light(vec3 base_color, vec3 normal, vec3 shadow) {
	if (u_lighting_enabled == 0)
		return base_color;

	float n_dot_up = normal.y;
	float sky_w = max(n_dot_up, 0.0);
	float ground_w = max(-n_dot_up, 0.0);
	float horiz_w = 1.0 - sky_w - ground_w;
	vec3 ambient = u_ambient_color * sky_w + u_horizon_ambient_color * horiz_w + u_ground_ambient_color * ground_w;
	float n_dot_l = max(dot(normal, u_light_dir), 0.0);
	return base_color * (ambient + shadow * u_direct_color * n_dot_l);
}

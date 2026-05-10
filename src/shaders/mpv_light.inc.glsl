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

// overload with precomputed light (WMO baked vertex lighting added to ambient)
vec3 calc_exterior_light(vec3 base_color, vec3 normal, vec3 precomputed_light) {
	if (u_lighting_enabled == 0)
		return base_color;

	float n_dot_up = normal.y;
	float n_dot_l = max(dot(normal, u_light_dir), 0.0);

	vec3 adj_ambient = u_ambient_color + precomputed_light;
	vec3 adj_horiz = u_horizon_ambient_color + precomputed_light;
	vec3 adj_ground = u_ground_ambient_color + precomputed_light;

	vec3 curr_color;
	if (n_dot_up >= 0.0)
		curr_color = mix(adj_horiz, adj_ambient, n_dot_up);
	else
		curr_color = mix(adj_horiz, adj_ground, -n_dot_up);

	vec3 sky_color = curr_color * 1.1;
	vec3 ground_color = curr_color * 0.7;
	curr_color = mix(ground_color, sky_color, 0.5 + 0.5 * n_dot_l);

	return base_color * (curr_color + u_direct_color * n_dot_l);
}

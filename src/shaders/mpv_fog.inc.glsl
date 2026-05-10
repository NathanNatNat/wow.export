// wow fog uniforms
uniform float u_fog_enabled;
uniform vec4 u_fog_density_params;       // (start, end, density, 0)
uniform vec4 u_fog_height_plane;         // (0, 1, 0, -fogHeight) -- world space Y-up
uniform vec4 u_fog_color_height_rate;    // (r, g, b, heightRate)
uniform vec4 u_fog_hdensity_end_color;   // (heightDensity, r, g, b)
uniform vec4 u_fog_sun_angle_color;      // (angle, r, g, b)
uniform vec4 u_fog_hcolor_end_dist;      // (r, g, b, endFogColorDist)
uniform vec4 u_fog_sun_pct_str;          // (sunPct, sunStrength, 0, 0)
uniform vec4 u_fog_sun_dir_z_scalar;     // (dirX, dirY, dirZ, fogZScalar)
uniform vec4 u_fog_height_coeff;         // cubic polynomial for height fog curve
uniform vec4 u_fog_main_coeff;           // cubic polynomial for main fog falloff
uniform vec4 u_fog_hdensity_coeff;       // cubic polynomial for height density
uniform vec4 u_fog_distances;            // (mainEndDist, mainStartDist, legacyScalar, blendAlpha)
uniform vec4 u_fog_hend_color_offset;    // (r, g, b, fogStartOffset)

vec3 apply_fog(vec3 color, vec3 world_pos, vec3 camera_pos) {
	if (u_fog_enabled < 0.5)
		return color;

	float fog_start = u_fog_density_params.x;
	float fog_end = u_fog_density_params.y;
	float fog_density = u_fog_density_params.z;
	float fog_height = -u_fog_height_plane.w;
	float height_rate = u_fog_color_height_rate.w;
	float height_density = u_fog_hdensity_end_color.x;
	float fog_z_scalar = u_fog_sun_dir_z_scalar.w;
	float main_fog_end_dist = u_fog_distances.x;
	float main_fog_start_dist = u_fog_distances.y;
	float legacy_fog_scalar = u_fog_distances.z;

	vec3 to_vertex = world_pos - camera_pos;
	float vert_component = to_vertex.y;

	// adjust vertical contribution to distance (fogZScalar)
	vec3 adjusted_pos = to_vertex;
	if (vert_component > 0.0)
		adjusted_pos.y -= clamp(fog_z_scalar, 0.0, vert_component);
	else
		adjusted_pos.y -= clamp(-fog_z_scalar, vert_component, 0.0);

	float v_length = length(adjusted_pos);
	float raw_length = length(to_vertex);
	float z = v_length;
	float exp_max = max(0.0, z - fog_start);

	// height fog factor
	float height = world_pos.y - fog_height;
	float height_fog_raw = clamp(height * height_rate, 0.0, 1.0);
	float height_fog_inv = 1.0 - height_fog_raw;

	// apply height fog polynomial
	float hf = height_fog_inv;
	float hf2 = hf * hf;
	float hf3 = hf2 * hf;
	vec4 hc = u_fog_height_coeff;
	float height_fog_poly = clamp(hc.x * hf3 + hc.y * hf2 + hc.z * hf + hc.w, 0.0, 1.0);
	float height_fog = 1.0 - height_fog_poly;

	// legacy exponential fog
	float legacy_exp_fog = 1.0 / exp(exp_max * fog_density);
	float legacy_exp_height = 1.0 / exp(exp_max * height_density);
	float legacy_fog_mixed = mix(legacy_exp_fog, legacy_exp_height, height_fog);
	float legacy_fog = 1.0 - (1.0 - legacy_fog_mixed);
	float end_fade = clamp(1.42857146 * (1.0 - (v_length / fog_end)), 0.0, 1.0);
	float legacy_result = min(legacy_fog, end_fade);

	// modern art fog (cubic polynomial curves)
	float art_norm_dist = clamp((v_length - main_fog_start_dist) / max(main_fog_end_dist - main_fog_start_dist, 0.001), 0.0, 1.0);
	float engine_norm_dist = clamp(v_length / max(main_fog_end_dist, 0.001), 0.0, 1.0);

	float ad = art_norm_dist;
	float ad2 = ad * ad;
	float ad3 = ad2 * ad;

	vec4 mc = u_fog_main_coeff;
	float main_curve = clamp(mc.x * ad3 + mc.y * ad2 + mc.z * ad + mc.w, 0.0, 1.0);
	float fog_result = clamp(1.0 - main_curve, 0.0, 1.0);

	vec4 hdc = u_fog_hdensity_coeff;
	float hdensity_curve = clamp(hdc.x * ad3 + hdc.y * ad2 + hdc.z * ad + hdc.w, 0.0, 1.0);
	float height_fog_result = clamp(1.0 - hdensity_curve, 0.0, 1.0);

	float art_fog_mixed = mix(fog_result, height_fog_result, height_fog);
	float art_fog = 1.0 - (1.0 - art_fog_mixed);

	// blend end region
	float end_pct = clamp(fog_end / max(main_fog_end_dist, 0.001), 0.0, 1.0);
	float blend_begin = end_pct - 0.3;
	float bb2 = blend_begin * blend_begin;
	float bb3 = bb2 * blend_begin;
	float main_at_blend = clamp(mc.x * bb3 + mc.y * bb2 + mc.z * blend_begin + mc.w, 0.0, 1.0);
	float hdensity_at_blend = clamp(hdc.x * bb3 + hdc.y * bb2 + hdc.z * blend_begin + hdc.w, 0.0, 1.0);
	float fog_at_blend = mix(main_at_blend, hdensity_at_blend, height_fog);

	float blend_value = clamp((engine_norm_dist - blend_begin) / 0.3, 0.0, 1.0);
	float end_fog_value = blend_value * (1.0 - fog_at_blend) + fog_at_blend;
	float end_fog_clamped = clamp(1.0 - end_fog_value, 0.0, 1.0);
	float final_art = min(art_fog, mix(art_fog, end_fog_clamped, blend_value));

	// blend between legacy and modern
	float fog_factor = mix(final_art, legacy_result, legacy_fog_scalar);

	// fog colour computation
	float fog_start_offset = u_fog_hend_color_offset.w;
	float end_color_dist = u_fog_hcolor_end_dist.w;
	float color_dist = clamp((raw_length - fog_start_offset) / max(end_color_dist, 0.001), 0.0, 1.0);
	float color_dist_cube = color_dist * color_dist * color_dist;

	vec3 fog_color = u_fog_color_height_rate.rgb;
	vec3 end_fog_color = u_fog_hdensity_end_color.yzw;
	vec3 height_color = u_fog_hcolor_end_dist.rgb;
	vec3 height_end_color = u_fog_hend_color_offset.rgb;

	// height fog colour blend
	vec3 h_color = mix(height_color, height_end_color, color_dist);

	// main fog colour blend
	vec3 main_color = mix(fog_color, end_fog_color, clamp(color_dist_cube, 0.0, 1.0));

	// combine main + height colours
	vec3 final_fog_color = mix(main_color, h_color, height_fog);

	// sun glow
	float sun_angle = u_fog_sun_angle_color.x;
	vec3 sun_fog_color = u_fog_sun_angle_color.yzw;
	float sun_pct = u_fog_sun_pct_str.x;
	vec3 sun_dir = u_fog_sun_dir_z_scalar.xyz;

	vec3 view_dir = normalize(to_vertex);
	float n_dot_sun = clamp(dot(view_dir, sun_dir), 0.0, 1.0);
	float sun_threshold = clamp(n_dot_sun - sun_angle, 0.0, 1.0);

	if (sun_threshold > 0.0) {
		float inv_range = 1.0 / max(1.0 - sun_angle, 0.001);
		float sun_factor = sun_threshold * inv_range;
		sun_factor = sun_factor * sun_factor * sun_factor;
		vec3 sun_blend_color = mix(final_fog_color, sun_fog_color, sun_pct);
		final_fog_color = mix(final_fog_color, sun_blend_color, sun_factor);
	}

	// final mix: fog replaces pixel colour
	return mix(final_fog_color, color, fog_factor);
}

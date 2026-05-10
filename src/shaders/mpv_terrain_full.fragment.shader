#version 300 es
precision highp float;
precision highp sampler2DArray;

in vec3 v_normal;
in vec3 v_position;
in vec2 v_texcoord;
in vec4 v_color;

out vec4 frag_color;

uniform sampler2D u_tex0;
uniform sampler2D u_tex1;
uniform sampler2D u_tex2;
uniform sampler2D u_tex3;
uniform sampler2D u_tex4;
uniform sampler2D u_tex5;
uniform sampler2D u_tex6;
uniform sampler2D u_tex7;
uniform sampler2DArray u_alpha_maps;

uniform int u_layer_count;
uniform int u_alpha_offset;
uniform vec2 u_chunk_offset;

uniform int u_diffuse_slot[8];
uniform int u_height_slot[8];
uniform float u_layer_scale[8];
uniform float u_height_scale[8];
uniform float u_height_offset[8];

uniform vec3 u_light_dir;
uniform vec3 u_sun_color;
uniform float u_sun_intensity;

uniform vec3 u_camera_pos;
uniform vec3 u_fog_color;
uniform float u_fog_start;
uniform float u_fog_end;

const vec3 SKY_COLOR = vec3(0.4, 0.5, 0.7);
const vec3 GROUND_COLOR = vec3(0.25, 0.2, 0.15);
const float AMBIENT = 0.25;

vec4 sample_slot(int slot, vec2 uv) {
	switch (slot) {
		case 0: return texture(u_tex0, uv);
		case 1: return texture(u_tex1, uv);
		case 2: return texture(u_tex2, uv);
		case 3: return texture(u_tex3, uv);
		case 4: return texture(u_tex4, uv);
		case 5: return texture(u_tex5, uv);
		case 6: return texture(u_tex6, uv);
		case 7: return texture(u_tex7, uv);
	}
	return vec4(0.0);
}

void main() {
	vec2 chunk_uv = v_texcoord * 16.0 - u_chunk_offset;

	float alphas[8];
	alphas[0] = 1.0;
	for (int i = 1; i < 8; i++) {
		if (i < u_layer_count)
			alphas[i] = texture(u_alpha_maps, vec3(chunk_uv, float(u_alpha_offset + i - 1))).r;
		else
			alphas[i] = 0.0;
	}

	float alpha_sum = alphas[1] + alphas[2] + alphas[3] + alphas[4] + alphas[5] + alphas[6] + alphas[7];

	float layer_weights[8];
	layer_weights[0] = 1.0 - clamp(alpha_sum, 0.0, 1.0);
	for (int i = 1; i < 8; i++)
		layer_weights[i] = alphas[i];

	// height-weighted blending
	float layer_pcts[8];
	for (int i = 0; i < 8; i++) {
		if (i >= u_layer_count) {
			layer_pcts[i] = 0.0;
			continue;
		}

		vec2 tc = chunk_uv * (8.0 / u_layer_scale[i]);
		float height_val = sample_slot(u_height_slot[i], tc).a;
		layer_pcts[i] = layer_weights[i] * (height_val * u_height_scale[i] + u_height_offset[i]);
	}

	float max_pct = 0.0;
	for (int i = 0; i < 8; i++)
		max_pct = max(max_pct, layer_pcts[i]);

	for (int i = 0; i < 8; i++)
		layer_pcts[i] *= 1.0 - clamp(max_pct - layer_pcts[i], 0.0, 1.0);

	float pct_sum = 0.0;
	for (int i = 0; i < 8; i++)
		pct_sum += layer_pcts[i];

	if (pct_sum > 0.0) {
		for (int i = 0; i < 8; i++)
			layer_pcts[i] /= pct_sum;
	}

	vec3 terrain_color = vec3(0.0);
	for (int i = 0; i < 8; i++) {
		if (i >= u_layer_count)
			break;

		vec2 tc = chunk_uv * (8.0 / u_layer_scale[i]);
		terrain_color += sample_slot(u_diffuse_slot[i], tc).rgb * layer_pcts[i];
	}

	terrain_color *= v_color.rgb * 2.0;

	vec3 n = normalize(v_normal);
	float sky_factor = 0.5 + 0.5 * n.y;
	vec3 ambient = mix(GROUND_COLOR, SKY_COLOR, sky_factor) * AMBIENT;
	float diffuse = max(dot(n, u_light_dir), 0.0);
	vec3 color = terrain_color * (ambient + u_sun_color * diffuse * u_sun_intensity);

	float dist = distance(v_position, u_camera_pos);
	float fog = clamp((dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
	color = mix(color, u_fog_color, fog);

	frag_color = vec4(color, 1.0);
}

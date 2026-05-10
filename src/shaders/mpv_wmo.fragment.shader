#version 300 es
precision highp float;
precision highp int;

// inputs from vertex shader
in vec2 v_texcoord;
in vec2 v_texcoord2;
in vec2 v_texcoord3;
in vec2 v_texcoord4;
in vec3 v_normal;
in vec3 v_position;
in vec4 v_color;
in vec4 v_color2;
in vec4 v_color3;

// textures
uniform sampler2D u_texture1;
uniform sampler2D u_texture2;
uniform sampler2D u_texture3;
uniform sampler2D u_texture4;
uniform sampler2D u_texture5;
uniform sampler2D u_texture6;
uniform sampler2D u_texture7;
uniform sampler2D u_texture8;
uniform sampler2D u_texture9;

// material parameters
uniform int u_pixel_shader;
uniform int u_blend_mode;
uniform int u_apply_lighting;

uniform vec3 u_camera_pos;

out vec4 frag_color;

#include "mpv_light.inc.glsl"
#include "mpv_fog.inc.glsl"

void main() {
	vec4 tex1 = texture(u_texture1, v_texcoord);
	vec4 tex2 = texture(u_texture2, v_texcoord2);

	vec3 mat_diffuse;
	vec3 emissive = vec3(0.0);
	float final_opacity = 1.0;

	switch (u_pixel_shader) {
		case 0: // MapObjDiffuse
			mat_diffuse = tex1.rgb;
			final_opacity = tex1.a;
			break;

		case 1: // MapObjSpecular
			mat_diffuse = tex1.rgb;
			final_opacity = tex1.a;
			break;

		case 2: // MapObjMetal
			mat_diffuse = tex1.rgb;
			final_opacity = tex1.a;
			break;

		case 3: // MapObjEnv
			mat_diffuse = tex1.rgb;
			emissive = tex2.rgb * tex1.a;
			final_opacity = 1.0;
			break;

		case 4: // MapObjOpaque
			mat_diffuse = tex1.rgb;
			final_opacity = 1.0;
			break;

		case 5: // MapObjEnvMetal
			mat_diffuse = tex1.rgb;
			emissive = (tex1.rgb * tex1.a) * tex2.rgb;
			final_opacity = 1.0;
			break;

		case 6: { // MapObjTwoLayerDiffuse
			vec3 layer1 = tex1.rgb;
			vec3 layer2 = mix(layer1, tex2.rgb, tex2.a);
			mat_diffuse = mix(layer2, layer1, v_color2.a);
			final_opacity = tex1.a;
			break;
		}

		case 7: { // MapObjTwoLayerEnvMetal
			vec4 tex3 = texture(u_texture3, v_texcoord3);
			vec4 color_mix = mix(tex1, tex1, 1.0 - v_color2.a);
			mat_diffuse = color_mix.rgb;
			emissive = (color_mix.rgb * color_mix.a) * tex3.rgb;
			final_opacity = tex1.a;
			break;
		}

		case 8: // MapObjTwoLayerTerrain
			mat_diffuse = mix(tex2.rgb, tex1.rgb, v_color2.a);
			final_opacity = tex1.a;
			break;

		case 9: // MapObjDiffuseEmissive
			mat_diffuse = tex1.rgb;
			emissive = tex2.rgb * tex2.a * v_color2.a;
			final_opacity = tex1.a;
			break;

		case 10: { // MapObjMaskedEnvMetal
			vec4 tex3 = texture(u_texture3, v_texcoord3);
			float mix_factor = clamp(tex3.a * v_color2.a, 0.0, 1.0);
			mat_diffuse = mix(mix((tex1.rgb * tex2.rgb) * 2.0, tex3.rgb, mix_factor), tex1.rgb, tex1.a);
			final_opacity = tex1.a;
			break;
		}

		case 11: { // MapObjEnvMetalEmissive
			vec4 tex3 = texture(u_texture3, v_texcoord3);
			mat_diffuse = tex1.rgb;
			emissive = ((tex1.rgb * tex1.a) * tex2.rgb) + ((tex3.rgb * tex3.a) * v_color2.a);
			final_opacity = tex1.a;
			break;
		}

		case 12: // MapObjTwoLayerDiffuseOpaque
			mat_diffuse = mix(tex2.rgb, tex1.rgb, v_color.a);
			final_opacity = 1.0;
			break;

		case 13: { // MapObjTwoLayerDiffuseEmissive
			vec3 t1_diffuse = tex2.rgb * (1.0 - tex2.a);
			mat_diffuse = mix(t1_diffuse, tex1.rgb, v_color2.a);
			emissive = (tex2.rgb * tex2.a) * (1.0 - v_color2.a);
			final_opacity = tex1.a;
			break;
		}

		case 14: { // MapObjAdditiveMaskedEnvMetal
			vec4 tex3 = texture(u_texture3, v_texcoord3);
			mat_diffuse = mix(
				(tex1.rgb * tex2.rgb * 2.0) + (tex3.rgb * clamp(tex3.a * v_color2.a, 0.0, 1.0)),
				tex1.rgb,
				tex1.a
			);
			final_opacity = 1.0;
			break;
		}

		case 15: { // MapObjTwoLayerDiffuseMod2x
			vec4 tex3 = texture(u_texture3, v_texcoord3);
			vec3 layer1 = tex1.rgb;
			vec3 layer2 = mix(layer1, tex2.rgb, tex2.a);
			vec3 layer3 = mix(layer2, layer1, v_color2.a);
			mat_diffuse = layer3 * tex3.rgb * 2.0;
			final_opacity = tex1.a;
			break;
		}

		case 16: // MapObjTwoLayerDiffuseMod2xNA
			mat_diffuse = mix(tex1.rgb, (tex1.rgb * tex2.rgb) * 2.0, v_color2.a);
			final_opacity = tex1.a;
			break;

		case 17: { // MapObjTwoLayerDiffuseAlpha
			vec4 tex3 = texture(u_texture3, v_texcoord3);
			vec3 layer1 = tex1.rgb;
			vec3 layer2 = mix(layer1, tex2.rgb, tex2.a);
			vec3 layer3 = mix(layer2, layer1, tex3.a);
			mat_diffuse = (layer3 * tex3.rgb) * 2.0;
			final_opacity = tex1.a;
			break;
		}

		case 18: // MapObjLod
			mat_diffuse = tex1.rgb;
			final_opacity = tex1.a;
			break;

		case 19: // MapObjParallax (simplified)
			mat_diffuse = tex1.rgb;
			final_opacity = tex1.a;
			break;

		case 20: { // MapObjUnkShader
			vec4 tex2_20 = texture(u_texture2, v_texcoord);
			vec4 tex3_20 = texture(u_texture3, v_texcoord2);
			vec4 tex4_20 = texture(u_texture4, v_texcoord3);
			vec4 tex5_20 = texture(u_texture5, v_texcoord4);
			vec4 tex6_20 = texture(u_texture6, v_texcoord);
			vec4 tex7_20 = texture(u_texture7, v_texcoord2);
			vec4 tex8_20 = texture(u_texture8, v_texcoord3);
			vec4 tex9_20 = texture(u_texture9, v_texcoord4);

			float second_color_sum = dot(v_color3.bgr, vec3(1.0));
			vec4 weights = vec4(v_color3.bgr, 1.0 - clamp(second_color_sum, 0.0, 1.0));
			vec4 heights = max(vec4(tex6_20.a, tex7_20.a, tex8_20.a, tex9_20.a), 0.004);
			vec4 alpha_vec = weights * heights;
			float weights_max = max(alpha_vec.r, max(alpha_vec.g, max(alpha_vec.b, alpha_vec.a)));
			vec4 alpha_vec2 = (1.0 - clamp(vec4(weights_max) - alpha_vec, 0.0, 1.0)) * alpha_vec;
			vec4 alpha_normalized = alpha_vec2 * (1.0 / dot(alpha_vec2, vec4(1.0)));

			vec4 tex_mixed = tex2_20 * alpha_normalized.r +
							tex3_20 * alpha_normalized.g +
							tex4_20 * alpha_normalized.b +
							tex5_20 * alpha_normalized.a;

			vec4 env_tex = vec4(0.0); // env texture would use posToTexCoord - simplified here
			emissive = (tex_mixed.a * env_tex.rgb) * tex_mixed.rgb;
			mat_diffuse = mix(tex_mixed.rgb, vec3(0.0), v_color3.a);
			final_opacity = tex_mixed.a;
			break;
		}
		
		default: // fallback
			mat_diffuse = tex1.rgb;
			final_opacity = tex1.a;
			break;
	}

	// alpha test for blend mode 1 (alpha key)
	if (u_blend_mode == 1 && final_opacity < 0.904)
		discard;

	// blend modes 0/1 force opaque output
	if (u_blend_mode <= 1)
		final_opacity = 1.0;

	// lighting: MOCV (v_color.rgb) is precomputed light added to ambient for lit batches
	vec3 lit;
	if (u_apply_lighting != 0)
		lit = calc_exterior_light(mat_diffuse, normalize(v_normal), v_color.rgb);
	else
		lit = mat_diffuse;

	lit += emissive;
	lit = apply_fog(lit, v_position, u_camera_pos);

	frag_color = vec4(lit, final_opacity);
}

/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const core = require('../core');
const Shaders = require('../3D/Shaders');
const ShaderMapper = require('../3D/ShaderMapper');
const VertexArray = require('../3D/gl/VertexArray');
const UniformBuffer = require('../3D/gl/UniformBuffer');
const GLTexture = require('../3D/gl/GLTexture');
const GLContext = require('../3D/gl/GLContext');
const BLPFile = require('../casc/blp');
const M2Loader = require('../3D/loaders/M2Loader');
const log = require('../log');

const MAX_BONES = 256;
const ALPHA_TEST_VALUE = 0.501960814;
const M2_VERTEX_STRIDE = 48;

const IDENTITY_MAT4 = new Float32Array([
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1
]);

const VERTEX_SHADER_IDS = {
	'Diffuse_T1': 0, 'Diffuse_Env': 1, 'Diffuse_T1_T2': 2,
	'Diffuse_T1_Env': 3, 'Diffuse_Env_T1': 4, 'Diffuse_Env_Env': 5,
	'Diffuse_T1_Env_T1': 6, 'Diffuse_T1_T1': 7, 'Diffuse_T1_T1_T1': 8,
	'Diffuse_EdgeFade_T1': 9, 'Diffuse_T2': 10, 'Diffuse_T1_Env_T2': 11,
	'Diffuse_EdgeFade_T1_T2': 12, 'Diffuse_EdgeFade_Env': 13,
	'Diffuse_T1_T2_T1': 14, 'Diffuse_T1_T2_T3': 15, 'Color_T1_T2_T3': 16,
	'BW_Diffuse_T1': 17, 'BW_Diffuse_T1_T2': 18
};

const PIXEL_SHADER_IDS = {
	'Combiners_Opaque': 0, 'Combiners_Mod': 1, 'Combiners_Opaque_Mod': 2,
	'Combiners_Opaque_Mod2x': 3, 'Combiners_Opaque_Mod2xNA': 4,
	'Combiners_Opaque_Opaque': 5, 'Combiners_Mod_Mod': 6,
	'Combiners_Mod_Mod2x': 7, 'Combiners_Mod_Add': 8,
	'Combiners_Mod_Mod2xNA': 9, 'Combiners_Mod_AddNA': 10,
	'Combiners_Mod_Opaque': 11, 'Combiners_Opaque_Mod2xNA_Alpha': 12,
	'Combiners_Opaque_AddAlpha': 13, 'Combiners_Opaque_AddAlpha_Alpha': 14,
	'Combiners_Opaque_Mod2xNA_Alpha_Add': 15, 'Combiners_Mod_AddAlpha': 16,
	'Combiners_Mod_AddAlpha_Alpha': 17, 'Combiners_Opaque_Alpha_Alpha': 18,
	'Combiners_Opaque_Mod2xNA_Alpha_3s': 19, 'Combiners_Opaque_AddAlpha_Wgt': 20,
	'Combiners_Mod_Add_Alpha': 21, 'Combiners_Opaque_ModNA_Alpha': 22,
	'Combiners_Mod_AddAlpha_Wgt': 23, 'Combiners_Opaque_Mod_Add_Wgt': 24,
	'Combiners_Opaque_Mod2xNA_Alpha_UnshAlpha': 25,
	'Combiners_Mod_Dual_Crossfade': 26, 'Combiners_Opaque_Mod2xNA_Alpha_Alpha': 27,
	'Combiners_Mod_Masked_Dual_Crossfade': 28, 'Combiners_Opaque_Alpha': 29,
	'Guild': 30, 'Guild_NoBorder': 31, 'Guild_Opaque': 32,
	'Combiners_Mod_Depth': 33, 'Illum': 34,
	'Combiners_Mod_Mod_Mod_Const': 35, 'Combiners_Mod_Mod_Depth': 36
};

const M2BLEND_TO_EGX = [
	GLContext.BlendMode.OPAQUE,
	GLContext.BlendMode.ALPHA_KEY,
	GLContext.BlendMode.ALPHA,
	GLContext.BlendMode.NO_ALPHA_ADD,
	GLContext.BlendMode.ADD,
	GLContext.BlendMode.MOD,
	GLContext.BlendMode.MOD2X,
	GLContext.BlendMode.BLEND_ADD
];

// ---- shared animation utilities (same as M2Renderer) ----

function find_keyframe(timestamps, time) {
	let lo = 0, hi = timestamps.length - 1;
	while (lo < hi) {
		const mid = (lo + hi + 1) >> 1;
		if (timestamps[mid] <= time)
			lo = mid;
		else
			hi = mid - 1;
	}
	return lo;
}

function lerp(a, b, t) {
	return a + (b - a) * t;
}

function quat_slerp(out, ax, ay, az, aw, bx, by, bz, bw, t) {
	let cosom = ax * bx + ay * by + az * bz + aw * bw;

	if (cosom < 0) {
		cosom = -cosom;
		bx = -bx; by = -by; bz = -bz; bw = -bw;
	}

	let scale0, scale1;
	if (1 - cosom > 0.000001) {
		const omega = Math.acos(cosom);
		const sinom = Math.sin(omega);
		scale0 = Math.sin((1 - t) * omega) / sinom;
		scale1 = Math.sin(t * omega) / sinom;
	} else {
		scale0 = 1 - t;
		scale1 = t;
	}

	out[0] = scale0 * ax + scale1 * bx;
	out[1] = scale0 * ay + scale1 * by;
	out[2] = scale0 * az + scale1 * bz;
	out[3] = scale0 * aw + scale1 * bw;
}

function sample_vec3(timestamps, values, time_ms) {
	if (!timestamps || timestamps.length === 0)
		return null;

	if (timestamps.length === 1 || time_ms <= timestamps[0])
		return values[0];

	if (time_ms >= timestamps[timestamps.length - 1])
		return values[values.length - 1];

	const frame = find_keyframe(timestamps, time_ms);
	const t0 = timestamps[frame];
	const t1 = timestamps[frame + 1];
	const dt = t1 - t0;
	const alpha = dt > 0 ? Math.min((time_ms - t0) / dt, 1) : 0;
	const v0 = values[frame];
	const v1 = values[frame + 1];
	return [lerp(v0[0], v1[0], alpha), lerp(v0[1], v1[1], alpha), lerp(v0[2], v1[2], alpha)];
}

function sample_quat(timestamps, values, time_ms) {
	if (!timestamps || timestamps.length === 0)
		return null;

	if (timestamps.length === 1 || time_ms <= timestamps[0])
		return values[0];

	if (time_ms >= timestamps[timestamps.length - 1])
		return values[values.length - 1];

	const frame = find_keyframe(timestamps, time_ms);
	const t0 = timestamps[frame];
	const t1 = timestamps[frame + 1];
	const dt = t1 - t0;
	const alpha = dt > 0 ? Math.min((time_ms - t0) / dt, 1) : 0;
	const q0 = values[frame];
	const q1 = values[frame + 1];
	const out = [0, 0, 0, 1];
	quat_slerp(out, q0[0], q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3], alpha);
	return out;
}

const _s_local = new Float32Array(16);
const _s_trans = new Float32Array(16);
const _s_rot = new Float32Array(16);
const _s_scale = new Float32Array(16);
const _s_pivot = new Float32Array(16);
const _s_neg_pivot = new Float32Array(16);
const _s_result = new Float32Array(16);
const _s_calculated = new Uint8Array(MAX_BONES);

function mat4_multiply(out, a, b) {
	const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
	const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
	const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
	const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

	let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
	out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
	out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
	out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
	out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

	b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
	out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
	out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
	out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
	out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

	b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
	out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
	out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
	out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
	out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

	b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
	out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
	out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
	out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
	out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
}

function mat4_from_translation(out, x, y, z) {
	out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0;
	out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 0;
	out[8] = 0; out[9] = 0; out[10] = 1; out[11] = 0;
	out[12] = x; out[13] = y; out[14] = z; out[15] = 1;
}

function mat4_from_quat(out, x, y, z, w) {
	const x2 = x + x, y2 = y + y, z2 = z + z;
	const xx = x * x2, xy = x * y2, xz = x * z2;
	const yy = y * y2, yz = y * z2, zz = z * z2;
	const wx = w * x2, wy = w * y2, wz = w * z2;

	out[0] = 1 - (yy + zz); out[1] = xy + wz; out[2] = xz - wy; out[3] = 0;
	out[4] = xy - wz; out[5] = 1 - (xx + zz); out[6] = yz + wx; out[7] = 0;
	out[8] = xz + wy; out[9] = yz - wx; out[10] = 1 - (xx + yy); out[11] = 0;
	out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
}

function mat4_from_scale(out, x, y, z) {
	out[0] = x; out[1] = 0; out[2] = 0; out[3] = 0;
	out[4] = 0; out[5] = y; out[6] = 0; out[7] = 0;
	out[8] = 0; out[9] = 0; out[10] = z; out[11] = 0;
	out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
}

function compute_bone_matrices(bones, anim_idx, time_ms, bone_matrices, global_seq_times, global_loops) {
	const bone_count = Math.min(bones.length, MAX_BONES);
	_s_calculated.fill(0);

	const calc_bone = (idx) => {
		if (_s_calculated[idx])
			return;

		const bone = bones[idx];
		const parent_idx = bone.parentBone;

		if (parent_idx >= 0 && parent_idx < bone_count)
			calc_bone(parent_idx);

		const pivot = bone.pivot;
		const px = pivot[0], py = pivot[1], pz = pivot[2];

		const resolve_time = (track) => {
			const gs = track.globalSeq;
			if (gs !== undefined && global_loops && gs < global_loops.length) {
				const gs_dur = global_loops[gs];
				if (gs_dur > 0 && global_seq_times)
					return global_seq_times[gs] ?? 0;
			}
			return time_ms;
		};

		const has_trans = bone.translation?.timestamps?.[anim_idx]?.length > 0;
		const has_rot = bone.rotation?.timestamps?.[anim_idx]?.length > 0;
		const has_scale = bone.scale?.timestamps?.[anim_idx]?.length > 0;
		const has_scale_fb = !has_scale && anim_idx !== 0 && bone.scale?.timestamps?.[0]?.length > 0;
		const has_animation = has_trans || has_rot || has_scale || has_scale_fb;

		_s_local.set(IDENTITY_MAT4);

		if (has_animation) {
			mat4_from_translation(_s_pivot, px, py, pz);
			mat4_multiply(_s_result, _s_local, _s_pivot);
			_s_local.set(_s_result);

			if (has_trans) {
				const t_time = resolve_time(bone.translation);
				const ts = bone.translation.timestamps[anim_idx];
				const vals = bone.translation.values[anim_idx];
				const t = sample_vec3(ts, vals, t_time);
				if (t) {
					mat4_from_translation(_s_trans, t[0], t[1], t[2]);
					mat4_multiply(_s_result, _s_local, _s_trans);
					_s_local.set(_s_result);
				}
			}

			if (has_rot) {
				const r_time = resolve_time(bone.rotation);
				const ts = bone.rotation.timestamps[anim_idx];
				const vals = bone.rotation.values[anim_idx];
				const q = sample_quat(ts, vals, r_time);
				if (q) {
					mat4_from_quat(_s_rot, q[0], q[1], q[2], q[3]);
					mat4_multiply(_s_result, _s_local, _s_rot);
					_s_local.set(_s_result);
				}
			}

			if (has_scale || has_scale_fb) {
				const scale_idx = has_scale ? anim_idx : 0;
				const s_time = has_scale ? resolve_time(bone.scale) : 0;
				const ts = bone.scale.timestamps[scale_idx];
				const vals = bone.scale.values[scale_idx];
				const s = sample_vec3(ts, vals, s_time);
				if (s) {
					mat4_from_scale(_s_scale, s[0], s[1], s[2]);
					mat4_multiply(_s_result, _s_local, _s_scale);
					_s_local.set(_s_result);
				}
			}

			mat4_from_translation(_s_neg_pivot, -px, -py, -pz);
			mat4_multiply(_s_result, _s_local, _s_neg_pivot);
			_s_local.set(_s_result);
		}

		const offset = idx * 16;
		if (parent_idx >= 0 && parent_idx < bone_count) {
			const parent_offset = parent_idx * 16;
			const parent_mat = bone_matrices.subarray(parent_offset, parent_offset + 16);
			mat4_multiply(bone_matrices.subarray(offset, offset + 16), parent_mat, _s_local);
		} else {
			bone_matrices.set(_s_local, offset);
		}

		_s_calculated[idx] = 1;
	};

	for (let i = 0; i < bone_count; i++)
		calc_bone(i);
}

function find_anim_index(animations, primary_id, fallback_id) {
	if (!animations || animations.length === 0)
		return -1;

	let primary = -1;
	let fallback = -1;

	for (let i = 0; i < animations.length; i++) {
		if (animations[i].id === primary_id) {
			primary = i;
			break;
		}
		if (fallback_id !== undefined && animations[i].id === fallback_id)
			fallback = i;
	}

	return primary >= 0 ? primary : fallback;
}

function build_model_geometry(m2, skin) {
	const vertex_count = m2.vertices.length / 3;
	if (vertex_count === 0)
		return null;

	const vertex_data = new ArrayBuffer(vertex_count * M2_VERTEX_STRIDE);
	const view = new DataView(vertex_data);
	const has_bones = m2.boneIndices && m2.boneWeights;

	for (let i = 0; i < vertex_count; i++) {
		const offset = i * M2_VERTEX_STRIDE;
		const v = i * 3;
		const uv = i * 2;
		const bi = i * 4;

		view.setFloat32(offset, m2.vertices[v], true);
		view.setFloat32(offset + 4, m2.vertices[v + 1], true);
		view.setFloat32(offset + 8, m2.vertices[v + 2], true);
		view.setFloat32(offset + 12, m2.normals[v], true);
		view.setFloat32(offset + 16, m2.normals[v + 1], true);
		view.setFloat32(offset + 20, m2.normals[v + 2], true);

		if (has_bones) {
			view.setUint8(offset + 24, m2.boneIndices[bi]);
			view.setUint8(offset + 25, m2.boneIndices[bi + 1]);
			view.setUint8(offset + 26, m2.boneIndices[bi + 2]);
			view.setUint8(offset + 27, m2.boneIndices[bi + 3]);
			view.setUint8(offset + 28, m2.boneWeights[bi]);
			view.setUint8(offset + 29, m2.boneWeights[bi + 1]);
			view.setUint8(offset + 30, m2.boneWeights[bi + 2]);
			view.setUint8(offset + 31, m2.boneWeights[bi + 3]);
		}

		view.setFloat32(offset + 32, m2.uv[uv], true);
		view.setFloat32(offset + 36, m2.uv[uv + 1], true);
		view.setFloat32(offset + 40, m2.uv2[uv], true);
		view.setFloat32(offset + 44, m2.uv2[uv + 1], true);
	}

	const index_data = new Uint16Array(skin.triangles.length);
	for (let i = 0; i < skin.triangles.length; i++)
		index_data[i] = skin.indices[skin.triangles[i]];

	return { vertex_data, index_data, index_count: index_data.length };
}

function build_draw_calls(m2, skin) {
	const draw_calls = [];

	for (let i = 0; i < skin.subMeshes.length; i++) {
		const submesh = skin.subMeshes[i];
		const tex_unit = skin.textureUnits.find(tu => tu.skinSectionIndex === i);

		let tex_file_ids = [0, 0, 0, 0];
		let tex_flags = [0, 0, 0, 0];
		let vertex_shader = 0;
		let pixel_shader = 0;
		let blend_mode = 0;
		let flags = 0;

		if (tex_unit) {
			const texture_count = tex_unit.textureCount;

			for (let j = 0; j < Math.min(texture_count, 4); j++) {
				const combo_idx = tex_unit.textureComboIndex + j;
				if (combo_idx < m2.textureCombos.length) {
					const tex_idx = m2.textureCombos[combo_idx];
					if (tex_idx < m2.textures.length) {
						const tex = m2.textures[tex_idx];
						if (tex.fileDataID > 0) {
							tex_file_ids[j] = tex.fileDataID;
							tex_flags[j] = tex.flags;
						}
					}
				}
			}

			const vs_name = ShaderMapper.getVertexShader(texture_count, tex_unit.shaderID);
			vertex_shader = VERTEX_SHADER_IDS[vs_name] ?? 0;

			const ps_name = ShaderMapper.getPixelShader(texture_count, tex_unit.shaderID);
			pixel_shader = PIXEL_SHADER_IDS[ps_name] ?? 0;

			const mat = m2.materials?.[tex_unit.materialIndex];
			if (mat) {
				blend_mode = M2BLEND_TO_EGX[mat.blendingMode] ?? mat.blendingMode;
				flags = mat.flags;
			}
		}

		draw_calls.push({
			start: submesh.triangleStart,
			count: submesh.triangleCount,
			vertex_shader, pixel_shader,
			blend_mode, flags,
			tex_file_ids, tex_flags
		});
	}

	draw_calls.sort((a, b) => {
		const a_opaque = a.blend_mode <= 1 ? 0 : 1;
		const b_opaque = b.blend_mode <= 1 ? 0 : 1;
		if (a_opaque !== b_opaque)
			return a_opaque - b_opaque;

		return a.blend_mode - b.blend_mode;
	});

	return draw_calls;
}

class SkyboxM2Renderer {
	constructor(gl_context) {
		this.ctx = gl_context;
		this.gl = gl_context.gl;
		this.shader = Shaders.create_program(gl_context, 'mpv_m2');
		this._casc = core.view.casc;

		this._texture_cache = new Map();
		this._default_texture = this._create_default_texture();

		// current loaded model state
		this._file_data_id = 0;
		this._flags = 0;
		this._vao = null;
		this._instance_buffer = null;
		this._draw_calls = null;
		this._texture_ids = null;

		// animation state
		this._bones = null;
		this._bone_count = 0;
		this._bone_ubo = null;
		this._bone_matrices = null;
		this._anim_index = -1;
		this._anim_duration = 0;
		this._anim_time = 0;
		this._global_loops = null;
		this._global_seq_times = null;

		// loading state
		this._loading_id = 0;
		this._disposed = false;

		// model matrix (set to camera position each frame)
		this._model_matrix = new Float32Array(IDENTITY_MAT4);

		this.shader.bind_uniform_block('BoneMatrices', 0);
	}

	_create_default_texture() {
		const tex = new GLTexture(this.ctx);
		tex.set_rgba(new Uint8Array([255, 255, 255, 255]), 1, 1, { has_alpha: false });
		return tex;
	}

	get is_loaded() {
		return this._vao !== null;
	}

	get is_loading() {
		return this._loading_id > 0;
	}

	/**
	 * set the active skybox model. if file_data_id differs from current,
	 * disposes old model and starts loading the new one.
	 * @param {number} file_data_id - M2 fileDataID (0 to clear)
	 * @param {number} flags - LightSkybox.Flags
	 */
	set_model(file_data_id, flags) {
		if (file_data_id === this._file_data_id) {
			this._flags = flags;
			return;
		}

		this._dispose_model();

		if (file_data_id <= 0) {
			this._file_data_id = 0;
			this._flags = 0;
			return;
		}

		this._file_data_id = file_data_id;
		this._flags = flags;
		this._loading_id = file_data_id;
		this._load_model(file_data_id);
	}

	async _load_model(file_data_id) {
		try {
			const file = await this._casc.getFile(file_data_id);

			if (this._disposed || this._loading_id !== file_data_id)
				return;

			const m2 = new M2Loader(file);
			await m2.load();

			if (!m2.vertices || m2.vertices.length === 0 || this._disposed || this._loading_id !== file_data_id)
				return;

			const skins = m2.skins;
			if (!skins || skins.length === 0)
				return;

			const skin = skins[0];
			if (!skin.isLoaded)
				await skin.load();

			if (!skin.isLoaded || this._disposed || this._loading_id !== file_data_id)
				return;

			// load animation data — use first available animation (skyboxes often use anim 0)
			let bones = null;
			let bone_count = 0;
			let anim_index = -1;
			let anim_duration = 0;
			let global_loops = [];
			let global_seq_times = null;

			if (m2.bones && m2.bones.length > 0) {
				bones = m2.bones;
				bone_count = Math.min(bones.length, MAX_BONES);
				global_loops = m2.globalLoops || [];
				global_seq_times = new Float32Array(global_loops.length);

				// skybox M2s typically use anim 0 (stand)
				anim_index = find_anim_index(m2.animations, 0, 147);

				if (anim_index >= 0) {
					await m2.loadAnimsForIndex(anim_index);

					let resolved_idx = anim_index;
					let resolved_anim = m2.animations[anim_index];
					while (resolved_anim && (resolved_anim.flags & 0x40) === 0x40) {
						resolved_idx = resolved_anim.aliasNext;
						resolved_anim = m2.animations[resolved_idx];
						await m2.loadAnimsForIndex(resolved_idx);
					}

					anim_index = resolved_idx;
					anim_duration = m2.animations[anim_index]?.duration ?? 0;
				}
			}

			const geo = build_model_geometry(m2, skin);
			if (!geo || this._disposed || this._loading_id !== file_data_id)
				return;

			const draw_calls = build_draw_calls(m2, skin);

			// fetch textures
			const texture_loads = new Map();
			for (const dc of draw_calls) {
				for (let j = 0; j < 4; j++) {
					const fid = dc.tex_file_ids[j];
					if (fid > 0 && !texture_loads.has(fid) && !this._texture_cache.has(fid))
						texture_loads.set(fid, { flags: dc.tex_flags[j], blp: null });
				}
			}

			for (const [fid, entry] of texture_loads) {
				if (this._disposed || this._loading_id !== file_data_id)
					return;

				try {
					const data = await this._casc.getFile(fid);
					entry.blp = new BLPFile(data);
				} catch (e) {
					log.write('Failed to load skybox texture ' + fid + ': ' + e.message);
				}
			}

			if (this._disposed || this._loading_id !== file_data_id)
				return;

			this._upload_model(geo, draw_calls, texture_loads, bones, bone_count, anim_index, anim_duration, global_loops, global_seq_times);
		} catch (e) {
			log.write('Failed to load skybox M2 ' + file_data_id + ': ' + e.message);
		}

		if (this._loading_id === file_data_id)
			this._loading_id = 0;
	}

	_upload_model(geo, draw_calls, texture_loads, bones, bone_count, anim_index, anim_duration, global_loops, global_seq_times) {
		const gl = this.gl;
		const vao = new VertexArray(this.ctx);
		vao.bind();

		vao.set_vertex_buffer(geo.vertex_data);

		const stride = M2_VERTEX_STRIDE;
		gl.enableVertexAttribArray(0);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
		gl.enableVertexAttribArray(1);
		gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 12);
		gl.enableVertexAttribArray(2);
		gl.vertexAttribIPointer(2, 4, gl.UNSIGNED_BYTE, stride, 24);
		gl.enableVertexAttribArray(3);
		gl.vertexAttribPointer(3, 4, gl.UNSIGNED_BYTE, true, stride, 28);
		gl.enableVertexAttribArray(4);
		gl.vertexAttribPointer(4, 2, gl.FLOAT, false, stride, 32);
		gl.enableVertexAttribArray(5);
		gl.vertexAttribPointer(5, 2, gl.FLOAT, false, stride, 40);

		vao.set_index_buffer(geo.index_data);

		// instance buffer (single instance for model matrix)
		const instance_buffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, instance_buffer);
		gl.bufferData(gl.ARRAY_BUFFER, this._model_matrix, gl.DYNAMIC_DRAW);

		for (let i = 0; i < 4; i++) {
			const loc = 6 + i;
			gl.enableVertexAttribArray(loc);
			gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 64, i * 16);
			gl.vertexAttribDivisor(loc, 1);
		}

		// upload textures
		const texture_ids = new Set();
		for (const [fid, tex_load] of texture_loads) {
			texture_ids.add(fid);
			let cached = this._texture_cache.get(fid);
			if (cached) {
				cached.ref_count++;
			} else {
				const gl_tex = new GLTexture(this.ctx);
				if (tex_load.blp)
					gl_tex.set_blp(tex_load.blp, { flags: tex_load.flags });

				cached = { texture: gl_tex, ref_count: 1 };
				this._texture_cache.set(fid, cached);
			}
		}

		this._vao = vao;
		this._instance_buffer = instance_buffer;
		this._draw_calls = draw_calls;
		this._texture_ids = texture_ids;

		// animation
		this._bones = bones;
		this._bone_count = bone_count;
		this._anim_index = anim_index;
		this._anim_duration = anim_duration;
		this._anim_time = 0;
		this._global_loops = global_loops;
		this._global_seq_times = global_seq_times;

		if (bone_count > 0) {
			const ubo = new UniformBuffer(this.ctx, MAX_BONES * 64);
			this._bone_ubo = ubo;
			this._bone_matrices = new Float32Array(bone_count * 16);

			for (let i = 0; i < bone_count; i++)
				this._bone_matrices.set(IDENTITY_MAT4, i * 16);

			if (anim_index >= 0)
				compute_bone_matrices(bones, anim_index, 0, this._bone_matrices, global_seq_times, global_loops);

			ubo.set_mat4_array(0, this._bone_matrices, bone_count);
			ubo.upload();
		}
	}

	/**
	 * update animation and model matrix.
	 * @param {Float32Array} camera_pos - viewer camera position
	 * @param {number} delta_time - seconds since last frame
	 * @param {number} time_of_day - 0-2880 half-minutes
	 */
	update(camera_pos, delta_time, time_of_day) {
		if (!this._vao)
			return;

		// position model at camera (skybox follows camera)
		this._model_matrix[12] = camera_pos[0];
		this._model_matrix[13] = camera_pos[1];
		this._model_matrix[14] = camera_pos[2];

		// upload model matrix to instance buffer
		const gl = this.gl;
		gl.bindBuffer(gl.ARRAY_BUFFER, this._instance_buffer);
		gl.bufferSubData(gl.ARRAY_BUFFER, 0, this._model_matrix);

		// update animation
		if (this._bone_ubo && this._anim_index >= 0)
			this._update_animation(delta_time, time_of_day);
	}

	_update_animation(delta_time, time_of_day) {
		const animate_with_tod = (this._flags & 0x1) !== 0;

		if (animate_with_tod) {
			// drive animation by time-of-day percentage
			const pct = time_of_day / 2880.0;
			this._anim_time = this._anim_duration > 0 ? pct * this._anim_duration : 0;
		} else {
			// real-time animation
			const dt_ms = delta_time * 1000;
			this._anim_time += dt_ms;
			if (this._anim_duration > 0)
				this._anim_time %= this._anim_duration;
		}

		// advance global sequence timers (always real-time)
		if (this._global_seq_times) {
			const dt_ms = delta_time * 1000;
			for (let i = 0; i < this._global_seq_times.length; i++) {
				this._global_seq_times[i] += dt_ms;
				const gs_dur = this._global_loops[i];
				if (gs_dur > 0)
					this._global_seq_times[i] %= gs_dur;
			}
		}

		compute_bone_matrices(
			this._bones, this._anim_index, this._anim_time,
			this._bone_matrices, this._global_seq_times, this._global_loops
		);

		this._bone_ubo.set_mat4_array(0, this._bone_matrices, this._bone_count);
		this._bone_ubo.upload();
	}

	/**
	 * render the skybox M2 model.
	 * renders with: no depth write, no fog, no scene lighting, far depth range.
	 */
	render(view_matrix, projection_matrix) {
		if (!this._vao || !this._draw_calls)
			return;

		const gl = this.gl;
		const ctx = this.ctx;
		const shader = this.shader;

		shader.use();
		shader.set_uniform_mat4('u_view', false, view_matrix);
		shader.set_uniform_mat4('u_projection', false, projection_matrix);
		shader.set_uniform_1f('u_alpha_test', ALPHA_TEST_VALUE);
		shader.set_uniform_4f('u_mesh_color', 1, 1, 1, 1);

		shader.set_uniform_1i('u_texture1', 0);
		shader.set_uniform_1i('u_texture2', 1);
		shader.set_uniform_1i('u_texture3', 2);
		shader.set_uniform_1i('u_texture4', 3);

		// skybox: unlit, no fog
		shader.set_uniform_1i('u_lighting_enabled', 0);
		shader.set_uniform_1f('u_fog_enabled', 0.0);

		// no depth write; depth test on so dome geometry self-occludes correctly
		ctx.set_depth_test(true);
		ctx.set_depth_write(false);

		// push skybox to far depth range
		gl.depthRange(0.997, 1.0);

		this._vao.bind();

		if (this._bone_ubo) {
			this._bone_ubo.bind(0);
			shader.set_uniform_1i('u_bone_count', this._bone_count);
		} else {
			shader.set_uniform_1i('u_bone_count', 0);
		}

		for (const dc of this._draw_calls) {
			shader.set_uniform_1i('u_vertex_shader', dc.vertex_shader);
			shader.set_uniform_1i('u_pixel_shader', dc.pixel_shader);
			shader.set_uniform_1i('u_blend_mode', dc.blend_mode);
			shader.set_uniform_1i('u_apply_lighting', 0);

			ctx.apply_blend_mode(dc.blend_mode);

			if (dc.flags & 0x04)
				ctx.set_cull_face(false);
			else {
				ctx.set_cull_face(true);
				ctx.set_cull_mode(gl.BACK);
			}

			for (let t = 0; t < 4; t++) {
				const file_id = dc.tex_file_ids[t];
				const cached = file_id > 0 ? this._texture_cache.get(file_id) : null;
				const tex = cached?.texture ?? this._default_texture;
				tex.bind(t);
			}

			gl.drawElementsInstanced(gl.TRIANGLES, dc.count, this._vao.index_type, dc.start * 2, 1);
		}

		// restore state
		gl.depthRange(0.0, 1.0);
		ctx.set_blend(false);
		ctx.set_depth_test(true);
		ctx.set_depth_write(true);
		ctx.set_cull_face(false);
	}

	_dispose_model() {
		if (this._vao) {
			this._vao.dispose();
			this._vao = null;
		}

		if (this._instance_buffer) {
			this.gl.deleteBuffer(this._instance_buffer);
			this._instance_buffer = null;
		}

		if (this._bone_ubo) {
			this._bone_ubo.dispose();
			this._bone_ubo = null;
		}

		if (this._texture_ids) {
			for (const fid of this._texture_ids) {
				const cached = this._texture_cache.get(fid);
				if (!cached)
					continue;

				cached.ref_count--;
				if (cached.ref_count <= 0) {
					cached.texture.dispose();
					this._texture_cache.delete(fid);
				}
			}
			this._texture_ids = null;
		}

		this._draw_calls = null;
		this._bones = null;
		this._bone_count = 0;
		this._bone_matrices = null;
		this._anim_index = -1;
		this._anim_duration = 0;
		this._anim_time = 0;
		this._global_loops = null;
		this._global_seq_times = null;
		this._model_matrix.set(IDENTITY_MAT4);
	}

	dispose() {
		this._disposed = true;
		this._loading_id = 0;
		this._dispose_model();

		for (const cached of this._texture_cache.values())
			cached.texture.dispose();

		this._texture_cache.clear();

		if (this._default_texture) {
			this._default_texture.dispose();
			this._default_texture = null;
		}

		if (this.shader) {
			Shaders.unregister(this.shader);
			this.shader.dispose();
			this.shader = null;
		}
	}
}

module.exports = SkyboxM2Renderer;

/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const core = require('../core');
const constants = require('../constants');
const Shaders = require('../3D/Shaders');
const ShaderMapper = require('../3D/ShaderMapper');
const VertexArray = require('../3D/gl/VertexArray');
const UniformBuffer = require('../3D/gl/UniformBuffer');
const GLTexture = require('../3D/gl/GLTexture');
const GLContext = require('../3D/gl/GLContext');
const BLPFile = require('../casc/blp');
const M2Loader = require('../3D/loaders/M2Loader');
const ADTLoader = require('../3D/loaders/ADTLoader');
const listfile = require('../casc/listfile');
const log = require('../log');

const DEG_TO_RAD = Math.PI / 180;
const MAP_COORD_BASE = constants.GAME.MAP_COORD_BASE;
const MAX_OBJ0_CONCURRENT = 4;
const MAX_M2_CONCURRENT = 2;
const UPLOAD_BUDGET = 2;
const CAMERA_DIRTY_THRESHOLD_SQ = 33.33 * 33.33;
const M2_VERTEX_STRIDE = 48;
const MAX_BONES = 256;
const ALPHA_TEST_VALUE = 0.501960814;

// stand animation ID 0, fallback to closed (147)
const ANIM_STAND = 0;
const ANIM_CLOSED = 147;

const BBOX_COLOR = new Float32Array([1, 1, 1]);
const DEFAULT_CAMERA_POS = new Float32Array(3);
const DEFAULT_AMBIENT = new Float32Array([0.5, 0.5, 0.5]);
const DEFAULT_GROUND = new Float32Array([0.35, 0.3, 0.25]);
const DEFAULT_DIRECT = new Float32Array([0.5, 0.475, 0.425]);
const DEFAULT_LIGHT_DIR = new Float32Array([-0.4394, 0.8192, 0.3687]);

function set_scene_uniforms(shader, params) {
	const lu = params?.light_uniforms;
	if (lu) {
		shader.set_uniform_3fv('u_light_dir', lu.light_dir);
		shader.set_uniform_3fv('u_ambient_color', lu.ambient_color);
		shader.set_uniform_3fv('u_horizon_ambient_color', lu.horizon_ambient_color);
		shader.set_uniform_3fv('u_ground_ambient_color', lu.ground_ambient_color);
		shader.set_uniform_3fv('u_direct_color', lu.direct_color);
	} else {
		shader.set_uniform_3fv('u_light_dir', DEFAULT_LIGHT_DIR);
		shader.set_uniform_3fv('u_ambient_color', DEFAULT_AMBIENT);
		shader.set_uniform_3fv('u_horizon_ambient_color', DEFAULT_AMBIENT);
		shader.set_uniform_3fv('u_ground_ambient_color', DEFAULT_GROUND);
		shader.set_uniform_3fv('u_direct_color', DEFAULT_DIRECT);
	}

	const fog = params?.fog_uniforms;
	shader.set_uniform_3fv('u_camera_pos', params?.camera_pos ?? DEFAULT_CAMERA_POS);
	if (fog) {
		shader.set_uniform_1f('u_fog_enabled', fog.enabled);
		shader.set_uniform_4fv('u_fog_density_params', fog.density_params);
		shader.set_uniform_4fv('u_fog_height_plane', fog.height_plane);
		shader.set_uniform_4fv('u_fog_color_height_rate', fog.color_height_rate);
		shader.set_uniform_4fv('u_fog_hdensity_end_color', fog.hdensity_end_color);
		shader.set_uniform_4fv('u_fog_sun_angle_color', fog.sun_angle_color);
		shader.set_uniform_4fv('u_fog_hcolor_end_dist', fog.hcolor_end_dist);
		shader.set_uniform_4fv('u_fog_sun_pct_str', fog.sun_pct_str);
		shader.set_uniform_4fv('u_fog_sun_dir_z_scalar', fog.sun_dir_z_scalar);
		shader.set_uniform_4fv('u_fog_height_coeff', fog.height_coeff);
		shader.set_uniform_4fv('u_fog_main_coeff', fog.main_coeff);
		shader.set_uniform_4fv('u_fog_hdensity_coeff', fog.hdensity_coeff);
		shader.set_uniform_4fv('u_fog_distances', fog.distances);
		shader.set_uniform_4fv('u_fog_hend_color_offset', fog.hend_color_offset);
	} else {
		shader.set_uniform_1f('u_fog_enabled', 0.0);
	}
}

const VERTEX_SHADER_IDS = {
	'Diffuse_T1': 0,
	'Diffuse_Env': 1,
	'Diffuse_T1_T2': 2,
	'Diffuse_T1_Env': 3,
	'Diffuse_Env_T1': 4,
	'Diffuse_Env_Env': 5,
	'Diffuse_T1_Env_T1': 6,
	'Diffuse_T1_T1': 7,
	'Diffuse_T1_T1_T1': 8,
	'Diffuse_EdgeFade_T1': 9,
	'Diffuse_T2': 10,
	'Diffuse_T1_Env_T2': 11,
	'Diffuse_EdgeFade_T1_T2': 12,
	'Diffuse_EdgeFade_Env': 13,
	'Diffuse_T1_T2_T1': 14,
	'Diffuse_T1_T2_T3': 15,
	'Color_T1_T2_T3': 16,
	'BW_Diffuse_T1': 17,
	'BW_Diffuse_T1_T2': 18
};

const PIXEL_SHADER_IDS = {
	'Combiners_Opaque': 0,
	'Combiners_Mod': 1,
	'Combiners_Opaque_Mod': 2,
	'Combiners_Opaque_Mod2x': 3,
	'Combiners_Opaque_Mod2xNA': 4,
	'Combiners_Opaque_Opaque': 5,
	'Combiners_Mod_Mod': 6,
	'Combiners_Mod_Mod2x': 7,
	'Combiners_Mod_Add': 8,
	'Combiners_Mod_Mod2xNA': 9,
	'Combiners_Mod_AddNA': 10,
	'Combiners_Mod_Opaque': 11,
	'Combiners_Opaque_Mod2xNA_Alpha': 12,
	'Combiners_Opaque_AddAlpha': 13,
	'Combiners_Opaque_AddAlpha_Alpha': 14,
	'Combiners_Opaque_Mod2xNA_Alpha_Add': 15,
	'Combiners_Mod_AddAlpha': 16,
	'Combiners_Mod_AddAlpha_Alpha': 17,
	'Combiners_Opaque_Alpha_Alpha': 18,
	'Combiners_Opaque_Mod2xNA_Alpha_3s': 19,
	'Combiners_Opaque_AddAlpha_Wgt': 20,
	'Combiners_Mod_Add_Alpha': 21,
	'Combiners_Opaque_ModNA_Alpha': 22,
	'Combiners_Mod_AddAlpha_Wgt': 23,
	'Combiners_Opaque_Mod_Add_Wgt': 24,
	'Combiners_Opaque_Mod2xNA_Alpha_UnshAlpha': 25,
	'Combiners_Mod_Dual_Crossfade': 26,
	'Combiners_Opaque_Mod2xNA_Alpha_Alpha': 27,
	'Combiners_Mod_Masked_Dual_Crossfade': 28,
	'Combiners_Opaque_Alpha': 29,
	'Guild': 30,
	'Guild_NoBorder': 31,
	'Guild_Opaque': 32,
	'Combiners_Mod_Depth': 33,
	'Illum': 34,
	'Combiners_Mod_Mod_Mod_Const': 35,
	'Combiners_Mod_Mod_Depth': 36
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

/**
 * compute model matrix from MDDF placement data.
 * MDDF uses WoW global coords [X, Y_up, Z]:
 *   position[0] = WoW X (horizontal), position[1] = WoW Y (altitude), position[2] = WoW Z (horizontal)
 * viewer (MCNK-derived) coords:
 *   viewer_x = MAP_COORD_BASE - WoW_X, viewer_y = WoW_Y, viewer_z = MAP_COORD_BASE - WoW_Z
 * rotation order: Ry * Rx * Rz (degrees), scale: uint16 (1024 = 1.0)
 */
function compute_model_matrix(position, rotation, scale) {
	const s = scale / 1024;
	const ax = rotation[0] * DEG_TO_RAD;
	const ay = (rotation[1] - 90) * DEG_TO_RAD;
	const az = rotation[2] * DEG_TO_RAD;

	const ca = Math.cos(ax), sa = Math.sin(ax);
	const cb = Math.cos(ay), sb = Math.sin(ay);
	const cc = Math.cos(az), sc = Math.sin(az);

	// R = Ry * Rx * Rz
	const r00 = cb * cc + sa * sb * sc;
	const r01 = -cb * sc + sa * sb * cc;
	const r02 = ca * sb;
	const r10 = ca * sc;
	const r11 = ca * cc;
	const r12 = -sa;
	const r20 = -sb * cc + sa * cb * sc;
	const r21 = sb * sc + sa * cb * cc;
	const r22 = ca * cb;

	const px = position[0];
	const py = position[1];
	const pz = position[2];

	// column-major; rows 0/2 negated for axis inversion (MAP_COORD_BASE - value)
	const mat = new Float32Array(16);
	mat[0]  = -s * r00;  mat[4]  = -s * r01;  mat[8]  = -s * r02;  mat[12] = MAP_COORD_BASE - px;
	mat[1]  =  s * r10;  mat[5]  =  s * r11;  mat[9]  =  s * r12;  mat[13] = py;
	mat[2]  = -s * r20;  mat[6]  = -s * r21;  mat[10] = -s * r22;  mat[14] = MAP_COORD_BASE - pz;
	mat[3]  = 0;          mat[7]  = 0;          mat[11] = 0;          mat[15] = 1;

	return mat;
}

/**
 * build interleaved vertex + index data from parsed M2 and Skin.
 * vertices are stored in raw wow local coords (un-swizzled from M2Loader output).
 * format: position(3f) + normal(3f) + bone_idx(4ub) + bone_weight(4ub) + uv(2f) + uv2(2f) = 48 bytes
 */
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

	// map triangle indices through skin indirection
	const index_data = new Uint16Array(skin.triangles.length);
	for (let i = 0; i < skin.triangles.length; i++)
		index_data[i] = skin.indices[skin.triangles[i]];

	return { vertex_data, index_data, index_count: index_data.length };
}

// ---- animation utilities ----

const IDENTITY_MAT4 = new Float32Array([
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1
]);

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

	return [
		lerp(v0[0], v1[0], alpha),
		lerp(v0[1], v1[1], alpha),
		lerp(v0[2], v1[2], alpha)
	];
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

// scratch matrices for bone computation (shared across all models)
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

/**
 * find animation index by animation ID.
 * tries primary ID, then fallback ID, then returns -1.
 */
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

	if (primary >= 0)
		return primary;

	return fallback;
}

/**
 * compute bone matrices for a model at a given animation time.
 * writes results into the provided bone_matrices Float32Array.
 */
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

		// resolve effective time for tracks with global sequences
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

/**
 * build draw call descriptors from skin texture units.
 * resolves shader IDs, blend modes, material flags, and texture fileDataIDs.
 */
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
			vertex_shader,
			pixel_shader,
			blend_mode,
			flags,
			tex_file_ids,
			tex_flags
		});
	}

	// sort: opaque/alpha_key first, then transparent by blend mode
	draw_calls.sort((a, b) => {
		const a_opaque = a.blend_mode <= 1 ? 0 : 1;
		const b_opaque = b.blend_mode <= 1 ? 0 : 1;
		if (a_opaque !== b_opaque)
			return a_opaque - b_opaque;

		return a.blend_mode - b.blend_mode;
	});

	return draw_calls;
}

class M2Renderer {
	constructor(gl_context) {
		this.ctx = gl_context;
		this.gl = gl_context.gl;
		this.shader = Shaders.create_program(gl_context, 'mpv_m2');
		this.bbox_shader = Shaders.create_program(gl_context, 'mpv_terrain_wire');
		this._casc = core.view.casc;

		this._model_cache = new Map();
		this._tile_data = new Map();
		this._texture_cache = new Map();

		this._obj0_load_queue = [];
		this._obj0_loading = new Set();
		this._model_load_queue = [];
		this._model_loading = new Set();
		this._upload_queue = [];

		this.enabled = true;
		this.render_distance = 500;

		this._last_cam = new Float32Array(3);
		this._instances_dirty = false;
		this._disposed = false;

		this._selected_id = 0;
		this._selected_instance = null;
		this._bbox_vao = null;

		this._default_texture = this._create_default_texture();

		// bind bone UBO block
		this.shader.bind_uniform_block('BoneMatrices', 0);
	}

	_create_default_texture() {
		const tex = new GLTexture(this.ctx);
		tex.set_rgba(new Uint8Array([255, 255, 255, 255]), 1, 1, { has_alpha: false });
		return tex;
	}

	get model_count() {
		let count = 0;
		for (const entry of this._model_cache.values()) {
			if (entry.vao)
				count++;
		}
		return count;
	}

	get instance_count() {
		let count = 0;
		for (const entry of this._model_cache.values())
			count += entry.instance_count;
		return count;
	}

	get loading_count() {
		return this._obj0_load_queue.length + this._obj0_loading.size
			+ this._model_load_queue.length + this._model_loading.size
			+ this._upload_queue.length;
	}

	on_tile_loaded(key, info) {
		if (!info.obj0_id || info.obj0_id <= 0)
			return;

		this._tile_data.set(key, {
			obj0_id: info.obj0_id,
			loaded: false,
			placements: null
		});

		if (this.enabled)
			this._obj0_load_queue.push(key);
	}

	on_tile_unloaded(key) {
		const data = this._tile_data.get(key);
		if (!data)
			return;

		this._obj0_loading.delete(key);

		if (data.placements) {
			const model_ids = new Set();
			for (const p of data.placements)
				model_ids.add(p.file_data_id);

			for (const id of model_ids) {
				const entry = this._model_cache.get(id);
				if (!entry)
					continue;

				entry.tile_instances.delete(key);
				entry.ref_count--;

				if (entry.ref_count <= 0)
					this._dispose_model(id);
			}

			this._instances_dirty = true;
		}

		this._tile_data.delete(key);
	}

	update(camera_pos, delta_time = 0) {
		if (this._disposed)
			return;

		this._process_uploads();

		if (this.enabled) {
			this._pump_obj0_queue();
			this._pump_model_queue();
		}

		const dx = camera_pos[0] - this._last_cam[0];
		const dy = camera_pos[1] - this._last_cam[1];
		const dz = camera_pos[2] - this._last_cam[2];

		if (dx * dx + dy * dy + dz * dz > CAMERA_DIRTY_THRESHOLD_SQ) {
			this._instances_dirty = true;
			this._last_cam.set(camera_pos);

			if (this.enabled)
				this._queue_in_range_models();
		}

		if (this._instances_dirty)
			this._rebuild_instance_buffers(camera_pos);

		// advance animations
		if (delta_time > 0)
			this._update_animations(delta_time);
	}

	render(view, proj, scene_params) {
		if (!this.enabled)
			return 0;

		const gl = this.gl;
		const ctx = this.ctx;
		const shader = this.shader;

		shader.use();
		shader.set_uniform_mat4('u_view', false, view);
		shader.set_uniform_mat4('u_projection', false, proj);
		shader.set_uniform_1f('u_alpha_test', ALPHA_TEST_VALUE);
		shader.set_uniform_4f('u_mesh_color', 1, 1, 1, 1);

		shader.set_uniform_1i('u_texture1', 0);
		shader.set_uniform_1i('u_texture2', 1);
		shader.set_uniform_1i('u_texture3', 2);
		shader.set_uniform_1i('u_texture4', 3);

		set_scene_uniforms(shader, scene_params);

		let drawn = 0;

		for (const entry of this._model_cache.values()) {
			if (!entry.vao || entry.instance_count === 0)
				continue;

			entry.vao.bind();

			// bind bone UBO for this model
			if (entry.bone_ubo) {
				entry.bone_ubo.bind(0);
				shader.set_uniform_1i('u_bone_count', entry.bone_count);
			} else {
				shader.set_uniform_1i('u_bone_count', 0);
			}

			for (const dc of entry.draw_calls) {
				shader.set_uniform_1i('u_vertex_shader', dc.vertex_shader);
				shader.set_uniform_1i('u_pixel_shader', dc.pixel_shader);
				shader.set_uniform_1i('u_blend_mode', dc.blend_mode);
				shader.set_uniform_1i('u_apply_lighting', (dc.flags & 0x1) ? 0 : 1);

				ctx.apply_blend_mode(dc.blend_mode);

				if (dc.flags & 0x04) {
					ctx.set_cull_face(false);
				} else {
					ctx.set_cull_face(true);
					ctx.set_cull_mode(gl.BACK);
				}

				if (dc.flags & 0x08)
					ctx.set_depth_test(false);
				else
					ctx.set_depth_test(true);

				if (dc.flags & 0x10)
					ctx.set_depth_write(false);
				else
					ctx.set_depth_write(true);

				for (let t = 0; t < 4; t++) {
					const file_id = dc.tex_file_ids[t];
					const cached = file_id > 0 ? this._texture_cache.get(file_id) : null;
					const tex = cached?.texture ?? this._default_texture;
					tex.bind(t);
				}

				gl.drawElementsInstanced(gl.TRIANGLES, dc.count, entry.vao.index_type, dc.start * 2, entry.instance_count);
			}

			drawn += entry.instance_count;
		}

		// reset state
		ctx.set_blend(false);
		ctx.set_depth_test(true);
		ctx.set_depth_write(true);
		ctx.set_cull_face(false);

		return drawn;
	}

	set_render_distance(value) {
		if (this.render_distance === value)
			return;

		this.render_distance = value;
		this._instances_dirty = true;

		if (this.enabled)
			this._queue_in_range_models();
	}

	set_enabled(val) {
		if (this.enabled === val)
			return;

		this.enabled = val;

		if (val) {
			// queue obj0 loading for tiles that arrived while disabled
			for (const [key, data] of this._tile_data) {
				if (!data.loaded && !this._obj0_loading.has(key))
					this._obj0_load_queue.push(key);
			}

			this._queue_in_range_models();
		}
	}

	_pump_obj0_queue() {
		while (this._obj0_loading.size < MAX_OBJ0_CONCURRENT && this._obj0_load_queue.length > 0) {
			const key = this._obj0_load_queue.shift();
			if (!this._tile_data.has(key) || this._obj0_loading.has(key))
				continue;

			this._start_obj0_load(key);
		}
	}

	async _start_obj0_load(key) {
		this._obj0_loading.add(key);
		const tile = this._tile_data.get(key);

		try {
			const obj0_file = await this._casc.getFile(tile.obj0_id);

			if (!this._tile_data.has(key) || this._disposed) {
				this._obj0_loading.delete(key);
				return;
			}

			const obj_adt = new ADTLoader(obj0_file);
			obj_adt.loadObj();

			if (!obj_adt.models || obj_adt.models.length === 0) {
				this._obj0_loading.delete(key);
				tile.loaded = true;
				return;
			}

			const using_names = !!obj_adt.m2Names;
			const placements = [];

			for (const mddf of obj_adt.models) {
				let file_data_id;

				if (using_names) {
					const name = obj_adt.m2Names[obj_adt.m2Offsets[mddf.mmidEntry]];
					if (!name)
						continue;

					file_data_id = listfile.getByFilename(name);
				} else {
					file_data_id = mddf.mmidEntry;
				}

				if (!file_data_id || file_data_id <= 0)
					continue;

				const matrix = compute_model_matrix(mddf.position, mddf.rotation, mddf.scale);

				const world_pos = new Float32Array([
					MAP_COORD_BASE - mddf.position[0],
					mddf.position[1],
					MAP_COORD_BASE - mddf.position[2]
				]);

				placements.push({ file_data_id, matrix, world_pos });
			}

			tile.placements = placements;
			tile.loaded = true;

			// group by model and register in cache
			const by_model = new Map();
			for (const p of placements) {
				let arr = by_model.get(p.file_data_id);
				if (!arr) {
					arr = [];
					by_model.set(p.file_data_id, arr);
				}
				arr.push(p);
			}

			for (const [id, instances] of by_model) {
				let entry = this._model_cache.get(id);

				if (!entry) {
					entry = {
						vao: null,
						index_count: 0,
						draw_calls: null,
						texture_ids: null,
						instance_buffer: null,
						instance_count: 0,
						ref_count: 0,
						tile_instances: new Map(),
						bounding_box: null,
						queued: false,
						bones: null,
						bone_count: 0,
						bone_ubo: null,
						bone_matrices: null,
						anim_index: -1,
						anim_duration: 0,
						anim_time: 0,
						global_loops: null,
						global_seq_times: null
					};
					this._model_cache.set(id, entry);
				}

				entry.ref_count++;
				entry.tile_instances.set(key, instances);

				if (!entry.vao && !entry.queued && !this._model_loading.has(id) && this._has_in_range_instances(entry)) {
					entry.queued = true;
					this._model_load_queue.push(id);
				}
			}

			this._instances_dirty = true;
		} catch (e) {
			log.write('Failed to load obj0 for tile ' + key + ': ' + e.message);
		}

		this._obj0_loading.delete(key);
	}

	_pump_model_queue() {
		while (this._model_loading.size < MAX_M2_CONCURRENT && this._model_load_queue.length > 0) {
			const id = this._model_load_queue.shift();
			const entry = this._model_cache.get(id);

			if (!entry || entry.vao || this._model_loading.has(id))
				continue;

			this._start_model_load(id);
		}
	}

	async _start_model_load(file_data_id) {
		this._model_loading.add(file_data_id);

		try {
			const file = await this._casc.getFile(file_data_id);

			if (this._disposed || !this._model_cache.has(file_data_id)) {
				this._model_loading.delete(file_data_id);
				return;
			}

			const m2 = new M2Loader(file);
			await m2.load();

			if (!m2.vertices || m2.vertices.length === 0) {
				this._model_loading.delete(file_data_id);
				return;
			}

			const skins = m2.skins;
			if (!skins || skins.length === 0) {
				this._model_loading.delete(file_data_id);
				return;
			}

			const skin = skins[0];
			if (!skin.isLoaded)
				await skin.load();

			if (!skin.isLoaded || this._disposed || !this._model_cache.has(file_data_id)) {
				this._model_loading.delete(file_data_id);
				return;
			}

			// load animation data for stand/idle
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

				anim_index = find_anim_index(m2.animations, ANIM_STAND, ANIM_CLOSED);

				if (anim_index >= 0) {
					// load external .anim file if needed
					await m2.loadAnimsForIndex(anim_index);

					const anim = m2.animations[anim_index];

					// resolve alias
					let resolved_idx = anim_index;
					let resolved_anim = anim;
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
			if (!geo) {
				this._model_loading.delete(file_data_id);
				return;
			}

			const draw_calls = build_draw_calls(m2, skin);

			// collect unique textures and fetch BLP data
			const texture_loads = new Map();
			for (const dc of draw_calls) {
				for (let j = 0; j < 4; j++) {
					const fid = dc.tex_file_ids[j];
					if (fid > 0 && !texture_loads.has(fid) && !this._texture_cache.has(fid))
						texture_loads.set(fid, { flags: dc.tex_flags[j], blp: null });
				}
			}

			// fetch and parse BLP files
			for (const [fid, entry] of texture_loads) {
				if (this._disposed || !this._model_cache.has(file_data_id))
					break;

				try {
					const data = await this._casc.getFile(fid);
					entry.blp = new BLPFile(data);
				} catch (e) {
					log.write('Failed to load M2 texture ' + fid + ': ' + e.message);
				}
			}

			if (this._disposed || !this._model_cache.has(file_data_id)) {
				this._model_loading.delete(file_data_id);
				return;
			}

			// swizzle bounding box from M2 coords [X, Y, Z] to viewer coords [X, Z, -Y]
			const raw_bb = m2.boundingBox ?? null;
			const bounding_box = raw_bb ? {
				min: [raw_bb.min[0], raw_bb.min[2], -raw_bb.max[1]],
				max: [raw_bb.max[0], raw_bb.max[2], -raw_bb.min[1]]
			} : null;

			this._upload_queue.push({
				file_data_id, draw_calls, texture_loads, bounding_box,
				bones, bone_count, anim_index, anim_duration,
				global_loops, global_seq_times,
				...geo
			});
		} catch (e) {
			log.write('Failed to load M2 model ' + file_data_id + ': ' + e.message);
		}

		this._model_loading.delete(file_data_id);
	}

	_process_uploads() {
		let budget = UPLOAD_BUDGET;

		while (budget > 0 && this._upload_queue.length > 0) {
			const item = this._upload_queue.shift();
			const entry = this._model_cache.get(item.file_data_id);

			if (!entry)
				continue;

			this._upload_model(entry, item);
			budget--;
			this._instances_dirty = true;
		}
	}

	_upload_model(entry, data) {
		const gl = this.gl;
		const vao = new VertexArray(this.ctx);
		vao.bind();

		// vertex buffer
		vao.set_vertex_buffer(data.vertex_data);

		// vertex format: pos(3f) + normal(3f) + bone_idx(4ub) + bone_weight(4ub) + uv(2f) + uv2(2f) = 48 bytes
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

		// index buffer
		vao.set_index_buffer(data.index_data);

		// instance buffer with model matrix attribs (locations 6-9)
		const instance_buffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, instance_buffer);
		gl.bufferData(gl.ARRAY_BUFFER, 64, gl.DYNAMIC_DRAW);

		for (let i = 0; i < 4; i++) {
			const loc = 6 + i;
			gl.enableVertexAttribArray(loc);
			gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 64, i * 16);
			gl.vertexAttribDivisor(loc, 1);
		}

		// upload textures to GPU
		const texture_ids = new Set();
		for (const [fid, tex_load] of data.texture_loads) {
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

		// ref-count textures already in cache (from other models)
		for (const dc of data.draw_calls) {
			for (let j = 0; j < 4; j++) {
				const fid = dc.tex_file_ids[j];
				if (fid > 0 && !texture_ids.has(fid)) {
					const cached = this._texture_cache.get(fid);
					if (cached) {
						cached.ref_count++;
						texture_ids.add(fid);
					}
				}
			}
		}

		entry.vao = vao;
		entry.index_count = data.index_count;
		entry.draw_calls = data.draw_calls;
		entry.texture_ids = texture_ids;
		entry.instance_buffer = instance_buffer;
		entry.bounding_box = data.bounding_box;
		entry.queued = false;

		// animation state
		entry.bones = data.bones;
		entry.bone_count = data.bone_count;
		entry.anim_index = data.anim_index;
		entry.anim_duration = data.anim_duration;
		entry.anim_time = 0;
		entry.global_loops = data.global_loops;
		entry.global_seq_times = data.global_seq_times;

		if (data.bone_count > 0) {
			const ubo_size = MAX_BONES * 64; // 256 mat4 * 64 bytes
			const ubo = new UniformBuffer(this.ctx, ubo_size);
			entry.bone_ubo = ubo;
			entry.bone_matrices = new Float32Array(data.bone_count * 16);

			// initialize with identity
			for (let i = 0; i < data.bone_count; i++)
				entry.bone_matrices.set(IDENTITY_MAT4, i * 16);

			// compute initial pose at time 0
			if (entry.anim_index >= 0)
				compute_bone_matrices(entry.bones, entry.anim_index, 0, entry.bone_matrices, entry.global_seq_times, entry.global_loops);

			ubo.set_mat4_array(0, entry.bone_matrices, data.bone_count);
			ubo.upload();
		} else {
			entry.bone_ubo = null;
			entry.bone_matrices = null;
		}
	}

	_update_animations(dt) {
		const dt_ms = dt * 1000;

		for (const entry of this._model_cache.values()) {
			if (!entry.bone_ubo || entry.anim_index < 0 || entry.instance_count === 0)
				continue;

			// advance time
			entry.anim_time += dt_ms;
			if (entry.anim_duration > 0)
				entry.anim_time %= entry.anim_duration;

			// advance global sequence timers
			if (entry.global_seq_times) {
				for (let i = 0; i < entry.global_seq_times.length; i++) {
					entry.global_seq_times[i] += dt_ms;
					const gs_dur = entry.global_loops[i];
					if (gs_dur > 0)
						entry.global_seq_times[i] %= gs_dur;
				}
			}

			// compute bone matrices
			compute_bone_matrices(
				entry.bones, entry.anim_index, entry.anim_time,
				entry.bone_matrices, entry.global_seq_times, entry.global_loops
			);

			// upload to GPU
			entry.bone_ubo.set_mat4_array(0, entry.bone_matrices, entry.bone_count);
			entry.bone_ubo.upload();
		}
	}

	_rebuild_instance_buffers(camera_pos) {
		this._instances_dirty = false;
		const gl = this.gl;
		const rd_sq = this.render_distance * this.render_distance;

		for (const entry of this._model_cache.values()) {
			if (!entry.vao)
				continue;

			const matrices = [];

			for (const instances of entry.tile_instances.values()) {
				for (const inst of instances) {
					const dx = inst.world_pos[0] - camera_pos[0];
					const dy = inst.world_pos[1] - camera_pos[1];
					const dz = inst.world_pos[2] - camera_pos[2];

					if (dx * dx + dy * dy + dz * dz <= rd_sq)
						matrices.push(inst.matrix);
				}
			}

			entry.instance_count = matrices.length;

			if (matrices.length === 0)
				continue;

			const buf = new Float32Array(matrices.length * 16);
			for (let i = 0; i < matrices.length; i++)
				buf.set(matrices[i], i * 16);

			gl.bindBuffer(gl.ARRAY_BUFFER, entry.instance_buffer);
			gl.bufferData(gl.ARRAY_BUFFER, buf, gl.DYNAMIC_DRAW);
		}
	}

	_has_in_range_instances(entry) {
		const rd_sq = this.render_distance * this.render_distance;
		const cam = this._last_cam;

		for (const instances of entry.tile_instances.values()) {
			for (const inst of instances) {
				const dx = inst.world_pos[0] - cam[0];
				const dy = inst.world_pos[1] - cam[1];
				const dz = inst.world_pos[2] - cam[2];

				if (dx * dx + dy * dy + dz * dz <= rd_sq)
					return true;
			}
		}

		return false;
	}

	_queue_in_range_models() {
		for (const [id, entry] of this._model_cache) {
			if (entry.vao || entry.queued || this._model_loading.has(id))
				continue;

			if (this._has_in_range_instances(entry)) {
				entry.queued = true;
				this._model_load_queue.push(id);
			}
		}
	}

	_release_textures(texture_ids) {
		if (!texture_ids)
			return;

		for (const fid of texture_ids) {
			const cached = this._texture_cache.get(fid);
			if (!cached)
				continue;

			cached.ref_count--;
			if (cached.ref_count <= 0) {
				cached.texture.dispose();
				this._texture_cache.delete(fid);
			}
		}
	}

	_dispose_model(file_data_id) {
		const entry = this._model_cache.get(file_data_id);
		if (!entry)
			return;

		if (this._selected_id === file_data_id)
			this.deselect();

		this._release_textures(entry.texture_ids);

		if (entry.vao)
			entry.vao.dispose();

		if (entry.instance_buffer)
			this.gl.deleteBuffer(entry.instance_buffer);

		if (entry.bone_ubo)
			entry.bone_ubo.dispose();

		this._model_cache.delete(file_data_id);
	}

	get selected_id() {
		return this._selected_id;
	}

	get selected_instance() {
		return this._selected_instance;
	}

	pick(ray_origin, ray_dir) {
		let best_t = Infinity;
		let best_id = 0;
		let best_inst = null;
		const rd_sq = this.render_distance * this.render_distance;

		for (const [id, entry] of this._model_cache) {
			if (!entry.vao || !entry.bounding_box)
				continue;

			const bb = entry.bounding_box;

			for (const instances of entry.tile_instances.values()) {
				for (const inst of instances) {
					const dx = inst.world_pos[0] - this._last_cam[0];
					const dy = inst.world_pos[1] - this._last_cam[1];
					const dz = inst.world_pos[2] - this._last_cam[2];

					if (dx * dx + dy * dy + dz * dz > rd_sq)
						continue;

					const t = this._ray_aabb_test(ray_origin, ray_dir, bb, inst.matrix);
					if (t >= 0 && t < best_t) {
						best_t = t;
						best_id = id;
						best_inst = inst;
					}
				}
			}
		}

		if (best_inst) {
			this._selected_id = best_id;
			this._selected_instance = best_inst;
			this._update_bbox_vao();
			return { file_data_id: best_id, instance: best_inst, t: best_t };
		}

		return null;
	}

	select(file_data_id, instance) {
		this._selected_id = file_data_id;
		this._selected_instance = instance;
		this._update_bbox_vao();
	}

	deselect() {
		this._selected_id = 0;
		this._selected_instance = null;
	}

	// ray-AABB intersection in model space via inverse transform
	_ray_aabb_test(ray_origin, ray_dir, bb, matrix) {
		// invert the 3x3 rotation+scale part and translation
		const m = matrix;
		const m00 = m[0], m01 = m[4], m02 = m[8],  tx = m[12];
		const m10 = m[1], m11 = m[5], m12 = m[9],  ty = m[13];
		const m20 = m[2], m21 = m[6], m22 = m[10], tz = m[14];

		// cofactor matrix for 3x3 inverse
		const c00 = m11 * m22 - m12 * m21;
		const c01 = m12 * m20 - m10 * m22;
		const c02 = m10 * m21 - m11 * m20;
		const c10 = m02 * m21 - m01 * m22;
		const c11 = m00 * m22 - m02 * m20;
		const c12 = m01 * m20 - m00 * m21;
		const c20 = m01 * m12 - m02 * m11;
		const c21 = m02 * m10 - m00 * m12;
		const c22 = m00 * m11 - m01 * m10;

		const det = m00 * c00 + m01 * c01 + m02 * c02;
		if (Math.abs(det) < 1e-12)
			return -1;

		const inv_det = 1.0 / det;

		// transform ray origin to local space
		const ox = ray_origin[0] - tx;
		const oy = ray_origin[1] - ty;
		const oz = ray_origin[2] - tz;

		const lo_x = (c00 * ox + c10 * oy + c20 * oz) * inv_det;
		const lo_y = (c01 * ox + c11 * oy + c21 * oz) * inv_det;
		const lo_z = (c02 * ox + c12 * oy + c22 * oz) * inv_det;

		const ld_x = (c00 * ray_dir[0] + c10 * ray_dir[1] + c20 * ray_dir[2]) * inv_det;
		const ld_y = (c01 * ray_dir[0] + c11 * ray_dir[1] + c21 * ray_dir[2]) * inv_det;
		const ld_z = (c02 * ray_dir[0] + c12 * ray_dir[1] + c22 * ray_dir[2]) * inv_det;

		// slab method
		const min = bb.min, max = bb.max;
		let tmin = -Infinity, tmax = Infinity;

		if (Math.abs(ld_x) > 1e-12) {
			const t1 = (min[0] - lo_x) / ld_x;
			const t2 = (max[0] - lo_x) / ld_x;
			tmin = Math.max(tmin, Math.min(t1, t2));
			tmax = Math.min(tmax, Math.max(t1, t2));
		} else if (lo_x < min[0] || lo_x > max[0]) {
			return -1;
		}

		if (Math.abs(ld_y) > 1e-12) {
			const t1 = (min[1] - lo_y) / ld_y;
			const t2 = (max[1] - lo_y) / ld_y;
			tmin = Math.max(tmin, Math.min(t1, t2));
			tmax = Math.min(tmax, Math.max(t1, t2));
		} else if (lo_y < min[1] || lo_y > max[1]) {
			return -1;
		}

		if (Math.abs(ld_z) > 1e-12) {
			const t1 = (min[2] - lo_z) / ld_z;
			const t2 = (max[2] - lo_z) / ld_z;
			tmin = Math.max(tmin, Math.min(t1, t2));
			tmax = Math.min(tmax, Math.max(t1, t2));
		} else if (lo_z < min[2] || lo_z > max[2]) {
			return -1;
		}

		if (tmax < 0 || tmin > tmax)
			return -1;

		return tmin >= 0 ? tmin : tmax;
	}

	_update_bbox_vao() {
		const entry = this._model_cache.get(this._selected_id);
		if (!entry?.bounding_box || !this._selected_instance)
			return;

		const bb = entry.bounding_box;
		const m = this._selected_instance.matrix;
		const min = bb.min, max = bb.max;

		// 8 corners of AABB in local space
		const corners = [
			[min[0], min[1], min[2]],
			[max[0], min[1], min[2]],
			[max[0], max[1], min[2]],
			[min[0], max[1], min[2]],
			[min[0], min[1], max[2]],
			[max[0], min[1], max[2]],
			[max[0], max[1], max[2]],
			[min[0], max[1], max[2]]
		];

		// transform corners to world space
		const world = corners.map(c => {
			const x = m[0] * c[0] + m[4] * c[1] + m[8]  * c[2] + m[12];
			const y = m[1] * c[0] + m[5] * c[1] + m[9]  * c[2] + m[13];
			const z = m[2] * c[0] + m[6] * c[1] + m[10] * c[2] + m[14];
			return [x, y, z];
		});

		// 12 edges of a box: pairs of corner indices
		const edges = [
			0, 1, 1, 2, 2, 3, 3, 0,
			4, 5, 5, 6, 6, 7, 7, 4,
			0, 4, 1, 5, 2, 6, 3, 7
		];

		const positions = new Float32Array(edges.length * 3);
		for (let i = 0; i < edges.length; i++) {
			const c = world[edges[i]];
			positions[i * 3] = c[0];
			positions[i * 3 + 1] = c[1];
			positions[i * 3 + 2] = c[2];
		}

		const gl = this.gl;

		if (!this._bbox_vao) {
			this._bbox_vao = new VertexArray(this.ctx);
			this._bbox_vao.bind();
			this._bbox_vao.vbo = gl.createBuffer();
			gl.bindBuffer(gl.ARRAY_BUFFER, this._bbox_vao.vbo);
			gl.enableVertexAttribArray(0);
			gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
		} else {
			this._bbox_vao.bind();
			gl.bindBuffer(gl.ARRAY_BUFFER, this._bbox_vao.vbo);
		}

		gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
	}

	render_selection(view_matrix, projection_matrix) {
		if (!this._selected_id || !this._selected_instance || !this._bbox_vao)
			return;

		if (!this.bbox_shader.is_valid())
			return;

		const gl = this.gl;

		this.bbox_shader.use();
		this.bbox_shader.set_uniform_mat4('u_view', false, view_matrix);
		this.bbox_shader.set_uniform_mat4('u_projection', false, projection_matrix);
		this.bbox_shader.set_uniform_3fv('u_terrain_color', BBOX_COLOR);

		this.ctx.set_depth_test(true);
		this.ctx.set_depth_write(false);

		this._bbox_vao.bind();
		gl.drawArrays(gl.LINES, 0, 24);

		this.ctx.set_depth_write(true);
	}

	dispose() {
		this._disposed = true;

		this._obj0_load_queue.length = 0;
		this._obj0_loading.clear();
		this._model_load_queue.length = 0;
		this._model_loading.clear();
		this._upload_queue.length = 0;

		for (const id of [...this._model_cache.keys()])
			this._dispose_model(id);

		this._model_cache.clear();
		this._tile_data.clear();

		// dispose remaining cached textures
		for (const cached of this._texture_cache.values())
			cached.texture.dispose();

		this._texture_cache.clear();

		if (this._default_texture) {
			this._default_texture.dispose();
			this._default_texture = null;
		}

		if (this._bbox_vao) {
			this._bbox_vao.dispose();
			this._bbox_vao = null;
		}

		if (this.shader) {
			Shaders.unregister(this.shader);
			this.shader.dispose();
			this.shader = null;
		}

		if (this.bbox_shader) {
			Shaders.unregister(this.bbox_shader);
			this.bbox_shader.dispose();
			this.bbox_shader = null;
		}
	}
}

module.exports = M2Renderer;

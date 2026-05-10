/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const core = require('../core');
const constants = require('../constants');
const Shaders = require('../3D/Shaders');
const VertexArray = require('../3D/gl/VertexArray');
const GLTexture = require('../3D/gl/GLTexture');
const BLPFile = require('../casc/blp');
const WMOLoader = require('../3D/loaders/WMOLoader');
const WMOShaderMapper = require('../3D/WMOShaderMapper');
const ADTLoader = require('../3D/loaders/ADTLoader');
const listfile = require('../casc/listfile');
const log = require('../log');

const DEG_TO_RAD = Math.PI / 180;
const MAP_COORD_BASE = constants.GAME.MAP_COORD_BASE;
const MAX_OBJ0_CONCURRENT = 2;
const MAX_WMO_CONCURRENT = 1;
const UPLOAD_BUDGET_GROUPS = 2;
const CAMERA_DIRTY_THRESHOLD_SQ = 33.33 * 33.33;

// pos(3f) + normal(3f) + uv1(2f) + uv2(2f) + uv3(2f) + color1(4ub) + color2(4ub) = 56
const WMO_VERTEX_STRIDE = 56;

const BBOX_COLOR = new Float32Array([1, 0.8, 0]);

/**
 * compute model matrix for MODF placement with pre-swizzled WMO vertices.
 * WMOLoader swizzles [x,y,z] → [x, z, -y] during parsing.
 * matrix is diag(-s, s, -s) * R (same as M2 doodads).
 */
function compute_wmo_model_matrix(position, rotation, scale) {
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

	// column-major: rows 0/2 negated for axis inversion
	const mat = new Float32Array(16);
	mat[0]  = -s * r00;  mat[4]  = -s * r01;  mat[8]  = -s * r02;  mat[12] = MAP_COORD_BASE - px;
	mat[1]  =  s * r10;  mat[5]  =  s * r11;  mat[9]  =  s * r12;  mat[13] = py;
	mat[2]  = -s * r20;  mat[6]  = -s * r21;  mat[10] = -s * r22;  mat[14] = MAP_COORD_BASE - pz;
	mat[3]  = 0;          mat[7]  = 0;          mat[11] = 0;          mat[15] = 1;

	return mat;
}

/**
 * build interleaved vertex data from a WMO group.
 * format: pos(3f) + normal(3f) + uv1(2f) + uv2(2f) + uv3(2f) + color1(4ub) + color2(4ub) = 56 bytes
 */
function build_group_geometry(group) {
	if (!group.vertices || !group.normals || !group.indices)
		return null;

	const vert_count = group.vertices.length / 3;
	if (vert_count === 0)
		return null;

	const vertex_data = new ArrayBuffer(vert_count * WMO_VERTEX_STRIDE);
	const view = new DataView(vertex_data);

	const uvs1 = group.uvs?.[0];
	const uvs2 = group.uvs?.[1];
	const uvs3 = group.uvs?.[2];
	const colors1 = group.vertexColours?.[0];
	const colors2 = group.vertexColours?.[1];

	for (let i = 0; i < vert_count; i++) {
		const offset = i * WMO_VERTEX_STRIDE;
		const v = i * 3;
		const uv = i * 2;
		const c = i * 4;

		// position
		view.setFloat32(offset, group.vertices[v], true);
		view.setFloat32(offset + 4, group.vertices[v + 1], true);
		view.setFloat32(offset + 8, group.vertices[v + 2], true);

		// normal
		view.setFloat32(offset + 12, group.normals[v], true);
		view.setFloat32(offset + 16, group.normals[v + 1], true);
		view.setFloat32(offset + 20, group.normals[v + 2], true);

		// uv1
		if (uvs1) {
			view.setFloat32(offset + 24, uvs1[uv], true);
			view.setFloat32(offset + 28, uvs1[uv + 1], true);
		}

		// uv2
		if (uvs2) {
			view.setFloat32(offset + 32, uvs2[uv], true);
			view.setFloat32(offset + 36, uvs2[uv + 1], true);
		}

		// uv3
		if (uvs3) {
			view.setFloat32(offset + 40, uvs3[uv], true);
			view.setFloat32(offset + 44, uvs3[uv + 1], true);
		}

		// color1 (BGRA in file → RGBA swizzle)
		if (colors1) {
			view.setUint8(offset + 48, colors1[c + 2]);
			view.setUint8(offset + 49, colors1[c + 1]);
			view.setUint8(offset + 50, colors1[c]);
			view.setUint8(offset + 51, colors1[c + 3]);
		} else {
			view.setUint32(offset + 48, 0xFFFFFFFF);
		}

		// color2 (BGRA → RGBA)
		if (colors2) {
			view.setUint8(offset + 52, colors2[c + 2]);
			view.setUint8(offset + 53, colors2[c + 1]);
			view.setUint8(offset + 54, colors2[c]);
			view.setUint8(offset + 55, colors2[c + 3]);
		} else {
			view.setUint32(offset + 52, 0xFFFFFFFF);
		}
	}

	const index_data = new Uint16Array(group.indices);

	return { vertex_data, index_data, index_count: index_data.length };
}

/**
 * build draw call descriptors from a group's render batches.
 */
function build_group_draw_calls(group, materials, material_tex_ids) {
	if (!group.renderBatches || group.renderBatches.length === 0)
		return [];

	const draw_calls = [];

	for (const batch of group.renderBatches) {
		const mat_id = ((batch.flags & 2) === 2) ? batch.possibleBox2[2] : batch.materialID;
		const material = materials?.[mat_id];

		if (!material) {
			draw_calls.push({
				start: batch.firstFace,
				count: batch.numFaces,
				blend_mode: 0,
				vertex_shader: 0,
				pixel_shader: 0,
				flags: 0,
				tex_ids: [0, 0, 0]
			});
			continue;
		}

		const shader_info = WMOShaderMapper.WMOShaderMap[material.shader] ?? { VertexShader: 0, PixelShader: 0 };
		const tex_ids = material_tex_ids.get(mat_id) ?? [0, 0, 0];

		draw_calls.push({
			start: batch.firstFace,
			count: batch.numFaces,
			blend_mode: material.blendMode,
			vertex_shader: shader_info.VertexShader,
			pixel_shader: shader_info.PixelShader,
			flags: material.flags,
			tex_ids
		});
	}

	return draw_calls;
}

/**
 * collect texture fileDataIDs from WMO materials.
 * returns material_tex_ids map and unique_textures map with wrap info.
 */
function collect_wmo_textures(wmo) {
	const material_tex_ids = new Map();
	const unique_textures = new Map();
	const is_classic = !!wmo.textureNames;

	if (!wmo.materials)
		return { material_tex_ids, unique_textures };

	for (let i = 0; i < wmo.materials.length; i++) {
		const mat = wmo.materials[i];
		const shader_info = WMOShaderMapper.WMOShaderMap[mat.shader];

		// skip LOD materials
		if (shader_info && shader_info.PixelShader === 18)
			continue;

		let tex_ids;
		if (is_classic) {
			tex_ids = [
				listfile.getByFilename(wmo.textureNames[mat.texture1]) || 0,
				listfile.getByFilename(wmo.textureNames[mat.texture2]) || 0,
				listfile.getByFilename(wmo.textureNames[mat.texture3]) || 0
			];
		} else {
			tex_ids = [mat.texture1, mat.texture2, mat.texture3];
		}

		material_tex_ids.set(i, tex_ids);

		// WMO wrap flags are inverted: 0x40=clamp_s, 0x80=clamp_t
		const wrap_s = !(mat.flags & 0x40);
		const wrap_t = !(mat.flags & 0x80);

		for (const fid of tex_ids) {
			if (fid > 0 && !unique_textures.has(fid))
				unique_textures.set(fid, { wrap_s, wrap_t });
		}
	}

	return { material_tex_ids, unique_textures };
}

class WMORenderer {
	constructor(gl_context) {
		this.ctx = gl_context;
		this.gl = gl_context.gl;
		this.shader = Shaders.create_program(gl_context, 'mpv_wmo');
		this.bbox_shader = Shaders.create_program(gl_context, 'mpv_terrain_wire');
		this._casc = core.view.casc;

		this._model_cache = new Map();
		this._tile_data = new Map();
		this._texture_cache = new Map();

		this._obj0_load_queue = [];
		this._obj0_loading = new Set();
		this._wmo_load_queue = [];
		this._wmo_loading = new Set();
		this._upload_queue = [];

		this.enabled = true;
		this.render_distance = 5000;

		this._last_cam = new Float32Array(3);
		this._culled_dirty = false;
		this._disposed = false;

		this._selected_id = 0;
		this._selected_instance = null;
		this._bbox_vao = null;

		this._default_texture = this._create_default_texture();
	}

	_create_default_texture() {
		const tex = new GLTexture(this.ctx);
		tex.set_rgba(new Uint8Array([255, 255, 255, 255]), 1, 1, { has_alpha: false });
		return tex;
	}

	get model_count() {
		let count = 0;
		for (const entry of this._model_cache.values()) {
			if (entry.groups)
				count++;
		}
		return count;
	}

	get instance_count() {
		let count = 0;
		for (const entry of this._model_cache.values())
			count += entry.culled_count;
		return count;
	}

	get loading_count() {
		return this._obj0_load_queue.length + this._obj0_loading.size
			+ this._wmo_load_queue.length + this._wmo_loading.size
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

				entry.tile_placements.delete(key);
				entry.ref_count--;

				if (entry.ref_count <= 0)
					this._dispose_model(id);
			}

			this._culled_dirty = true;
		}

		this._tile_data.delete(key);
	}

	update(camera_pos) {
		if (this._disposed)
			return;

		this._process_uploads();

		if (this.enabled) {
			this._pump_obj0_queue();
			this._pump_wmo_queue();
		}

		const dx = camera_pos[0] - this._last_cam[0];
		const dy = camera_pos[1] - this._last_cam[1];
		const dz = camera_pos[2] - this._last_cam[2];

		if (dx * dx + dy * dy + dz * dz > CAMERA_DIRTY_THRESHOLD_SQ) {
			this._culled_dirty = true;
			this._last_cam.set(camera_pos);

			if (this.enabled)
				this._queue_in_range_models();
		}

		if (this._culled_dirty)
			this._rebuild_culled_lists(camera_pos);
	}

	render(view, proj, light_dir, sun_color, sun_intensity, fog_params) {
		if (!this.enabled)
			return 0;

		const gl = this.gl;
		const ctx = this.ctx;
		const shader = this.shader;

		shader.use();
		shader.set_uniform_mat4('u_view', false, view);
		shader.set_uniform_mat4('u_projection', false, proj);
		shader.set_uniform_3fv('u_light_dir', light_dir);
		shader.set_uniform_3fv('u_sun_color', sun_color);
		shader.set_uniform_1f('u_sun_intensity', sun_intensity);

		shader.set_uniform_1i('u_texture1', 0);
		shader.set_uniform_1i('u_texture2', 1);
		shader.set_uniform_1i('u_texture3', 2);

		if (fog_params && fog_params.fog_uniforms) {
			const fog = fog_params.fog_uniforms;
			shader.set_uniform_3fv('u_camera_pos', fog_params.camera_pos);
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
			shader.set_uniform_3fv('u_camera_pos', fog_params?.camera_pos ?? new Float32Array(3));
			shader.set_uniform_1f('u_fog_enabled', 0.0);
		}

		ctx.set_depth_test(true);
		ctx.set_depth_write(true);
		ctx.set_cull_face(false);
		ctx.set_blend(false);

		let drawn = 0;

		for (const entry of this._model_cache.values()) {
			if (!entry.groups || entry.culled_count === 0)
				continue;

			for (const inst of entry.culled_instances) {
				shader.set_uniform_mat4('u_model', false, inst.matrix);

				for (const group of entry.groups) {
					group.vao.bind();

					for (const dc of group.draw_calls) {
						shader.set_uniform_1i('u_vertex_shader', dc.vertex_shader);
						shader.set_uniform_1i('u_pixel_shader', dc.pixel_shader);
						shader.set_uniform_1i('u_blend_mode', dc.blend_mode);
						shader.set_uniform_1i('u_apply_lighting', (dc.flags & 0x01) ? 0 : 1);

						ctx.apply_blend_mode(dc.blend_mode);

						for (let t = 0; t < 3; t++) {
							const fid = dc.tex_ids[t];
							const cached = fid > 0 ? this._texture_cache.get(fid) : null;
							const tex = cached?.texture ?? this._default_texture;
							tex.bind(t);
						}

						gl.drawElements(gl.TRIANGLES, dc.count, gl.UNSIGNED_SHORT, dc.start * 2);
					}
				}

				drawn++;
			}
		}

		// reset state after transparent batches
		ctx.set_blend(false);
		ctx.set_depth_write(true);
		ctx.set_cull_face(false);

		return drawn;
	}

	set_render_distance(value) {
		if (this.render_distance === value)
			return;

		this.render_distance = value;
		this._culled_dirty = true;

		if (this.enabled)
			this._queue_in_range_models();
	}

	set_enabled(val) {
		if (this.enabled === val)
			return;

		this.enabled = val;

		if (val) {
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

			if (!obj_adt.worldModels || obj_adt.worldModels.length === 0) {
				this._obj0_loading.delete(key);
				tile.loaded = true;
				return;
			}

			const using_names = !!obj_adt.wmoNames;
			const placements = [];

			for (const modf of obj_adt.worldModels) {
				let file_data_id;

				if (using_names) {
					const name = obj_adt.wmoNames[obj_adt.wmoOffsets[modf.mwidEntry]];
					if (!name)
						continue;

					file_data_id = listfile.getByFilename(name);
				} else {
					file_data_id = modf.mwidEntry;
				}

				if (!file_data_id || file_data_id <= 0)
					continue;

				const matrix = compute_wmo_model_matrix(modf.position, modf.rotation, modf.scale);

				const world_pos = new Float32Array([
					MAP_COORD_BASE - modf.position[0],
					modf.position[1],
					MAP_COORD_BASE - modf.position[2]
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
						groups: null,
						bounding_box: null,
						texture_ids: null,
						ref_count: 0,
						tile_placements: new Map(),
						culled_instances: [],
						culled_count: 0,
						queued: false
					};
					this._model_cache.set(id, entry);
				}

				entry.ref_count++;
				entry.tile_placements.set(key, instances);

				if (!entry.groups && !entry.queued && !this._wmo_loading.has(id) && this._has_in_range_instances(entry)) {
					entry.queued = true;
					this._wmo_load_queue.push(id);
				}
			}

			this._culled_dirty = true;
		} catch (e) {
			log.write('Failed to load obj0 (WMO) for tile ' + key + ': ' + e.message);
		}

		this._obj0_loading.delete(key);
	}

	_pump_wmo_queue() {
		while (this._wmo_loading.size < MAX_WMO_CONCURRENT && this._wmo_load_queue.length > 0) {
			const id = this._wmo_load_queue.shift();
			const entry = this._model_cache.get(id);

			if (!entry || entry.groups || this._wmo_loading.has(id))
				continue;

			this._start_wmo_load(id);
		}
	}

	async _start_wmo_load(file_data_id) {
		this._wmo_loading.add(file_data_id);

		try {
			const file = await this._casc.getFile(file_data_id);

			if (this._disposed || !this._model_cache.has(file_data_id)) {
				this._wmo_loading.delete(file_data_id);
				return;
			}

			const wmo = new WMOLoader(file, file_data_id, true);
			await wmo.load();

			if (this._disposed || !this._model_cache.has(file_data_id)) {
				this._wmo_loading.delete(file_data_id);
				return;
			}

			// collect material texture references
			const { material_tex_ids, unique_textures } = collect_wmo_textures(wmo);

			// load groups, build geometry + draw calls
			const group_data = [];
			for (let i = 0; i < wmo.groupCount; i++) {
				if (this._disposed || !this._model_cache.has(file_data_id))
					break;

				try {
					const group = await wmo.getGroup(i);
					const geo = build_group_geometry(group);
					if (!geo)
						continue;

					const draw_calls = build_group_draw_calls(group, wmo.materials, material_tex_ids);
					if (draw_calls.length === 0)
						continue;

					group_data.push({ ...geo, draw_calls });
				} catch (e) {
					log.write('Failed to load WMO group ' + i + ' for ' + file_data_id + ': ' + e.message);
				}
			}

			if (this._disposed || !this._model_cache.has(file_data_id)) {
				this._wmo_loading.delete(file_data_id);
				return;
			}

			// fetch BLP files for uncached textures
			const texture_loads = new Map();
			for (const [fid, wrap_info] of unique_textures) {
				if (this._disposed || !this._model_cache.has(file_data_id))
					break;

				if (this._texture_cache.has(fid))
					continue;

				try {
					const data = await this._casc.getFile(fid);
					texture_loads.set(fid, { blp: new BLPFile(data), wrap_s: wrap_info.wrap_s, wrap_t: wrap_info.wrap_t });
				} catch (e) {
					log.write('Failed to load WMO texture ' + fid + ': ' + e.message);
				}
			}

			if (this._disposed || !this._model_cache.has(file_data_id)) {
				this._wmo_loading.delete(file_data_id);
				return;
			}

			// swizzle bounding box from WMO coords [X, Z, -Y] (already swizzled in loader)
			const bb1 = wmo.boundingBox1;
			const bb2 = wmo.boundingBox2;
			const bounding_box = (bb1 && bb2) ? {
				min: [bb1[0], bb1[2], -bb2[1]],
				max: [bb2[0], bb2[2], -bb1[1]]
			} : null;

			if (group_data.length > 0)
				this._upload_queue.push({ file_data_id, group_data, texture_loads, bounding_box });
		} catch (e) {
			log.write('Failed to load WMO model ' + file_data_id + ': ' + e.message);
		}

		this._wmo_loading.delete(file_data_id);
	}

	_process_uploads() {
		let budget = UPLOAD_BUDGET_GROUPS;

		while (budget > 0 && this._upload_queue.length > 0) {
			const item = this._upload_queue[0];
			const entry = this._model_cache.get(item.file_data_id);

			if (!entry) {
				this._upload_queue.shift();
				continue;
			}

			// upload textures + init groups on first access
			if (!entry.groups) {
				entry.groups = [];
				this._upload_textures(entry, item);
			}

			// upload one group at a time to spread GPU work
			const geo = item.group_data.shift();
			if (geo) {
				this._upload_group(entry, geo);
				budget--;
			}

			if (item.group_data.length === 0) {
				entry.bounding_box = item.bounding_box;
				entry.queued = false;
				this._culled_dirty = true;
				this._upload_queue.shift();
			}
		}
	}

	_upload_textures(entry, data) {
		// upload new BLPs to GPU
		for (const [fid, tex_load] of data.texture_loads) {
			if (!this._texture_cache.has(fid)) {
				const gl_tex = new GLTexture(this.ctx);
				if (tex_load.blp)
					gl_tex.set_blp(tex_load.blp, { wrap_s: tex_load.wrap_s, wrap_t: tex_load.wrap_t });

				this._texture_cache.set(fid, { texture: gl_tex, ref_count: 0 });
			}
		}

		// ref-count all unique textures used by this model's draw calls
		const model_tex_ids = new Set();
		for (const group of data.group_data) {
			for (const dc of group.draw_calls) {
				for (const fid of dc.tex_ids) {
					if (fid > 0 && !model_tex_ids.has(fid)) {
						model_tex_ids.add(fid);
						const cached = this._texture_cache.get(fid);
						if (cached)
							cached.ref_count++;
					}
				}
			}
		}

		entry.texture_ids = model_tex_ids;
	}

	_upload_group(entry, geo) {
		const gl = this.gl;
		const vao = new VertexArray(this.ctx);
		vao.bind();

		vao.set_vertex_buffer(geo.vertex_data);

		const stride = WMO_VERTEX_STRIDE;

		// pos(3f) @ 0
		gl.enableVertexAttribArray(0);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);

		// normal(3f) @ 12
		gl.enableVertexAttribArray(1);
		gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 12);

		// uv1(2f) @ 24
		gl.enableVertexAttribArray(4);
		gl.vertexAttribPointer(4, 2, gl.FLOAT, false, stride, 24);

		// uv2(2f) @ 32
		gl.enableVertexAttribArray(5);
		gl.vertexAttribPointer(5, 2, gl.FLOAT, false, stride, 32);

		// uv3(2f) @ 40
		gl.enableVertexAttribArray(8);
		gl.vertexAttribPointer(8, 2, gl.FLOAT, false, stride, 40);

		// color1(4ub) @ 48
		gl.enableVertexAttribArray(6);
		gl.vertexAttribPointer(6, 4, gl.UNSIGNED_BYTE, true, stride, 48);

		// color2(4ub) @ 52
		gl.enableVertexAttribArray(7);
		gl.vertexAttribPointer(7, 4, gl.UNSIGNED_BYTE, true, stride, 52);

		vao.set_index_buffer(geo.index_data);

		entry.groups.push({ vao, draw_calls: geo.draw_calls });
	}

	_rebuild_culled_lists(camera_pos) {
		this._culled_dirty = false;
		const rd_sq = this.render_distance * this.render_distance;

		for (const entry of this._model_cache.values()) {
			if (!entry.groups)
				continue;

			const culled = [];

			for (const instances of entry.tile_placements.values()) {
				for (const inst of instances) {
					const dx = inst.world_pos[0] - camera_pos[0];
					const dy = inst.world_pos[1] - camera_pos[1];
					const dz = inst.world_pos[2] - camera_pos[2];

					if (dx * dx + dy * dy + dz * dz <= rd_sq)
						culled.push(inst);
				}
			}

			entry.culled_instances = culled;
			entry.culled_count = culled.length;
		}
	}

	_has_in_range_instances(entry) {
		const rd_sq = this.render_distance * this.render_distance;
		const cam = this._last_cam;

		for (const instances of entry.tile_placements.values()) {
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
			if (entry.groups || entry.queued || this._wmo_loading.has(id))
				continue;

			if (this._has_in_range_instances(entry)) {
				entry.queued = true;
				this._wmo_load_queue.push(id);
			}
		}
	}

	_dispose_model(file_data_id) {
		const entry = this._model_cache.get(file_data_id);
		if (!entry)
			return;

		if (this._selected_id === file_data_id)
			this.deselect();

		if (entry.groups) {
			for (const group of entry.groups)
				group.vao.dispose();
		}

		// release ref-counted textures
		if (entry.texture_ids) {
			for (const fid of entry.texture_ids) {
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
			if (!entry.groups || !entry.bounding_box)
				continue;

			const bb = entry.bounding_box;

			for (const instances of entry.tile_placements.values()) {
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
		const m = matrix;
		const m00 = m[0], m01 = m[4], m02 = m[8],  tx = m[12];
		const m10 = m[1], m11 = m[5], m12 = m[9],  ty = m[13];
		const m20 = m[2], m21 = m[6], m22 = m[10], tz = m[14];

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

		const ox = ray_origin[0] - tx;
		const oy = ray_origin[1] - ty;
		const oz = ray_origin[2] - tz;

		const lo_x = (c00 * ox + c10 * oy + c20 * oz) * inv_det;
		const lo_y = (c01 * ox + c11 * oy + c21 * oz) * inv_det;
		const lo_z = (c02 * ox + c12 * oy + c22 * oz) * inv_det;

		const ld_x = (c00 * ray_dir[0] + c10 * ray_dir[1] + c20 * ray_dir[2]) * inv_det;
		const ld_y = (c01 * ray_dir[0] + c11 * ray_dir[1] + c21 * ray_dir[2]) * inv_det;
		const ld_z = (c02 * ray_dir[0] + c12 * ray_dir[1] + c22 * ray_dir[2]) * inv_det;

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

		const world = corners.map(c => {
			const x = m[0] * c[0] + m[4] * c[1] + m[8]  * c[2] + m[12];
			const y = m[1] * c[0] + m[5] * c[1] + m[9]  * c[2] + m[13];
			const z = m[2] * c[0] + m[6] * c[1] + m[10] * c[2] + m[14];
			return [x, y, z];
		});

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
		this._wmo_load_queue.length = 0;
		this._wmo_loading.clear();
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

module.exports = WMORenderer;

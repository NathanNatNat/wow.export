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
const WDCReader = require('../db/WDCReader');
const listfile = require('../casc/listfile');
const log = require('../log');

const TILE_SIZE = constants.GAME.TILE_SIZE;
const UNIT_SIZE = (TILE_SIZE / 16) / 8;
const LIQUID_VERTEX_STRIDE = 24; // pos(3f) + uv(2f) + depth(1f)
const MAX_TEXTURE_LOADS = 2;

const LIQUID_COLORS = {
	water: new Float32Array([0.0, 0.2, 0.5]),
	ocean: new Float32Array([0.0, 0.1, 0.4]),
	magma: new Float32Array([0.6, 0.15, 0.0]),
	slime: new Float32Array([0.2, 0.5, 0.1])
};

const MATERIAL_MAGMA = [2, 4];

function liquid_category(liquid_type) {
	if (liquid_type === 2 || liquid_type === 14 || liquid_type === 15)
		return 'ocean';

	if (liquid_type === 3 || liquid_type === 4 || liquid_type === 7 || liquid_type === 8 || liquid_type === 19)
		return 'magma';

	if (liquid_type === 5 || liquid_type === 6 || liquid_type === 9 || liquid_type === 17 || liquid_type === 20 || liquid_type === 21)
		return 'slime';

	return 'water';
}

class LiquidRenderer {
	constructor(gl_context) {
		this.ctx = gl_context;
		this.gl = gl_context.gl;
		this.shader = Shaders.create_program(gl_context, 'mpv_liquid');
		this._casc = core.view.casc;

		this._tile_data = new Map();
		this._texture_cache = new Map();
		this._texture_load_queue = [];
		this._texture_loading = new Set();

		this._db_loaded = false;
		this._db_loading = false;
		this._liquid_type_cache = new Map();

		this._default_texture = this._create_default_texture();

		this._frustum_planes = new Float32Array(24);
		this._vp_matrix = new Float32Array(16);

		this.enabled = true;
		this._disposed = false;
		this._start_time = performance.now();
	}

	_create_default_texture() {
		const tex = new GLTexture(this.ctx);
		tex.set_rgba(new Uint8Array([0, 0, 0, 0]), 1, 1, { has_alpha: true });
		return tex;
	}

	get instance_count() {
		let count = 0;
		for (const tile of this._tile_data.values())
			count += tile.instances.length;
		return count;
	}

	get loading_count() {
		return this._texture_load_queue.length + this._texture_loading.size;
	}

	async _ensure_db() {
		if (this._db_loaded || this._db_loading)
			return;

		this._db_loading = true;
		try {
			// load DB2 tables directly so getRow() is synchronous
			const lt = new WDCReader('DBFilesClient/LiquidType.db2');
			await lt.parse();
			lt.preload();
			this._liquid_type_db = lt;

			try {
				const lo = new WDCReader('DBFilesClient/LiquidObject.db2');
				await lo.parse();
				lo.preload();
				this._liquid_object_db = lo;
			} catch {
				// LiquidObject may not exist in older clients
			}

			try {
				const lm = new WDCReader('DBFilesClient/LiquidMaterial.db2');
				await lm.parse();
				lm.preload();
				this._liquid_material_db = lm;
			} catch {
				// LiquidMaterial may not exist in older clients
			}

			// LiquidTypeXTexture maps LiquidTypeID → FileDataID
			this._liquid_type_textures = new Map();
			try {
				const ltxt = new WDCReader('DBFilesClient/LiquidTypeXTexture.db2');
				await ltxt.parse();
				ltxt.preload();

				for (const [, row] of ltxt.getAllRows()) {
					const type_id = row.LiquidTypeID;
					if (!type_id)
						continue;

					if (!this._liquid_type_textures.has(type_id))
						this._liquid_type_textures.set(type_id, []);

					this._liquid_type_textures.get(type_id).push({
						file_data_id: row.FileDataID,
						order: row.OrderIndex ?? 0,
						type: row.Type ?? 0
					});
				}

				// sort each entry list by order index
				for (const entries of this._liquid_type_textures.values())
					entries.sort((a, b) => a.order - b.order);

				log.write('LiquidTypeXTexture loaded: %d liquid types mapped', this._liquid_type_textures.size);
			} catch {
				// LiquidTypeXTexture may not exist in older clients
			}

			this._db_loaded = true;
			this._liquid_type_cache.clear();
			log.write('Liquid DB2 tables loaded');
		} catch (e) {
			log.write('Failed to load liquid DB2 tables: %s', e.message);
		}
		this._db_loading = false;
	}

	_get_liquid_info(liquid_type, liquid_object) {
		const cache_key = liquid_type + '_' + liquid_object;
		if (this._liquid_type_cache.has(cache_key))
			return this._liquid_type_cache.get(cache_key);

		const info = {
			material_id: 1,
			color: LIQUID_COLORS[liquid_category(liquid_type)],
			flags: 0,
			float0: 1.0,
			float1: 0.0,
			texture_id: 0,
			generate_uv_from_pos: true
		};

		if (!this._db_loaded) {
			this._liquid_type_cache.set(cache_key, info);
			return info;
		}

		let resolved_type = liquid_type;

		// resolve via LiquidObject if >= 42
		if (liquid_object >= 42 && this._liquid_object_db) {
			const lo_row = this._liquid_object_db.getRow(liquid_object);
			if (lo_row)
				resolved_type = lo_row.LiquidTypeID ?? resolved_type;
		}

		const lt_row = this._liquid_type_db?.getRow(resolved_type);
		if (lt_row) {
			info.material_id = lt_row.MaterialID ?? 1;
			info.flags = lt_row.Flags ?? 0;

			// animation floats
			const floats = lt_row.Float;
			if (floats) {
				info.float0 = floats[0] ?? 1.0;
				info.float1 = floats[1] ?? 0.0;
			}

			// color (BGR→RGB)
			const color1 = lt_row.Color;
			if (color1 && (color1[0] > 0 || color1[1] > 0 || color1[2] > 0))
				info.color = new Float32Array([color1[2] / 255, color1[1] / 255, color1[0] / 255]);

			// texture resolution: prefer LiquidTypeXTexture, fall back to string paths
			const ltxt_entries = this._liquid_type_textures?.get(resolved_type);
			if (ltxt_entries && ltxt_entries.length > 0) {
				info.texture_id = ltxt_entries[0].file_data_id;
			} else {
				const tex_paths = lt_row.Texture;
				if (tex_paths && tex_paths[0]) {
					// replace %d frame placeholder with 0 for first frame
					const tex_path = tex_paths[0].replace(/\\/g, '/').replace(/%d/g, '0').toLowerCase();
					if (tex_path.length > 0) {
						const fid = listfile.getByFilename(tex_path);
						if (fid)
							info.texture_id = fid;
						else
							info.texture_path = tex_path;
					}
				}
			}
		}

		// determine UV generation mode
		const mat_id = info.material_id;
		if (MATERIAL_MAGMA.includes(mat_id)) {
			info.generate_uv_from_pos = false;
		} else if (liquid_type === 2 || liquid_type === 14) {
			info.generate_uv_from_pos = true;
		} else if (liquid_object < 42 && mat_id === 1) {
			info.generate_uv_from_pos = true;
		} else {
			info.generate_uv_from_pos = true;
		}

		this._liquid_type_cache.set(cache_key, info);
		return info;
	}

	on_tile_loaded(key, liquid_chunks, chunk_positions) {
		if (!liquid_chunks || this._disposed)
			return;

		const instances = [];

		for (let i = 0; i < 256; i++) {
			const chunk = liquid_chunks[i];
			if (!chunk?.instances)
				continue;

			const pos = chunk_positions[i];
			if (!pos)
				continue;

			for (const inst of chunk.instances) {
				if (!inst)
					continue;

				const geo = this._build_liquid_geometry(inst, pos);
				if (!geo)
					continue;

				const info = this._get_liquid_info(inst.liquidType, inst.liquidObject);
				const vao = this._upload_liquid_vao(geo);

				instances.push({
					vao,
					index_count: geo.index_count,
					info,
					bounds_min: geo.bounds_min,
					bounds_max: geo.bounds_max
				});

				// queue texture load
				const tex_key = info.texture_id || info.texture_path;
				if (tex_key && !this._texture_cache.has(tex_key) && !this._texture_loading.has(tex_key))
					this._texture_load_queue.push({ key: tex_key, id: info.texture_id, path: info.texture_path });
			}
		}

		if (instances.length === 0)
			return;

		// sort by info reference to minimize uniform changes during render
		const info_order = new Map();
		let next_order = 0;
		for (const inst of instances) {
			if (!info_order.has(inst.info))
				info_order.set(inst.info, next_order++);
		}
		instances.sort((a, b) => info_order.get(a.info) - info_order.get(b.info));

		// build flat bounds array and tile-level AABB
		const count = instances.length;
		const bounds = new Float32Array(count * 6);
		const tile_min = new Float32Array([Infinity, Infinity, Infinity]);
		const tile_max = new Float32Array([-Infinity, -Infinity, -Infinity]);

		for (let i = 0; i < count; i++) {
			const inst = instances[i];
			const bo = i * 6;
			bounds[bo] = inst.bounds_min[0];
			bounds[bo + 1] = inst.bounds_min[1];
			bounds[bo + 2] = inst.bounds_min[2];
			bounds[bo + 3] = inst.bounds_max[0];
			bounds[bo + 4] = inst.bounds_max[1];
			bounds[bo + 5] = inst.bounds_max[2];

			for (let j = 0; j < 3; j++) {
				if (inst.bounds_min[j] < tile_min[j])
					tile_min[j] = inst.bounds_min[j];
				if (inst.bounds_max[j] > tile_max[j])
					tile_max[j] = inst.bounds_max[j];
			}

			// free per-instance bounds (now in flat array)
			delete inst.bounds_min;
			delete inst.bounds_max;
		}

		this._tile_data.set(key, { instances, bounds, bounds_min: tile_min, bounds_max: tile_max });
	}

	on_tile_unloaded(key) {
		const data = this._tile_data.get(key);
		if (!data)
			return;

		for (const inst of data.instances) {
			if (inst.vao)
				inst.vao.dispose();
		}

		this._tile_data.delete(key);
	}

	_build_liquid_geometry(inst, chunk_pos) {
		const width = inst.width;
		const height = inst.height;
		if (width === 0 || height === 0)
			return null;

		const vert_w = width + 1;
		const vert_h = height + 1;
		const vert_count = vert_w * vert_h;

		const heights = inst.vertexData?.height;
		if (!heights || heights.length < vert_count)
			return null;

		const info = this._get_liquid_info(inst.liquidType, inst.liquidObject);
		const has_uv = !info.generate_uv_from_pos && inst.vertexData?.uv;
		const has_depth = !!inst.vertexData?.depth;

		const vertex_data = new ArrayBuffer(vert_count * LIQUID_VERTEX_STRIDE);
		const view = new DataView(vertex_data);

		// chunk base position (same coord system as terrain)
		const cx = chunk_pos[0];
		const cy = chunk_pos[1];
		const cz = chunk_pos[2];

		let min_x = Infinity, min_y = Infinity, min_z = Infinity;
		let max_x = -Infinity, max_y = -Infinity, max_z = -Infinity;

		for (let row = 0; row < vert_h; row++) {
			for (let col = 0; col < vert_w; col++) {
				const idx = row * vert_w + col;
				const offset = idx * LIQUID_VERTEX_STRIDE;

				// position: same transform as terrain vertices
				const vx = cy - ((col + inst.xOffset) * UNIT_SIZE);
				const vy = heights[idx];
				const vz = cx - ((row + inst.yOffset) * UNIT_SIZE);

				view.setFloat32(offset, vx, true);
				view.setFloat32(offset + 4, vy, true);
				view.setFloat32(offset + 8, vz, true);

				if (vx < min_x) min_x = vx;
				if (vy < min_y) min_y = vy;
				if (vz < min_z) min_z = vz;
				if (vx > max_x) max_x = vx;
				if (vy > max_y) max_y = vy;
				if (vz > max_z) max_z = vz;

				// UV
				let u, v;
				if (has_uv && inst.vertexData.uv[idx]) {
					u = inst.vertexData.uv[idx].x * 3.0 / 256.0;
					v = inst.vertexData.uv[idx].y * 3.0 / 256.0;
				} else {
					u = vx * 0.06;
					v = vz * 0.06;
				}

				view.setFloat32(offset + 12, u, true);
				view.setFloat32(offset + 16, v, true);

				// depth
				const depth = has_depth ? (inst.vertexData.depth[idx] / 255) : 1.0;
				view.setFloat32(offset + 20, depth, true);
			}
		}

		const bounds_min = new Float32Array([min_x, min_y, min_z]);
		const bounds_max = new Float32Array([max_x, max_y, max_z]);

		// build index buffer using existence bitmap
		const max_quads = width * height;
		const indices = [];

		for (let row = 0; row < height; row++) {
			for (let col = 0; col < width; col++) {
				const quad_idx = row * width + col;

				// check bitmap
				if (inst.bitmap && inst.bitmap.length > 0) {
					const byte_idx = quad_idx >> 3;
					const bit_idx = quad_idx & 7;
					if (byte_idx < inst.bitmap.length && !(inst.bitmap[byte_idx] & (1 << bit_idx)))
						continue;
				}

				const tl = row * vert_w + col;
				const tr = tl + 1;
				const bl = (row + 1) * vert_w + col;
				const br = bl + 1;

				indices.push(tl, tr, bl);
				indices.push(tr, br, bl);
			}
		}

		if (indices.length === 0)
			return null;

		return {
			vertex_data,
			index_data: new Uint16Array(indices),
			index_count: indices.length,
			bounds_min,
			bounds_max
		};
	}

	_upload_liquid_vao(geo) {
		const gl = this.gl;
		const vao = new VertexArray(this.ctx);
		vao.bind();

		vao.set_vertex_buffer(geo.vertex_data);

		// pos(3f) at 0, uv(2f) at 12, depth(1f) at 20, stride 24
		gl.enableVertexAttribArray(0);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, LIQUID_VERTEX_STRIDE, 0);
		gl.enableVertexAttribArray(1);
		gl.vertexAttribPointer(1, 2, gl.FLOAT, false, LIQUID_VERTEX_STRIDE, 12);
		gl.enableVertexAttribArray(2);
		gl.vertexAttribPointer(2, 1, gl.FLOAT, false, LIQUID_VERTEX_STRIDE, 20);

		vao.set_index_buffer(geo.index_data);

		return vao;
	}

	update() {
		if (this._disposed)
			return;

		if (this.enabled)
			this._pump_texture_queue();
	}

	_pump_texture_queue() {
		while (this._texture_loading.size < MAX_TEXTURE_LOADS && this._texture_load_queue.length > 0) {
			const entry = this._texture_load_queue.shift();
			if (this._texture_cache.has(entry.key) || this._texture_loading.has(entry.key))
				continue;

			this._start_texture_load(entry);
		}
	}

	async _start_texture_load(entry) {
		this._texture_loading.add(entry.key);

		try {
			let data;
			if (entry.id > 0)
				data = await this._casc.getFile(entry.id);
			else if (entry.path)
				data = await this._casc.getFileByName(entry.path);

			if (this._disposed || !data) {
				this._texture_loading.delete(entry.key);
				return;
			}

			const blp = new BLPFile(data);
			const gl_tex = new GLTexture(this.ctx);
			gl_tex.set_blp(blp, { flags: 0x3 }); // REPEAT both axes
			this._texture_cache.set(entry.key, gl_tex);
		} catch (e) {
			log.write('Failed to load liquid texture %s: %s', entry.key, e.message);
		}

		this._texture_loading.delete(entry.key);
	}

	render(view, proj, light_dir, sun_color, sun_intensity, fog_params) {
		if (!this.enabled || this._tile_data.size === 0)
			return 0;

		const gl = this.gl;
		const ctx = this.ctx;
		const shader = this.shader;

		if (!shader.is_valid())
			return 0;

		shader.use();
		shader.set_uniform_mat4('u_view', false, view);
		shader.set_uniform_mat4('u_projection', false, proj);
		shader.set_uniform_3fv('u_light_dir', light_dir);
		shader.set_uniform_3fv('u_sun_color', sun_color);
		shader.set_uniform_1f('u_sun_intensity', sun_intensity);
		shader.set_uniform_1i('u_texture', 0);

		const time = performance.now() - this._start_time;
		shader.set_uniform_1f('u_time', time);

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

		// alpha blending
		ctx.set_blend(true);
		ctx.set_blend_func_separate(
			gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA,
			gl.ONE, gl.ONE_MINUS_SRC_ALPHA
		);
		ctx.set_depth_test(true);
		ctx.set_depth_write(true);
		ctx.set_cull_face(false);

		this._compute_frustum(view, proj);
		const planes = this._frustum_planes;

		let drawn = 0;
		let last_info = null;

		for (const tile of this._tile_data.values()) {
			if (!this._is_tile_visible(tile, planes))
				continue;

			const bounds = tile.bounds;
			for (let i = 0; i < tile.instances.length; i++) {
				if (!this._is_aabb_visible(bounds, i * 6, planes))
					continue;

				const inst = tile.instances[i];

				if (inst.info !== last_info) {
					last_info = inst.info;
					shader.set_uniform_1i('u_material_id', last_info.material_id);
					shader.set_uniform_1i('u_liquid_flags', last_info.flags);
					shader.set_uniform_3fv('u_liquid_color', last_info.color);
					shader.set_uniform_1f('u_float0', last_info.float0);
					shader.set_uniform_1f('u_float1', last_info.float1);

					const tex_key = last_info.texture_id || last_info.texture_path;
					const tex = tex_key
						? (this._texture_cache.get(tex_key) ?? this._default_texture)
						: this._default_texture;

					tex.bind(0);
				}

				inst.vao.bind();
				gl.drawElements(gl.TRIANGLES, inst.index_count, gl.UNSIGNED_SHORT, 0);
				drawn++;
			}
		}

		// reset state
		ctx.set_blend(false);
		ctx.set_depth_write(true);

		return drawn;
	}

	set_enabled(val) {
		this.enabled = val;
	}

	_compute_frustum(view, proj) {
		const vp = this._vp_matrix;
		for (let i = 0; i < 4; i++) {
			for (let j = 0; j < 4; j++) {
				vp[j * 4 + i] =
					proj[i] * view[j * 4] +
					proj[4 + i] * view[j * 4 + 1] +
					proj[8 + i] * view[j * 4 + 2] +
					proj[12 + i] * view[j * 4 + 3];
			}
		}

		const p = this._frustum_planes;

		p[0] = vp[3] + vp[0]; p[1] = vp[7] + vp[4]; p[2] = vp[11] + vp[8]; p[3] = vp[15] + vp[12];
		p[4] = vp[3] - vp[0]; p[5] = vp[7] - vp[4]; p[6] = vp[11] - vp[8]; p[7] = vp[15] - vp[12];
		p[8] = vp[3] + vp[1]; p[9] = vp[7] + vp[5]; p[10] = vp[11] + vp[9]; p[11] = vp[15] + vp[13];
		p[12] = vp[3] - vp[1]; p[13] = vp[7] - vp[5]; p[14] = vp[11] - vp[9]; p[15] = vp[15] - vp[13];
		p[16] = vp[3] + vp[2]; p[17] = vp[7] + vp[6]; p[18] = vp[11] + vp[10]; p[19] = vp[15] + vp[14];
		p[20] = vp[3] - vp[2]; p[21] = vp[7] - vp[6]; p[22] = vp[11] - vp[10]; p[23] = vp[15] - vp[14];

		for (let i = 0; i < 6; i++) {
			const o = i * 4;
			const len = Math.sqrt(p[o] * p[o] + p[o + 1] * p[o + 1] + p[o + 2] * p[o + 2]);
			if (len > 0) {
				p[o] /= len;
				p[o + 1] /= len;
				p[o + 2] /= len;
				p[o + 3] /= len;
			}
		}
	}

	_is_tile_visible(tile, planes) {
		const min = tile.bounds_min;
		const max = tile.bounds_max;

		for (let i = 0; i < 6; i++) {
			const o = i * 4;
			const a = planes[o], b = planes[o + 1], c = planes[o + 2], d = planes[o + 3];
			const px = a >= 0 ? max[0] : min[0];
			const py = b >= 0 ? max[1] : min[1];
			const pz = c >= 0 ? max[2] : min[2];

			if (a * px + b * py + c * pz + d < 0)
				return false;
		}

		return true;
	}

	_is_aabb_visible(bounds, bo, planes) {
		for (let i = 0; i < 6; i++) {
			const o = i * 4;
			const a = planes[o], b = planes[o + 1], c = planes[o + 2], d = planes[o + 3];
			const px = a >= 0 ? bounds[bo + 3] : bounds[bo];
			const py = b >= 0 ? bounds[bo + 4] : bounds[bo + 1];
			const pz = c >= 0 ? bounds[bo + 5] : bounds[bo + 2];

			if (a * px + b * py + c * pz + d < 0)
				return false;
		}

		return true;
	}

	dispose() {
		this._disposed = true;
		this._texture_load_queue.length = 0;
		this._texture_loading.clear();

		for (const tile of this._tile_data.values()) {
			for (const inst of tile.instances) {
				if (inst.vao)
					inst.vao.dispose();
			}
		}
		this._tile_data.clear();

		for (const tex of this._texture_cache.values())
			tex.dispose();
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

module.exports = LiquidRenderer;

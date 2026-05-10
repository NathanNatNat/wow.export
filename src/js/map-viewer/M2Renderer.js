/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const core = require('../core');
const constants = require('../constants');
const Shaders = require('../3D/Shaders');
const VertexArray = require('../3D/gl/VertexArray');
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
const M2_VERTEX_STRIDE = 40;

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
	const ay = rotation[1] * DEG_TO_RAD;
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
 * format: position(3f) + normal(3f) + uv(2f) + uv2(2f) = 40 bytes
 */
function build_model_geometry(m2, skin) {
	const vertex_count = m2.vertices.length / 3;
	if (vertex_count === 0)
		return null;

	const vertex_data = new ArrayBuffer(vertex_count * M2_VERTEX_STRIDE);
	const view = new DataView(vertex_data);

	for (let i = 0; i < vertex_count; i++) {
		const offset = i * M2_VERTEX_STRIDE;
		const v = i * 3;
		const uv = i * 2;

		// un-swizzle M2Loader output [wow_x, wow_z, -wow_y] -> [wow_x, wow_y, wow_z]
		view.setFloat32(offset, m2.vertices[v], true);
		view.setFloat32(offset + 4, -m2.vertices[v + 2], true);
		view.setFloat32(offset + 8, m2.vertices[v + 1], true);

		view.setFloat32(offset + 12, m2.normals[v], true);
		view.setFloat32(offset + 16, -m2.normals[v + 2], true);
		view.setFloat32(offset + 20, m2.normals[v + 1], true);

		view.setFloat32(offset + 24, m2.uv[uv], true);
		view.setFloat32(offset + 28, m2.uv[uv + 1], true);

		view.setFloat32(offset + 32, m2.uv2[uv], true);
		view.setFloat32(offset + 36, m2.uv2[uv + 1], true);
	}

	// map triangle indices through skin indirection
	const index_data = new Uint16Array(skin.triangles.length);
	for (let i = 0; i < skin.triangles.length; i++)
		index_data[i] = skin.indices[skin.triangles[i]];

	return { vertex_data, index_data, index_count: index_data.length };
}

class M2Renderer {
	constructor(gl_context) {
		this.ctx = gl_context;
		this.gl = gl_context.gl;
		this.shader = Shaders.create_program(gl_context, 'mpv_m2');
		this._casc = core.view.casc;

		this._model_cache = new Map();
		this._tile_data = new Map();

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

	update(camera_pos) {
		if (this._disposed)
			return;

		this._process_uploads();
		this._pump_obj0_queue();
		this._pump_model_queue();

		const dx = camera_pos[0] - this._last_cam[0];
		const dy = camera_pos[1] - this._last_cam[1];
		const dz = camera_pos[2] - this._last_cam[2];

		if (dx * dx + dy * dy + dz * dz > CAMERA_DIRTY_THRESHOLD_SQ) {
			this._instances_dirty = true;
			this._last_cam.set(camera_pos);
		}

		if (this._instances_dirty)
			this._rebuild_instance_buffers(camera_pos);
	}

	render(view, proj, light_dir, sun_color, sun_intensity, fog_params) {
		if (!this.enabled)
			return 0;

		const gl = this.gl;
		const shader = this.shader;

		shader.use();
		shader.set_uniform_mat4('u_view', false, view);
		shader.set_uniform_mat4('u_projection', false, proj);
		shader.set_uniform_3fv('u_light_dir', light_dir);
		shader.set_uniform_3fv('u_sun_color', sun_color);
		shader.set_uniform_1f('u_sun_intensity', sun_intensity);

		if (fog_params) {
			shader.set_uniform_3fv('u_camera_pos', fog_params.camera_pos);
			shader.set_uniform_3fv('u_fog_color', fog_params.fog_color);
			shader.set_uniform_1f('u_fog_start', fog_params.fog_start);
			shader.set_uniform_1f('u_fog_end', fog_params.fog_end);
		} else {
			shader.set_uniform_1f('u_fog_start', 999999.0);
			shader.set_uniform_1f('u_fog_end', 999999.0);
		}

		let drawn = 0;

		for (const entry of this._model_cache.values()) {
			if (!entry.vao || entry.instance_count === 0)
				continue;

			entry.vao.bind();
			gl.drawElementsInstanced(gl.TRIANGLES, entry.index_count, entry.vao.index_type, 0, entry.instance_count);
			drawn += entry.instance_count;
		}

		return drawn;
	}

	set_render_distance(value) {
		if (this.render_distance === value)
			return;

		this.render_distance = value;
		this._instances_dirty = true;
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
						instance_buffer: null,
						instance_count: 0,
						ref_count: 0,
						tile_instances: new Map(),
						queued: false
					};
					this._model_cache.set(id, entry);
				}

				entry.ref_count++;
				entry.tile_instances.set(key, instances);

				if (!entry.vao && !entry.queued && !this._model_loading.has(id)) {
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

			const geo = build_model_geometry(m2, skin);
			if (!geo) {
				this._model_loading.delete(file_data_id);
				return;
			}

			this._upload_queue.push({ file_data_id, ...geo });
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

		// vertex format: pos(3f) + normal(3f) + uv(2f) + uv2(2f) = 40 bytes
		const stride = M2_VERTEX_STRIDE;
		gl.enableVertexAttribArray(0);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
		gl.enableVertexAttribArray(1);
		gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 12);
		gl.enableVertexAttribArray(4);
		gl.vertexAttribPointer(4, 2, gl.FLOAT, false, stride, 24);
		gl.enableVertexAttribArray(5);
		gl.vertexAttribPointer(5, 2, gl.FLOAT, false, stride, 32);

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

		entry.vao = vao;
		entry.index_count = data.index_count;
		entry.instance_buffer = instance_buffer;
		entry.queued = false;
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

	_dispose_model(file_data_id) {
		const entry = this._model_cache.get(file_data_id);
		if (!entry)
			return;

		if (entry.vao)
			entry.vao.dispose();

		if (entry.instance_buffer)
			this.gl.deleteBuffer(entry.instance_buffer);

		this._model_cache.delete(file_data_id);
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

		if (this.shader) {
			Shaders.unregister(this.shader);
			this.shader.dispose();
			this.shader = null;
		}
	}
}

module.exports = M2Renderer;

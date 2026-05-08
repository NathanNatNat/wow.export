const core = require('../core');
const constants = require('../constants');
const ADTLoader = require('../3D/loaders/ADTLoader');
const WDTLoader = require('../3D/loaders/WDTLoader');
const Shaders = require('../3D/Shaders');
const VertexArray = require('../3D/gl/VertexArray');
const GLTexture = require('../3D/gl/GLTexture');
const BLPFile = require('../casc/blp');

const MAP_SIZE = constants.GAME.MAP_SIZE;
const TILE_SIZE = constants.GAME.TILE_SIZE;
const CHUNK_SIZE = TILE_SIZE / 16;
const UNIT_SIZE = CHUNK_SIZE / 8;
const UNIT_SIZE_HALF = UNIT_SIZE / 2;

const MAX_CONCURRENT_LOADS = 4;
const UNLOAD_PADDING = 2;
const DEFAULT_RENDER_DISTANCE = 8;

// 256 chunks x 145 verts x 32 bytes (pos3f + normal3f + uv2f)
const MAX_TILE_VERTEX_BYTES = 256 * 145 * 32;
// 256 chunks x 768 indices x 2 bytes (uint16)
const MAX_TILE_INDEX_BYTES = 256 * 768 * 2;
// wireframe: each triangle (3 indices) becomes 3 line pairs (6 indices)
const MAX_TILE_WIRE_INDEX_BYTES = MAX_TILE_INDEX_BYTES * 2;

class TerrainRenderer {
	constructor(gl_context) {
		this.ctx = gl_context;
		this.gl = gl_context.gl;
		this.shader = Shaders.create_program(gl_context, 'mpv_terrain');
		this.wire_shader = Shaders.create_program(gl_context, 'mpv_terrain_wire');
		this.minimap_shader = Shaders.create_program(gl_context, 'mpv_terrain_minimap');
		this.render_distance = DEFAULT_RENDER_DISTANCE;
		this.map_center = [0, 0, 0];

		this._tiles = new Map();
		this._tile_info = new Map();
		this._loading = new Set();
		this._load_queue = [];
		this._upload_queue = [];
		this._casc = null;
		this._map_dir = null;
		this._chunk_count = 0;
		this._disposed = false;

		this._last_tx = NaN;
		this._last_ty = NaN;

		this._vao_pool = [];
		this._frustum_planes = new Float32Array(24);
		this._vp_matrix = new Float32Array(16);
	}

	get tile_count() {
		return this._tiles.size;
	}

	get loading_count() {
		return this._load_queue.length + this._loading.size + this._upload_queue.length;
	}

	get chunk_count() {
		return this._chunk_count;
	}

	set_render_distance(val) {
		this.render_distance = Math.max(1, Math.min(256, val));

		// force tile re-evaluation on next frame
		this._last_tx = NaN;
		this._last_ty = NaN;
	}

	async init(map_dir) {
		this._casc = core.view.casc;
		this._map_dir = map_dir;
		const prefix = 'world/maps/' + map_dir + '/' + map_dir;

		const wdt_file = await this._casc.getFileByName(prefix + '.wdt');
		const wdt = new WDTLoader(wdt_file);
		wdt.load();

		if (!wdt.entries)
			throw new Error('WDT has no tile entries');

		let min_tx = MAP_SIZE, min_ty = MAP_SIZE, max_tx = 0, max_ty = 0;

		for (let x = 0; x < MAP_SIZE; x++) {
			for (let y = 0; y < MAP_SIZE; y++) {
				const idx = (y * MAP_SIZE) + x;
				if (!wdt.tiles[idx])
					continue;

				const entry = wdt.entries[idx];
				if (!entry || !entry.rootADT)
					continue;

				const key = x + '_' + y;
				this._tile_info.set(key, { root_id: entry.rootADT, x, y });

				if (x < min_tx) min_tx = x;
				if (y < min_ty) min_ty = y;
				if (x > max_tx) max_tx = x;
				if (y > max_ty) max_ty = y;
			}
		}

		const center_tx = (min_tx + max_tx) / 2;
		const center_ty = (min_ty + max_ty) / 2;

		this.map_center = [
			(32 - center_ty) * TILE_SIZE,
			0,
			(32 - center_tx) * TILE_SIZE
		];
	}

	update(camera_pos) {
		this._process_uploads();

		const tx = Math.floor(32 - camera_pos[2] / TILE_SIZE);
		const ty = Math.floor(32 - camera_pos[0] / TILE_SIZE);

		if (tx !== this._last_tx || ty !== this._last_ty) {
			this._last_tx = tx;
			this._last_ty = ty;
			this._update_needed_tiles(tx, ty);
		}

		this._pump_load_queue();
	}

	_update_needed_tiles(center_tx, center_ty) {
		const rd = this.render_distance;
		const ud = rd + UNLOAD_PADDING;

		// unload tiles beyond unload distance
		for (const [key, tile] of this._tiles) {
			if (Math.abs(tile.x - center_tx) > ud || Math.abs(tile.y - center_ty) > ud) {
				if (tile.minimap_tex)
					tile.minimap_tex.dispose();

				this._release_vao(tile.vao);
				this._chunk_count -= tile.chunk_count;
				this._tiles.delete(key);
			}
		}

		// cancel in-flight loads beyond unload distance
		for (const key of this._loading) {
			const info = this._tile_info.get(key);
			if (Math.abs(info.x - center_tx) > ud || Math.abs(info.y - center_ty) > ud)
				this._loading.delete(key);
		}

		// build priority-sorted load queue (closest first)
		const to_load = [];
		for (let dx = -rd; dx <= rd; dx++) {
			for (let dy = -rd; dy <= rd; dy++) {
				const tx = center_tx + dx;
				const ty = center_ty + dy;
				const key = tx + '_' + ty;

				if (!this._tile_info.has(key))
					continue;

				if (this._tiles.has(key) || this._loading.has(key))
					continue;

				to_load.push({ key, dist: dx * dx + dy * dy });
			}
		}

		to_load.sort((a, b) => a.dist - b.dist);
		this._load_queue = to_load.map(e => e.key);
	}

	_pump_load_queue() {
		if (this._disposed)
			return;

		while (this._loading.size < MAX_CONCURRENT_LOADS && this._load_queue.length > 0) {
			const key = this._load_queue.shift();
			if (this._tiles.has(key) || this._loading.has(key))
				continue;

			this._start_tile_load(key);
		}
	}

	async _start_tile_load(key) {
		this._loading.add(key);
		const info = this._tile_info.get(key);

		try {
			const root_file = await this._casc.getFile(info.root_id);

			if (!this._loading.has(key))
				return this._pump_load_queue();

			const adt = new ADTLoader(root_file);
			adt.loadRoot();

			if (!this._loading.has(key))
				return this._pump_load_queue();

			const geo = this._build_tile_geometry(adt, info.x, info.y);
			if (!geo) {
				this._loading.delete(key);
				return;
			}

			// load minimap texture alongside geometry
			let minimap_blp = null;
			try {
				const px = info.x.toString().padStart(2, '0');
				const py = info.y.toString().padStart(2, '0');
				const blp_path = 'world/minimaps/' + this._map_dir + '/map' + px + '_' + py + '.blp';
				const blp_data = await this._casc.getFileByName(blp_path, false, true);
				if (blp_data)
					minimap_blp = new BLPFile(blp_data);
			} catch {
				// minimap not available for this tile
			}

			if (!this._loading.has(key))
				return this._pump_load_queue();

			this._upload_queue.push({ key, geo, minimap_blp });
		} catch (e) {
			this._loading.delete(key);
		}

		this._pump_load_queue();
	}

	_process_uploads(budget = 1) {
		while (budget-- > 0 && this._upload_queue.length > 0) {
			const { key, geo, minimap_blp } = this._upload_queue.shift();

			if (!this._loading.has(key))
				continue;

			this._loading.delete(key);

			const tile = this._upload_tile(geo, minimap_blp);
			this._tiles.set(key, tile);
			this._chunk_count += tile.chunk_count;
		}
	}

	_upload_tile(geo, minimap_blp) {
		const vao = this._acquire_vao();
		const gl = this.gl;

		vao.bind();
		gl.bindBuffer(gl.ARRAY_BUFFER, vao.vbo);
		gl.bufferSubData(gl.ARRAY_BUFFER, 0, geo.vertex_data);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.ebo);
		gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, geo.index_data);

		const wire_indices = VertexArray.triangles_to_lines(geo.index_data);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.wireframe_ebo);
		gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, wire_indices);

		// restore triangle EBO in VAO state
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.ebo);

		let minimap_tex = null;
		if (minimap_blp) {
			minimap_tex = new GLTexture(this.ctx);
			minimap_tex.set_blp(minimap_blp);
		}

		return {
			vao,
			minimap_tex,
			chunk_bounds: geo.chunk_bounds,
			chunk_draw: geo.chunk_draw,
			chunk_count: geo.chunk_count,
			x: geo.x,
			y: geo.y,
			bounds_min: geo.bounds_min,
			bounds_max: geo.bounds_max
		};
	}

	_acquire_vao() {
		if (this._vao_pool.length > 0)
			return this._vao_pool.pop();

		const vao = new VertexArray(this.ctx);
		const gl = this.gl;
		vao.bind();

		vao.vbo = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vao.vbo);
		gl.bufferData(gl.ARRAY_BUFFER, MAX_TILE_VERTEX_BYTES, gl.DYNAMIC_DRAW);

		const stride = 32;
		gl.enableVertexAttribArray(0);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
		gl.enableVertexAttribArray(1);
		gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 12);
		gl.enableVertexAttribArray(2);
		gl.vertexAttribPointer(2, 2, gl.FLOAT, false, stride, 24);

		vao.ebo = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.ebo);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, MAX_TILE_INDEX_BYTES, gl.DYNAMIC_DRAW);
		vao.index_type = gl.UNSIGNED_SHORT;

		vao.wireframe_ebo = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.wireframe_ebo);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, MAX_TILE_WIRE_INDEX_BYTES, gl.DYNAMIC_DRAW);

		// restore triangle EBO in VAO state
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.ebo);

		return vao;
	}

	_release_vao(vao) {
		this._vao_pool.push(vao);
	}

	_build_tile_geometry(adt, tx, ty) {
		let valid_chunks = 0;
		for (let i = 0; i < 256; i++) {
			if (adt.chunks[i]?.vertices)
				valid_chunks++;
		}

		if (valid_chunks === 0)
			return null;

		const vert_count = valid_chunks * 145;
		const vertex_data = new Float32Array(vert_count * 8);
		const index_data = new Uint16Array(valid_chunks * 768);

		const chunk_bounds = new Float32Array(valid_chunks * 6);
		const chunk_draw = new Uint32Array(valid_chunks * 2);

		const tile_min = [Infinity, Infinity, Infinity];
		const tile_max = [-Infinity, -Infinity, -Infinity];

		let vert_offset = 0;
		let chunk_vert_base = 0;
		let idx_offset = 0;
		let chunk_idx = 0;

		for (let x = 0; x < 16; x++) {
			for (let y = 0; y < 16; y++) {
				const chunk = adt.chunks[(x * 16) + y];
				if (!chunk?.vertices)
					continue;

				const chunk_min = [Infinity, Infinity, Infinity];
				const chunk_max = [-Infinity, -Infinity, -Infinity];

				const cx = chunk.position[0];
				const cy = chunk.position[1];
				const cz = chunk.position[2];

				for (let row = 0, idx = 0; row < 17; row++) {
					const is_short = !!(row % 2);
					const col_count = is_short ? 8 : 9;

					for (let col = 0; col < col_count; col++) {
						let vx = cy - (col * UNIT_SIZE);
						const vy = chunk.vertices[idx] + cz;
						const vz = cx - (row * UNIT_SIZE_HALF);

						if (is_short)
							vx -= UNIT_SIZE_HALF;

						const di = vert_offset * 8;
						vertex_data[di] = vx;
						vertex_data[di + 1] = vy;
						vertex_data[di + 2] = vz;

						if (chunk.normals) {
							const n = chunk.normals[idx];
							vertex_data[di + 3] = n[0] / 127;
							vertex_data[di + 4] = n[1] / 127;
							vertex_data[di + 5] = n[2] / 127;
						} else {
							vertex_data[di + 4] = 1;
						}

						// uv: map vertex to [0,1] across the tile
						const col_frac = is_short ? (col + 0.5) / 8 : col / 8;
						vertex_data[di + 6] = (y + col_frac) / 16;
						vertex_data[di + 7] = (x + row / 16) / 16;

						if (vx < tile_min[0]) tile_min[0] = vx;
						if (vy < tile_min[1]) tile_min[1] = vy;
						if (vz < tile_min[2]) tile_min[2] = vz;
						if (vx > tile_max[0]) tile_max[0] = vx;
						if (vy > tile_max[1]) tile_max[1] = vy;
						if (vz > tile_max[2]) tile_max[2] = vz;

						if (vx < chunk_min[0]) chunk_min[0] = vx;
						if (vy < chunk_min[1]) chunk_min[1] = vy;
						if (vz < chunk_min[2]) chunk_min[2] = vz;
						if (vx > chunk_max[0]) chunk_max[0] = vx;
						if (vy > chunk_max[1]) chunk_max[1] = vy;
						if (vz > chunk_max[2]) chunk_max[2] = vz;

						vert_offset++;
						idx++;
					}
				}

				const chunk_idx_start = idx_offset;

				for (let j = 9; j < 145; j++) {
					const ind = chunk_vert_base + j;
					index_data[idx_offset++] = ind;
					index_data[idx_offset++] = ind - 9;
					index_data[idx_offset++] = ind + 8;
					index_data[idx_offset++] = ind;
					index_data[idx_offset++] = ind - 8;
					index_data[idx_offset++] = ind - 9;
					index_data[idx_offset++] = ind;
					index_data[idx_offset++] = ind + 9;
					index_data[idx_offset++] = ind - 8;
					index_data[idx_offset++] = ind;
					index_data[idx_offset++] = ind + 8;
					index_data[idx_offset++] = ind + 9;

					if (!((j + 1) % 17))
						j += 9;
				}

				const bo = chunk_idx * 6;
				chunk_bounds[bo] = chunk_min[0];
				chunk_bounds[bo + 1] = chunk_min[1];
				chunk_bounds[bo + 2] = chunk_min[2];
				chunk_bounds[bo + 3] = chunk_max[0];
				chunk_bounds[bo + 4] = chunk_max[1];
				chunk_bounds[bo + 5] = chunk_max[2];

				const dw = chunk_idx * 2;
				chunk_draw[dw] = chunk_idx_start;
				chunk_draw[dw + 1] = idx_offset - chunk_idx_start;

				chunk_idx++;
				chunk_vert_base += 145;
			}
		}

		return {
			vertex_data,
			index_data,
			chunk_bounds,
			chunk_draw,
			chunk_count: chunk_idx,
			x: tx,
			y: ty,
			bounds_min: tile_min,
			bounds_max: tile_max
		};
	}

	render(view_matrix, projection_matrix, terrain_color) {
		if (!this.shader.is_valid() || this._tiles.size === 0)
			return 0;

		this.shader.use();
		this.shader.set_uniform_mat4('u_view', false, view_matrix);
		this.shader.set_uniform_mat4('u_projection', false, projection_matrix);
		this.shader.set_uniform_3fv('u_terrain_color', terrain_color);

		this._compute_frustum(view_matrix, projection_matrix);

		const gl = this.gl;
		const planes = this._frustum_planes;
		let visible = 0;

		for (const tile of this._tiles.values()) {
			if (!this._is_tile_visible(tile, planes))
				continue;

			tile.vao.bind();
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, tile.vao.ebo);
			visible += this._draw_tile_batched(tile, planes, gl.TRIANGLES, false);
		}

		return visible;
	}

	render_wireframe(view_matrix, projection_matrix, wire_color, sky_color, occlusion) {
		if (!this.wire_shader.is_valid() || this._tiles.size === 0)
			return 0;

		this._compute_frustum(view_matrix, projection_matrix);

		const gl = this.gl;
		const planes = this._frustum_planes;
		let visible = 0;

		this.wire_shader.use();
		this.wire_shader.set_uniform_mat4('u_view', false, view_matrix);
		this.wire_shader.set_uniform_mat4('u_projection', false, projection_matrix);

		if (occlusion) {
			// solid fill pass: populate depth buffer with sky-coloured triangles
			// so wireframe lines behind terrain are depth-rejected
			this.wire_shader.set_uniform_3fv('u_terrain_color', sky_color);

			gl.enable(gl.POLYGON_OFFSET_FILL);
			gl.polygonOffset(1, 1);

			for (const tile of this._tiles.values()) {
				if (!this._is_tile_visible(tile, planes))
					continue;

				tile.vao.bind();
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, tile.vao.ebo);
				this._draw_tile_batched(tile, planes, gl.TRIANGLES, false);
			}

			gl.disable(gl.POLYGON_OFFSET_FILL);
		}

		// wireframe pass
		this.wire_shader.set_uniform_3fv('u_terrain_color', wire_color);

		for (const tile of this._tiles.values()) {
			if (!this._is_tile_visible(tile, planes))
				continue;

			tile.vao.bind();
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, tile.vao.wireframe_ebo);
			visible += this._draw_tile_batched(tile, planes, gl.LINES, true);
		}

		return visible;
	}

	render_minimap(view_matrix, projection_matrix) {
		if (!this.minimap_shader.is_valid() || this._tiles.size === 0)
			return 0;

		this.minimap_shader.use();
		this.minimap_shader.set_uniform_mat4('u_view', false, view_matrix);
		this.minimap_shader.set_uniform_mat4('u_projection', false, projection_matrix);
		this.minimap_shader.set_uniform_1i('u_minimap', 0);

		this._compute_frustum(view_matrix, projection_matrix);

		const gl = this.gl;
		const planes = this._frustum_planes;
		let visible = 0;

		for (const tile of this._tiles.values()) {
			if (!tile.minimap_tex || !this._is_tile_visible(tile, planes))
				continue;

			tile.minimap_tex.bind(0);
			tile.vao.bind();
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, tile.vao.ebo);
			visible += this._draw_tile_batched(tile, planes, gl.TRIANGLES, false);
		}

		return visible;
	}

	_draw_tile_batched(tile, planes, mode, is_wireframe) {
		const gl = this.gl;
		const bounds = tile.chunk_bounds;
		const draw = tile.chunk_draw;
		const scale = is_wireframe ? 2 : 1;

		let batch_start = -1;
		let batch_count = 0;
		let visible = 0;

		for (let i = 0; i < tile.chunk_count; i++) {
			const bo = i * 6;
			const dw = i * 2;

			if (this._is_aabb_visible(bounds, bo, planes)) {
				visible++;
				const offset = draw[dw] * scale;
				const idx_count = draw[dw + 1] * scale;

				if (batch_start === -1) {
					batch_start = offset;
					batch_count = idx_count;
				} else if (offset === batch_start + batch_count) {
					batch_count += idx_count;
				} else {
					gl.drawElements(mode, batch_count, gl.UNSIGNED_SHORT, batch_start * 2);
					batch_start = offset;
					batch_count = idx_count;
				}
			} else if (batch_start !== -1) {
				gl.drawElements(mode, batch_count, gl.UNSIGNED_SHORT, batch_start * 2);
				batch_start = -1;
			}
		}

		if (batch_start !== -1)
			gl.drawElements(mode, batch_count, gl.UNSIGNED_SHORT, batch_start * 2);

		return visible;
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
		this._loading.clear();
		this._load_queue.length = 0;
		this._upload_queue.length = 0;

		for (const tile of this._tiles.values()) {
			if (tile.minimap_tex)
				tile.minimap_tex.dispose();

			tile.vao.dispose();
		}

		this._tiles.clear();

		for (const vao of this._vao_pool)
			vao.dispose();

		this._vao_pool.length = 0;

		if (this.shader) {
			Shaders.unregister(this.shader);
			this.shader.dispose();
			this.shader = null;
		}

		if (this.wire_shader) {
			Shaders.unregister(this.wire_shader);
			this.wire_shader.dispose();
			this.wire_shader = null;
		}

		if (this.minimap_shader) {
			Shaders.unregister(this.minimap_shader);
			this.minimap_shader.dispose();
			this.minimap_shader = null;
		}
	}
}

module.exports = TerrainRenderer;

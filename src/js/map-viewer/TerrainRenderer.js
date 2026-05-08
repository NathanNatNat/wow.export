const core = require('../core');
const constants = require('../constants');
const ADTLoader = require('../3D/loaders/ADTLoader');
const WDTLoader = require('../3D/loaders/WDTLoader');
const ShaderProgram = require('../3D/gl/ShaderProgram');
const VertexArray = require('../3D/gl/VertexArray');

const MAP_SIZE = constants.GAME.MAP_SIZE;
const TILE_SIZE = constants.GAME.TILE_SIZE;
const CHUNK_SIZE = TILE_SIZE / 16;
const UNIT_SIZE = CHUNK_SIZE / 8;
const UNIT_SIZE_HALF = UNIT_SIZE / 2;

const MAX_CONCURRENT_LOADS = 4;
const UNLOAD_PADDING = 2;
const DEFAULT_RENDER_DISTANCE = 8;

const VERT_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;
out vec3 v_position;

void main() {
	gl_Position = u_projection * u_view * vec4(a_position, 1.0);
	v_normal = a_normal;
	v_position = a_position;
}
`;

const FRAG_SHADER = `#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_position;

out vec4 frag_color;

const vec3 LIGHT_DIR = vec3(0.5051, 0.8081, 0.3031);
const vec3 SKY_COLOR = vec3(0.4, 0.5, 0.7);
const vec3 GROUND_COLOR = vec3(0.25, 0.2, 0.15);
const vec3 SUN_COLOR = vec3(1.0, 0.95, 0.85);
const float AMBIENT = 0.25;

void main() {
	vec3 n = normalize(v_normal);
	float n_dot_l = dot(n, LIGHT_DIR);

	// hemisphere ambient (sky vs ground)
	float sky_factor = 0.5 + 0.5 * n.y;
	vec3 ambient = mix(GROUND_COLOR, SKY_COLOR, sky_factor) * AMBIENT;

	// diffuse
	float diffuse = max(n_dot_l, 0.0);
	vec3 color = vec3(0.5, 0.55, 0.4) * (ambient + SUN_COLOR * diffuse * 0.75);

	frag_color = vec4(color, 1.0);
}
`;

class TerrainRenderer {
	constructor(gl_context) {
		this.ctx = gl_context;
		this.gl = gl_context.gl;
		this.shader = new ShaderProgram(gl_context, VERT_SHADER, FRAG_SHADER);
		this.render_distance = DEFAULT_RENDER_DISTANCE;
		this.map_center = [0, 0, 0];

		this._tiles = new Map();
		this._tile_info = new Map();
		this._loading = new Set();
		this._load_queue = [];
		this._casc = null;
		this._chunk_count = 0;
		this._disposed = false;

		this._last_tx = NaN;
		this._last_ty = NaN;

		this._frustum_planes = new Float32Array(24);
		this._vp_matrix = new Float32Array(16);
	}

	get tile_count() {
		return this._tiles.size;
	}

	get loading_count() {
		return this._loading.size;
	}

	get chunk_count() {
		return this._chunk_count;
	}

	async init(map_dir) {
		this._casc = core.view.casc;
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
				tile.vao.dispose();
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

			this._loading.delete(key);

			const tile = this._build_tile(adt, info.x, info.y);
			if (tile) {
				this._tiles.set(key, tile);
				this._chunk_count += tile.chunk_count;
			}
		} catch (e) {
			this._loading.delete(key);
		}

		this._pump_load_queue();
	}

	_build_tile(adt, tx, ty) {
		let valid_chunks = 0;
		for (let i = 0; i < 256; i++) {
			if (adt.chunks[i]?.vertices)
				valid_chunks++;
		}

		if (valid_chunks === 0)
			return null;

		const vert_count = valid_chunks * 145;
		const vertex_data = new Float32Array(vert_count * 6);
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

						const di = vert_offset * 6;
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

		const vao = new VertexArray(this.ctx);
		vao.bind();
		vao.set_vertex_buffer(vertex_data);

		const gl = this.gl;
		const stride = 24;
		vao.set_attribute(0, 3, gl.FLOAT, false, stride, 0);
		vao.set_attribute(1, 3, gl.FLOAT, false, stride, 12);
		vao.set_index_buffer(index_data);

		return {
			vao,
			chunk_bounds,
			chunk_draw,
			chunk_count: chunk_idx,
			x: tx,
			y: ty,
			bounds_min: tile_min,
			bounds_max: tile_max
		};
	}

	render(view_matrix, projection_matrix) {
		if (!this.shader.is_valid() || this._tiles.size === 0)
			return 0;

		this.shader.use();
		this.shader.set_uniform_mat4('u_view', false, view_matrix);
		this.shader.set_uniform_mat4('u_projection', false, projection_matrix);

		this._compute_frustum(view_matrix, projection_matrix);

		const gl = this.gl;
		const planes = this._frustum_planes;
		let visible = 0;

		for (const tile of this._tiles.values()) {
			if (!this._is_tile_visible(tile, planes))
				continue;

			tile.vao.bind();

			const bounds = tile.chunk_bounds;
			const draw = tile.chunk_draw;
			let batch_start = -1;
			let batch_count = 0;

			for (let i = 0; i < tile.chunk_count; i++) {
				const bo = i * 6;
				const dw = i * 2;

				if (this._is_aabb_visible(bounds, bo, planes)) {
					visible++;
					const offset = draw[dw];
					const idx_count = draw[dw + 1];

					if (batch_start === -1) {
						batch_start = offset;
						batch_count = idx_count;
					} else if (offset === batch_start + batch_count) {
						batch_count += idx_count;
					} else {
						tile.vao.draw(gl.TRIANGLES, batch_count, batch_start);
						batch_start = offset;
						batch_count = idx_count;
					}
				} else if (batch_start !== -1) {
					tile.vao.draw(gl.TRIANGLES, batch_count, batch_start);
					batch_start = -1;
				}
			}

			if (batch_start !== -1)
				tile.vao.draw(gl.TRIANGLES, batch_count, batch_start);
		}

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

		for (const tile of this._tiles.values())
			tile.vao.dispose();

		this._tiles.clear();

		if (this.shader) {
			this.shader.dispose();
			this.shader = null;
		}
	}
}

module.exports = TerrainRenderer;

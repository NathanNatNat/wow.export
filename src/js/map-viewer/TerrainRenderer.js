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

const BATCH_SIZE = 8;

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
		this.vao = null;
		this.tile_count = 0;
		this._pending_tiles = [];
		this._tile_ranges = [];
		this._frustum_planes = new Float32Array(24);
		this._vp_matrix = new Float32Array(16);
		this.bounds = {
			min: [Infinity, Infinity, Infinity],
			max: [-Infinity, -Infinity, -Infinity]
		};
	}

	async load_map(map_dir, on_progress) {
		const casc = core.view.casc;
		const prefix = 'world/maps/' + map_dir + '/' + map_dir;

		const wdt_file = await casc.getFileByName(prefix + '.wdt');
		const wdt = new WDTLoader(wdt_file);
		wdt.load();

		if (!wdt.entries)
			throw new Error('WDT has no tile entries');

		// collect valid tiles
		const tile_infos = [];
		for (let x = 0; x < MAP_SIZE; x++) {
			for (let y = 0; y < MAP_SIZE; y++) {
				const idx = (y * MAP_SIZE) + x;
				if (!wdt.tiles[idx])
					continue;

				const entry = wdt.entries[idx];
				if (!entry || !entry.rootADT)
					continue;

				tile_infos.push({ root_id: entry.rootADT, x, y, index: idx });
			}
		}

		const total = tile_infos.length;
		let loaded = 0;

		if (on_progress)
			on_progress(0, total);

		// load tiles in batches to avoid overwhelming CASC
		for (let i = 0; i < tile_infos.length; i += BATCH_SIZE) {
			const batch = tile_infos.slice(i, i + BATCH_SIZE);
			await Promise.all(batch.map(info =>
				this._load_tile(casc, info.root_id)
					.catch(() => {})
					.then(() => {
						loaded++;
						if (on_progress)
							on_progress(loaded, total);
					})
			));
		}

		this._build_merged_vao();
	}

	async _load_tile(casc, root_file_id) {
		const root_file = await casc.getFile(root_file_id);
		const adt = new ADTLoader(root_file);
		adt.loadRoot();

		const geo = this._build_tile_geometry(adt);
		if (geo.index_count === 0)
			return;

		this._pending_tiles.push(geo);
	}

	_build_merged_vao() {
		const pending = this._pending_tiles;
		this.tile_count = pending.length;

		if (pending.length === 0)
			return;

		let total_floats = 0;
		let total_indices = 0;

		for (const geo of pending) {
			total_floats += geo.vertex_data.length;
			total_indices += geo.index_count;
		}

		const vertex_data = new Float32Array(total_floats);
		const index_data = new Uint32Array(total_indices);

		let vert_write = 0;
		let idx_write = 0;
		let base_vertex = 0;

		this._tile_ranges = [];

		for (const geo of pending) {
			vertex_data.set(geo.vertex_data, vert_write);
			vert_write += geo.vertex_data.length;

			const range_start = idx_write;
			for (let i = 0; i < geo.index_count; i++)
				index_data[idx_write++] = geo.index_data[i] + base_vertex;

			this._tile_ranges.push({
				index_offset: range_start,
				index_count: geo.index_count,
				bounds_min: geo.bounds_min,
				bounds_max: geo.bounds_max
			});

			base_vertex += geo.vertex_data.length / 6;
		}

		this._pending_tiles.length = 0;

		const vao = new VertexArray(this.ctx);
		vao.bind();
		vao.set_vertex_buffer(vertex_data);

		const gl = this.gl;
		const stride = 24;
		vao.set_attribute(0, 3, gl.FLOAT, false, stride, 0);
		vao.set_attribute(1, 3, gl.FLOAT, false, stride, 12);
		vao.set_index_buffer(index_data);

		this.vao = vao;
	}

	_build_tile_geometry(adt) {
		// pre-calculate sizes
		let chunk_count = 0;
		for (let i = 0; i < 256; i++) {
			if (adt.chunks[i]?.vertices)
				chunk_count++;
		}

		const vert_count = chunk_count * 145;
		const vertex_data = new Float32Array(vert_count * 6);
		const index_data = new Uint16Array(chunk_count * 768);

		const tile_min = [Infinity, Infinity, Infinity];
		const tile_max = [-Infinity, -Infinity, -Infinity];

		let vert_offset = 0;
		let chunk_vert_base = 0;
		let idx_offset = 0;

		for (let x = 0; x < 16; x++) {
			for (let y = 0; y < 16; y++) {
				const chunk = adt.chunks[(x * 16) + y];
				if (!chunk?.vertices)
					continue;

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

						// update global bounds
						const b = this.bounds;
						if (vx < b.min[0]) b.min[0] = vx;
						if (vy < b.min[1]) b.min[1] = vy;
						if (vz < b.min[2]) b.min[2] = vz;
						if (vx > b.max[0]) b.max[0] = vx;
						if (vy > b.max[1]) b.max[1] = vy;
						if (vz > b.max[2]) b.max[2] = vz;

						// update tile bounds
						if (vx < tile_min[0]) tile_min[0] = vx;
						if (vy < tile_min[1]) tile_min[1] = vy;
						if (vz < tile_min[2]) tile_min[2] = vz;
						if (vx > tile_max[0]) tile_max[0] = vx;
						if (vy > tile_max[1]) tile_max[1] = vy;
						if (vz > tile_max[2]) tile_max[2] = vz;

						vert_offset++;
						idx++;
					}
				}

				// generate triangle indices for inner vertices
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

					// skip outer row vertices between inner rows
					if (!((j + 1) % 17))
						j += 9;
				}

				chunk_vert_base += 145;
			}
		}

		return {
			vertex_data,
			index_data,
			index_count: idx_offset,
			bounds_min: tile_min,
			bounds_max: tile_max
		};
	}

	get_center() {
		const b = this.bounds;
		return [
			(b.min[0] + b.max[0]) / 2,
			(b.min[1] + b.max[1]) / 2,
			(b.min[2] + b.max[2]) / 2
		];
	}

	render(view_matrix, projection_matrix) {
		if (!this.shader.is_valid() || !this.vao)
			return;

		this.shader.use();
		this.shader.set_uniform_mat4('u_view', false, view_matrix);
		this.shader.set_uniform_mat4('u_projection', false, projection_matrix);

		this._compute_frustum(view_matrix, projection_matrix);

		const gl = this.gl;
		this.vao.bind();

		for (const range of this._tile_ranges) {
			if (this._is_aabb_visible(range.bounds_min, range.bounds_max))
				this.vao.draw(gl.TRIANGLES, range.index_count, range.index_offset);
		}
	}

	_compute_frustum(view, proj) {
		// multiply projection * view (column-major)
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

		// extract 6 frustum planes (Gribb-Hartmann)
		const p = this._frustum_planes;

		// left: row3 + row0
		p[0] = vp[3] + vp[0]; p[1] = vp[7] + vp[4]; p[2] = vp[11] + vp[8]; p[3] = vp[15] + vp[12];
		// right: row3 - row0
		p[4] = vp[3] - vp[0]; p[5] = vp[7] - vp[4]; p[6] = vp[11] - vp[8]; p[7] = vp[15] - vp[12];
		// bottom: row3 + row1
		p[8] = vp[3] + vp[1]; p[9] = vp[7] + vp[5]; p[10] = vp[11] + vp[9]; p[11] = vp[15] + vp[13];
		// top: row3 - row1
		p[12] = vp[3] - vp[1]; p[13] = vp[7] - vp[5]; p[14] = vp[11] - vp[9]; p[15] = vp[15] - vp[13];
		// near: row3 + row2
		p[16] = vp[3] + vp[2]; p[17] = vp[7] + vp[6]; p[18] = vp[11] + vp[10]; p[19] = vp[15] + vp[14];
		// far: row3 - row2
		p[20] = vp[3] - vp[2]; p[21] = vp[7] - vp[6]; p[22] = vp[11] - vp[10]; p[23] = vp[15] - vp[14];

		// normalize each plane
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

	_is_aabb_visible(bmin, bmax) {
		const p = this._frustum_planes;

		for (let i = 0; i < 6; i++) {
			const o = i * 4;
			const a = p[o], b = p[o + 1], c = p[o + 2], d = p[o + 3];

			// p-vertex: corner most aligned with the plane normal
			const px = a >= 0 ? bmax[0] : bmin[0];
			const py = b >= 0 ? bmax[1] : bmin[1];
			const pz = c >= 0 ? bmax[2] : bmin[2];

			if (a * px + b * py + c * pz + d < 0)
				return false;
		}

		return true;
	}

	dispose() {
		if (this.vao) {
			this.vao.dispose();
			this.vao = null;
		}

		if (this.shader) {
			this.shader.dispose();
			this.shader = null;
		}
	}
}

module.exports = TerrainRenderer;

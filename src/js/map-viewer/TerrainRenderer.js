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
		this.tiles = new Map();
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
				this._load_tile(casc, info.root_id, info.index)
					.catch(() => {})
					.then(() => {
						loaded++;
						if (on_progress)
							on_progress(loaded, total);
					})
			));
		}
	}

	async _load_tile(casc, root_file_id, tile_index) {
		const root_file = await casc.getFile(root_file_id);
		const adt = new ADTLoader(root_file);
		adt.loadRoot();

		const geo = this._build_tile_geometry(adt);
		if (geo.index_count === 0)
			return;

		const vao = new VertexArray(this.ctx);
		vao.bind();
		vao.set_vertex_buffer(geo.vertex_data);

		const gl = this.gl;
		const stride = 24;

		// position (location 0)
		vao.set_attribute(0, 3, gl.FLOAT, false, stride, 0);
		// normal (location 1)
		vao.set_attribute(1, 3, gl.FLOAT, false, stride, 12);

		vao.set_index_buffer(geo.index_data);
		this.tiles.set(tile_index, vao);
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
		const indices = [];

		let vert_offset = 0;
		let chunk_vert_base = 0;

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

						// update bounds
						const b = this.bounds;
						if (vx < b.min[0]) b.min[0] = vx;
						if (vy < b.min[1]) b.min[1] = vy;
						if (vz < b.min[2]) b.min[2] = vz;
						if (vx > b.max[0]) b.max[0] = vx;
						if (vy > b.max[1]) b.max[1] = vy;
						if (vz > b.max[2]) b.max[2] = vz;

						vert_offset++;
						idx++;
					}
				}

				// generate triangle indices for inner vertices
				for (let j = 9; j < 145; j++) {
					const ind = chunk_vert_base + j;
					indices.push(ind, ind - 9, ind + 8);
					indices.push(ind, ind - 8, ind - 9);
					indices.push(ind, ind + 9, ind - 8);
					indices.push(ind, ind + 8, ind + 9);

					// skip outer row vertices between inner rows
					if (!((j + 1) % 17))
						j += 9;
				}

				chunk_vert_base += 145;
			}
		}

		return {
			vertex_data,
			index_data: new Uint32Array(indices),
			index_count: indices.length
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
		if (!this.shader.is_valid() || this.tiles.size === 0)
			return;

		this.shader.use();
		this.shader.set_uniform_mat4('u_view', false, view_matrix);
		this.shader.set_uniform_mat4('u_projection', false, projection_matrix);

		const gl = this.gl;
		for (const vao of this.tiles.values()) {
			vao.bind();
			vao.draw(gl.TRIANGLES);
		}
	}

	dispose() {
		for (const vao of this.tiles.values())
			vao.dispose();

		this.tiles.clear();

		if (this.shader) {
			this.shader.dispose();
			this.shader = null;
		}
	}
}

module.exports = TerrainRenderer;

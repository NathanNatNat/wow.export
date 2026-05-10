const core = require('../core');
const constants = require('../constants');
const ADTLoader = require('../3D/loaders/ADTLoader');
const WDTLoader = require('../3D/loaders/WDTLoader');
const Shaders = require('../3D/Shaders');
const VertexArray = require('../3D/gl/VertexArray');
const GLTexture = require('../3D/gl/GLTexture');
const BLPFile = require('../casc/blp');
const TEXLoader = require('../3D/loaders/TEXLoader');
const listfile = require('../casc/listfile');

const MAP_SIZE = constants.GAME.MAP_SIZE;
const TILE_SIZE = constants.GAME.TILE_SIZE;
const CHUNK_SIZE = TILE_SIZE / 16;
const UNIT_SIZE = CHUNK_SIZE / 8;
const UNIT_SIZE_HALF = UNIT_SIZE / 2;

const MAX_CONCURRENT_LOADS = 4;
const UNLOAD_PADDING = 2;
const DEFAULT_RENDER_DISTANCE = 8;
const GRID_Y_BIAS = 0.5;

const ADT_TEX_TILE_REPEAT = 8;
const ADT_TEX_CHUNK_RES = 64;
const ADT_TEX_ATLAS_RES = ADT_TEX_CHUNK_RES * 16;

const MAX_FULL_CONCURRENT_LOADS = 2;
const FULL_UPLOAD_BUDGET = 4;
const DEFAULT_FULL_LOD_DISTANCE = 12;
const ALPHA_UPLOAD_BATCH = 32;

// 256 chunks x 145 verts x 36 bytes (pos3f + normal3f + uv2f + color4ub)
const MAX_TILE_VERTEX_BYTES = 256 * 145 * 36;
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
		this.full_shader = Shaders.create_program(gl_context, 'mpv_terrain_full');
		this.full_legacy_shader = Shaders.create_program(gl_context, 'mpv_terrain_full_legacy');
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
		this._wdt = null;
		this._tex_loader = null;

		this._on_tile_load = null;
		this._on_tile_unload = null;

		this._last_tx = NaN;
		this._last_ty = NaN;
		this._last_cx = NaN;
		this._last_cy = NaN;

		this.light_uniforms = null;
		this.lighting_enabled = true;

		this.fog_enabled = false;
		this.fog_uniforms = null;
		this.camera_pos = new Float32Array([0, 0, 0]);

		this._vao_pool = [];
		this._frustum_planes = new Float32Array(24);
		this._vp_matrix = new Float32Array(16);

		this._grid_vao = null;
		this._grid_vertex_count = 0;
		this._grid_dirty = true;

		this._chunk_grid_vao = null;
		this._chunk_grid_vertex_count = 0;
		this._chunk_grid_dirty = true;

		this._texture_cache = new Map();
		this._full_loading = new Set();
		this._full_load_queue = [];
		this._full_upload_queue = [];
		this.full_lod_distance = DEFAULT_FULL_LOD_DISTANCE;
		this.texture_mode = 'Flat';
		this.render_holes = true;
	}

	get tile_info() {
		return this._tile_info;
	}

	get loaded_tiles() {
		return this._tiles;
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
		this._last_cx = NaN;
		this._last_cy = NaN;
		this._grid_dirty = true;
		this._chunk_grid_dirty = true;
	}

	async init(map_dir) {
		this._casc = core.view.casc;
		this._map_dir = map_dir;
		const prefix = 'world/maps/' + map_dir + '/' + map_dir;

		const wdt_file = await this._casc.getFileByName(prefix + '.wdt');
		const wdt = new WDTLoader(wdt_file);
		wdt.load();

		this._wdt = wdt;

		const has_maid = !!wdt.entries;

		let min_tx = MAP_SIZE, min_ty = MAP_SIZE, max_tx = 0, max_ty = 0;

		for (let x = 0; x < MAP_SIZE; x++) {
			for (let y = 0; y < MAP_SIZE; y++) {
				const idx = (y * MAP_SIZE) + x;
				if (!wdt.tiles[idx])
					continue;

				let root_id, tex0_id, obj0_id;

				if (has_maid) {
					const entry = wdt.entries[idx];
					if (!entry || !entry.rootADT)
						continue;

					root_id = entry.rootADT;
					tex0_id = entry.tex0ADT;
					obj0_id = entry.obj0ADT;
				} else {
					const tile_prefix = prefix + '_' + x + '_' + y;
					root_id = listfile.getByFilename(tile_prefix + '.adt');
					tex0_id = listfile.getByFilename(tile_prefix + '_tex0.adt');
					obj0_id = listfile.getByFilename(tile_prefix + '_obj0.adt');

					if (!root_id)
						continue;
				}

				const key = x + '_' + y;
				this._tile_info.set(key, { root_id, tex0_id, obj0_id, x, y });

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

		if (wdt.texFileDataID > 0) {
			try {
				const tex_file = await this._casc.getFile(wdt.texFileDataID);
				this._tex_loader = new TEXLoader(tex_file);
				this._tex_loader.load();
			} catch (e) {
				// tex blob not available
			}
		}
	}

	update(camera_pos) {
		this.camera_pos[0] = camera_pos[0];
		this.camera_pos[1] = camera_pos[1];
		this.camera_pos[2] = camera_pos[2];

		this._process_uploads();

		const tx = Math.floor(32 - camera_pos[2] / TILE_SIZE);
		const ty = Math.floor(32 - camera_pos[0] / TILE_SIZE);

		if (tx !== this._last_tx || ty !== this._last_ty) {
			this._last_tx = tx;
			this._last_ty = ty;
			this._update_needed_tiles(tx, ty);
			this._grid_dirty = true;
			this._chunk_grid_dirty = true;
		}

		if (this.texture_mode === 'Full') {
			const cx = Math.floor((32 * TILE_SIZE - camera_pos[2]) / CHUNK_SIZE);
			const cy = Math.floor((32 * TILE_SIZE - camera_pos[0]) / CHUNK_SIZE);

			if (cx !== this._last_cx || cy !== this._last_cy) {
				this._last_cx = cx;
				this._last_cy = cy;
				this._update_full_textures(cx, cy);
			}
		}

		this._process_full_uploads();
		this._pump_load_queue();
		this._pump_full_load_queue();
	}

	_update_needed_tiles(center_tx, center_ty) {
		const rd = this.render_distance;
		const ud = rd + UNLOAD_PADDING;

		// unload tiles beyond unload distance
		for (const [key, tile] of this._tiles) {
			if (Math.abs(tile.x - center_tx) > ud || Math.abs(tile.y - center_ty) > ud) {
				if (tile.minimap_tex)
					tile.minimap_tex.dispose();

				if (tile.adt_tex)
					tile.adt_tex.dispose();

				this._unload_full_tile(tile);
				this._release_vao(tile.vao);
				this._chunk_count -= tile.chunk_count;
				this._tiles.delete(key);

				if (this._on_tile_unload)
					this._on_tile_unload(key);
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

			// extract liquid data and chunk positions for LiquidRenderer
			const liquid_chunks = adt.liquidChunks ?? null;
			const chunk_positions = new Array(256);
			for (let i = 0; i < 256; i++)
				chunk_positions[i] = adt.chunks[i]?.position ?? null;

			const [minimap_blp, adt_tex_pixels] = await Promise.all([
				this._load_minimap_blp(info),
				this._load_adt_tex(info)
			]);

			if (!this._loading.has(key))
				return this._pump_load_queue();

			this._upload_queue.push({ key, geo, minimap_blp, adt_tex_pixels, liquid_chunks, chunk_positions });
		} catch (e) {
			this._loading.delete(key);
		}

		this._pump_load_queue();
	}

	async _load_minimap_blp(info) {
		try {
			const px = info.x.toString().padStart(2, '0');
			const py = info.y.toString().padStart(2, '0');
			const blp_path = 'world/minimaps/' + this._map_dir + '/map' + py + '_' + px + '.blp';
			const blp_data = await this._casc.getFileByName(blp_path, false, true);
			if (blp_data)
				return new BLPFile(blp_data);
		} catch {}
		return null;
	}

	async _load_adt_tex(info) {
		if (!this._tex_loader || !info.tex0_id || info.tex0_id <= 0)
			return null;

		try {
			const tex0_file = await this._casc.getFile(info.tex0_id);
			const tex_adt = new ADTLoader(tex0_file);
			tex_adt.loadTex(this._wdt);
			return this._composite_tile(tex_adt);
		} catch {}
		return null;
	}

	_composite_tile(tex_adt) {
		const diffuse_ids = tex_adt.diffuseTextureFileDataIDs;
		if (!diffuse_ids)
			return null;

		const atlas = new Uint8Array(ADT_TEX_ATLAS_RES * ADT_TEX_ATLAS_RES * 4);

		for (let cx = 0; cx < 16; cx++) {
			for (let cy = 0; cy < 16; cy++) {
				const chunk = tex_adt.texChunks[cx * 16 + cy];
				if (!chunk?.layers || chunk.layers.length === 0)
					continue;

				this._composite_chunk(atlas, cx, cy, chunk, diffuse_ids);
			}
		}

		return atlas;
	}

	_composite_chunk(atlas, cx, cy, chunk, diffuse_ids) {
		const layers = chunk.layers;
		const alpha_layers = chunk.alphaLayers;
		const base_px = cy * ADT_TEX_CHUNK_RES;
		const base_py = cx * ADT_TEX_CHUNK_RES;

		for (let py = 0; py < ADT_TEX_CHUNK_RES; py++) {
			for (let px = 0; px < ADT_TEX_CHUNK_RES; px++) {
				const u = (px / ADT_TEX_CHUNK_RES) * ADT_TEX_TILE_REPEAT;
				const v = (py / ADT_TEX_CHUNK_RES) * ADT_TEX_TILE_REPEAT;

				let r = 0, g = 0, b = 0;

				for (let li = 0; li < layers.length; li++) {
					const tex = this._tex_loader.get_texture(diffuse_ids[layers[li].textureId]);
					if (!tex)
						continue;

					const tx = Math.floor(u * tex.width) % tex.width;
					const ty = Math.floor(v * tex.height) % tex.height;
					const ti = (ty * tex.width + tx) * 4;

					if (li === 0) {
						r = tex.pixels[ti];
						g = tex.pixels[ti + 1];
						b = tex.pixels[ti + 2];
					} else {
						const a = alpha_layers?.[li] ? alpha_layers[li][py * 64 + px] / 255 : 0;
						r += (tex.pixels[ti] - r) * a;
						g += (tex.pixels[ti + 1] - g) * a;
						b += (tex.pixels[ti + 2] - b) * a;
					}
				}

				const dst = ((base_py + py) * ADT_TEX_ATLAS_RES + (base_px + px)) * 4;
				atlas[dst] = r;
				atlas[dst + 1] = g;
				atlas[dst + 2] = b;
				atlas[dst + 3] = 255;
			}
		}
	}

	_process_uploads(budget = 1) {
		while (budget-- > 0 && this._upload_queue.length > 0) {
			const { key, geo, minimap_blp, adt_tex_pixels, liquid_chunks, chunk_positions } = this._upload_queue.shift();

			if (!this._loading.has(key))
				continue;

			this._loading.delete(key);

			const tile = this._upload_tile(geo, minimap_blp, adt_tex_pixels);
			this._tiles.set(key, tile);
			this._chunk_count += tile.chunk_count;
			this._grid_dirty = true;
			this._chunk_grid_dirty = true;

			if (this._on_tile_load)
				this._on_tile_load(key, this._tile_info.get(key), liquid_chunks, chunk_positions);
		}
	}

	_upload_tile(geo, minimap_blp, adt_tex_pixels) {
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

		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.holes_ebo);
		gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, geo.index_data_holes);
		const holes_wire_indices = VertexArray.triangles_to_lines(geo.index_data_holes);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.holes_wireframe_ebo);
		gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, holes_wire_indices);

		// restore triangle EBO in VAO state
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.ebo);

		let minimap_tex = null;
		if (minimap_blp) {
			minimap_tex = new GLTexture(this.ctx);
			minimap_tex.set_blp(minimap_blp);
		}

		let adt_tex = null;
		if (adt_tex_pixels) {
			adt_tex = new GLTexture(this.ctx);
			adt_tex.set_rgba(adt_tex_pixels, ADT_TEX_ATLAS_RES, ADT_TEX_ATLAS_RES, { generate_mipmaps: true });
		}

		return {
			vao,
			minimap_tex,
			adt_tex,
			full: null,
			chunk_bounds: geo.chunk_bounds,
			chunk_draw: geo.chunk_draw,
			chunk_draw_holes: geo.chunk_draw_holes,
			chunk_grid_pos: geo.chunk_grid_pos,
			chunk_count: geo.chunk_count,
			x: geo.x,
			y: geo.y,
			bounds_min: geo.bounds_min,
			bounds_max: geo.bounds_max,
			height_data: geo.height_data
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

		const stride = 36;
		gl.enableVertexAttribArray(0);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
		gl.enableVertexAttribArray(1);
		gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 12);
		gl.enableVertexAttribArray(2);
		gl.vertexAttribPointer(2, 2, gl.FLOAT, false, stride, 24);
		gl.enableVertexAttribArray(3);
		gl.vertexAttribPointer(3, 4, gl.UNSIGNED_BYTE, true, stride, 32);

		vao.ebo = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.ebo);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, MAX_TILE_INDEX_BYTES, gl.DYNAMIC_DRAW);
		vao.index_type = gl.UNSIGNED_SHORT;

		vao.wireframe_ebo = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.wireframe_ebo);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, MAX_TILE_WIRE_INDEX_BYTES, gl.DYNAMIC_DRAW);

		vao.holes_ebo = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.holes_ebo);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, MAX_TILE_INDEX_BYTES, gl.DYNAMIC_DRAW);

		vao.holes_wireframe_ebo = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vao.holes_wireframe_ebo);
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
		const vertex_data = new Float32Array(vert_count * 9);
		const vertex_data_u8 = new Uint8Array(vertex_data.buffer);
		const index_data = new Uint16Array(valid_chunks * 768);
		const index_data_holes = new Uint16Array(valid_chunks * 768);

		const chunk_bounds = new Float32Array(valid_chunks * 6);
		const chunk_draw = new Uint32Array(valid_chunks * 2);
		const chunk_draw_holes = new Uint32Array(valid_chunks * 2);
		const chunk_grid_pos = new Uint16Array(valid_chunks * 2);

		const tile_min = [Infinity, Infinity, Infinity];
		const tile_max = [-Infinity, -Infinity, -Infinity];

		// height grid: 129x129 from outer vertices (9 per chunk * 16 chunks + 1 shared edge... use 128+1)
		// world x maps to grid cols, world z maps to grid rows
		// we compute origin from tile coords and step from TILE_SIZE / 128
		const h_cols = 129;
		const h_rows = 129;
		const height_grid = new Float32Array(h_cols * h_rows);
		height_grid.fill(NaN);

		// tile origin in world space (top-left corner of tile)
		const tile_origin_x = (32 - ty) * TILE_SIZE;
		const tile_origin_z = (32 - tx) * TILE_SIZE;
		const h_step_x = -TILE_SIZE / 128;
		const h_step_z = -TILE_SIZE / 128;

		let vert_offset = 0;
		let chunk_vert_base = 0;
		let idx_offset = 0;
		let holes_idx_offset = 0;
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

						const di = vert_offset * 9;
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

						const col_frac = is_short ? (col + 0.5) / 8 : col / 8;
						vertex_data[di + 6] = (y + col_frac) / 16;
						vertex_data[di + 7] = (x + row / 16) / 16;

						// vertex color (MCCV)
						const ci = vert_offset * 36 + 32;
						if (chunk.vertexShading) {
							const shade = chunk.vertexShading[idx];
							vertex_data_u8[ci] = shade.r;
							vertex_data_u8[ci + 1] = shade.g;
							vertex_data_u8[ci + 2] = shade.b;
							vertex_data_u8[ci + 3] = shade.a;
						} else {
							vertex_data_u8[ci] = 127;
							vertex_data_u8[ci + 1] = 127;
							vertex_data_u8[ci + 2] = 127;
							vertex_data_u8[ci + 3] = 255;
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

						// splat outer vertices into height grid
						if (!is_short) {
							const gx = Math.round((vx - tile_origin_x) / h_step_x);
							const gz = Math.round((vz - tile_origin_z) / h_step_z);

							if (gx >= 0 && gx < h_cols && gz >= 0 && gz < h_rows)
								height_grid[gz * h_cols + gx] = vy;
						}

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

				// holes-aware indices
				const holes_chunk_start = holes_idx_offset;
				const holes_high = chunk.holesHighRes;
				const use_high_res = !!(chunk.flags & 0x10000);

				for (let j = 9, xx = 0, yy = 0; j < 145; j++, xx++) {
					if (xx >= 8) {
						xx = 0;
						yy++;
					}

					let is_hole;
					if (use_high_res)
						is_hole = (holes_high[yy] >> xx) & 1;
					else
						is_hole = (chunk.holesLowRes >> ((xx >> 1) + (yy >> 1) * 4)) & 1;

					if (!is_hole) {
						const ind = chunk_vert_base + j;
						index_data_holes[holes_idx_offset++] = ind;
						index_data_holes[holes_idx_offset++] = ind - 9;
						index_data_holes[holes_idx_offset++] = ind + 8;
						index_data_holes[holes_idx_offset++] = ind;
						index_data_holes[holes_idx_offset++] = ind - 8;
						index_data_holes[holes_idx_offset++] = ind - 9;
						index_data_holes[holes_idx_offset++] = ind;
						index_data_holes[holes_idx_offset++] = ind + 9;
						index_data_holes[holes_idx_offset++] = ind - 8;
						index_data_holes[holes_idx_offset++] = ind;
						index_data_holes[holes_idx_offset++] = ind + 8;
						index_data_holes[holes_idx_offset++] = ind + 9;
					}

					if (!((j + 1) % 17))
						j += 9;
				}

				chunk_draw_holes[dw] = holes_chunk_start;
				chunk_draw_holes[dw + 1] = holes_idx_offset - holes_chunk_start;

				chunk_grid_pos[chunk_idx * 2] = x;
				chunk_grid_pos[chunk_idx * 2 + 1] = y;

				chunk_idx++;
				chunk_vert_base += 145;
			}
		}

		return {
			vertex_data,
			index_data,
			index_data_holes,
			chunk_bounds,
			chunk_draw,
			chunk_draw_holes,
			chunk_grid_pos,
			chunk_count: chunk_idx,
			x: tx,
			y: ty,
			bounds_min: tile_min,
			bounds_max: tile_max,
			height_data: {
				grid: height_grid,
				cols: h_cols,
				rows: h_rows,
				origin_x: tile_origin_x,
				origin_z: tile_origin_z,
				step_x: h_step_x,
				step_z: h_step_z
			}
		};
	}

	set_full_lod_distance(val) {
		this.full_lod_distance = val;
		this._last_cx = NaN;
		this._last_cy = NaN;
	}

	set_texture_mode(mode) {
		const prev = this.texture_mode;
		this.texture_mode = mode;

		if (prev === 'Full' && mode !== 'Full')
			this._dispose_all_full();

		// force chunk-level re-evaluation on next frame
		if (mode === 'Full') {
			this._last_cx = NaN;
			this._last_cy = NaN;
		}
	}

	_tile_overlaps_chunk_lod(tile, cam_cx, cam_cy, lod) {
		const tile_min_cx = tile.x * 16;
		const tile_min_cy = tile.y * 16;
		const tile_max_cx = tile_min_cx + 15;
		const tile_max_cy = tile_min_cy + 15;

		return tile_max_cx >= cam_cx - lod && tile_min_cx <= cam_cx + lod
			&& tile_max_cy >= cam_cy - lod && tile_min_cy <= cam_cy + lod;
	}

	_update_full_textures(cam_cx, cam_cy) {
		const lod = this.full_lod_distance;

		// unload tiles fully outside chunk LoD range
		for (const [key, tile] of this._tiles) {
			if (!tile.full)
				continue;

			if (!this._tile_overlaps_chunk_lod(tile, cam_cx, cam_cy, lod))
				this._unload_full_tile(tile);
		}

		// cancel in-flight full loads outside range
		for (const key of this._full_loading) {
			const tile = this._tiles.get(key);
			if (!tile || !this._tile_overlaps_chunk_lod(tile, cam_cx, cam_cy, lod))
				this._full_loading.delete(key);
		}

		// queue tiles overlapping chunk LoD range that need full textures
		const to_load = [];
		for (const [key, tile] of this._tiles) {
			if (tile.full || this._full_loading.has(key))
				continue;

			if (this._tile_overlaps_chunk_lod(tile, cam_cx, cam_cy, lod)) {
				const tile_cx = tile.x * 16 + 8;
				const tile_cy = tile.y * 16 + 8;
				to_load.push({ key, dist: (tile_cx - cam_cx) ** 2 + (tile_cy - cam_cy) ** 2 });
			}
		}

		to_load.sort((a, b) => a.dist - b.dist);
		this._full_load_queue = to_load.map(e => e.key);
	}

	_pump_full_load_queue() {
		if (this._disposed || this.texture_mode !== 'Full')
			return;

		while (this._full_loading.size < MAX_FULL_CONCURRENT_LOADS && this._full_load_queue.length > 0) {
			const key = this._full_load_queue.shift();
			const tile = this._tiles.get(key);

			if (!tile || tile.full || this._full_loading.has(key))
				continue;

			this._start_full_load(key);
		}
	}

	async _start_full_load(key) {
		this._full_loading.add(key);
		const tile = this._tiles.get(key);
		const info = this._tile_info.get(key);

		if (!tile || !info?.tex0_id || info.tex0_id <= 0) {
			this._full_loading.delete(key);
			return;
		}

		try {
			const tex0_file = await this._casc.getFile(info.tex0_id);
			if (!this._full_loading.has(key))
				return;

			const tex_adt = new ADTLoader(tex0_file);
			tex_adt.loadTex(this._wdt);

			const diffuse_ids = tex_adt.diffuseTextureFileDataIDs;
			const height_ids = tex_adt.heightTextureFileDataIDs;
			const tex_params = tex_adt.texParams;
			const has_height = (this._wdt.flags & 0x80) === 0x80;

			if (!diffuse_ids) {
				this._full_loading.delete(key);
				return;
			}

			// collect unique file IDs needed by this tile
			const unique_ids = new Set();
			for (let i = 0; i < tile.chunk_count; i++) {
				const grid_idx = tile.chunk_grid_pos[i * 2] * 16 + tile.chunk_grid_pos[i * 2 + 1];
				const tex_chunk = tex_adt.texChunks[grid_idx];
				if (!tex_chunk?.layers)
					continue;

				for (const layer of tex_chunk.layers) {
					const did = diffuse_ids[layer.textureId];
					if (did > 0)
						unique_ids.add(did);

					if (has_height && height_ids) {
						const hid = height_ids[layer.textureId];
						if (hid > 0)
							unique_ids.add(hid);
					}
				}
			}

			// fetch and parse all textures in parallel (no GPU upload)
			await Promise.all([...unique_ids].map(id => this._cache_texture_data(id)));

			if (!this._full_loading.has(key)) {
				for (const id of unique_ids)
					this._release_texture(id);
				return;
			}

			// build per-chunk metadata and count alpha layers
			let total_alpha_layers = 0;
			const chunk_meta = new Array(tile.chunk_count);

			for (let i = 0; i < tile.chunk_count; i++) {
				const cx = tile.chunk_grid_pos[i * 2];
				const cy = tile.chunk_grid_pos[i * 2 + 1];
				const grid_idx = cx * 16 + cy;
				const tex_chunk = tex_adt.texChunks[grid_idx];

				if (!tex_chunk?.layers || tex_chunk.layers.length === 0) {
					chunk_meta[i] = null;
					continue;
				}

				const layer_count = Math.min(tex_chunk.layers.length, 8);
				const alpha_offset = total_alpha_layers;
				total_alpha_layers += Math.max(0, layer_count - 1);

				// build texture slot mapping for this chunk
				const slot_map = new Map();
				const slot_ids = [];
				const diffuse_slots = new Int32Array(8);
				const height_slots = new Int32Array(8);
				const layer_scales = new Float32Array(8).fill(1);
				const height_scale_arr = new Float32Array(8);
				const height_offset_arr = new Float32Array(8).fill(1);

				for (let li = 0; li < layer_count; li++) {
					const layer = tex_chunk.layers[li];
					const did = diffuse_ids[layer.textureId];
					const hid = (has_height && height_ids) ? height_ids[layer.textureId] : 0;
					const effective_hid = hid > 0 ? hid : did;

					if (did > 0 && !slot_map.has(did)) {
						slot_map.set(did, slot_ids.length);
						slot_ids.push(did);
					}
					diffuse_slots[li] = did > 0 ? slot_map.get(did) : 0;

					if (has_height && effective_hid > 0 && !slot_map.has(effective_hid)) {
						slot_map.set(effective_hid, slot_ids.length);
						slot_ids.push(effective_hid);
					}
					height_slots[li] = (has_height && effective_hid > 0) ? slot_map.get(effective_hid) : diffuse_slots[li];

					if (tex_params?.[layer.textureId]) {
						const params = tex_params[layer.textureId];
						layer_scales[li] = Math.pow(2, (params.flags & 0xF0) >> 4);

						if (has_height) {
							height_scale_arr[li] = params.height;
							height_offset_arr[li] = params.offset;
						}
					}
				}

				chunk_meta[i] = {
					layer_count,
					alpha_offset,
					chunk_x: cx,
					chunk_y: cy,
					slot_ids,
					diffuse_slots,
					height_slots,
					layer_scales,
					height_scales: height_scale_arr,
					height_offsets: height_offset_arr
				};
			}

			// extract alpha layer data for deferred GPU upload
			const alpha_uploads = [];
			if (total_alpha_layers > 0) {
				for (let i = 0; i < tile.chunk_count; i++) {
					const meta = chunk_meta[i];
					if (!meta || meta.layer_count <= 1)
						continue;

					const grid_idx = meta.chunk_x * 16 + meta.chunk_y;
					const alpha_layers = tex_adt.texChunks[grid_idx]?.alphaLayers;

					for (let li = 1; li < meta.layer_count; li++)
						alpha_uploads.push(alpha_layers?.[li] ? new Uint8Array(alpha_layers[li]) : null);
				}
			}

			// collect textures needing parse and/or GPU upload
			const pending_textures = [];
			for (const id of unique_ids) {
				const entry = this._texture_cache.get(id);
				if (entry && !entry.texture)
					pending_textures.push(id);
			}

			this._full_upload_queue.push({
				key,
				pending_textures,
				texture_index: 0,
				alpha_uploads,
				alpha_upload_index: 0,
				alpha_tex: null,
				total_alpha_layers,
				chunk_meta,
				texture_refs: unique_ids
			});
		} catch (e) {
			this._full_loading.delete(key);
		}
	}

	_unload_full_tile(tile) {
		if (!tile.full)
			return;

		if (tile.full.alpha_tex)
			this.gl.deleteTexture(tile.full.alpha_tex);

		if (tile.full.texture_refs) {
			for (const id of tile.full.texture_refs)
				this._release_texture(id);
		}

		tile.full = null;
	}

	_dispose_all_full() {
		for (const tile of this._tiles.values())
			this._unload_full_tile(tile);

		this._full_loading.clear();
		this._full_load_queue.length = 0;

		for (const job of this._full_upload_queue) {
			if (job.alpha_tex)
				this.gl.deleteTexture(job.alpha_tex);
			for (const id of job.texture_refs)
				this._release_texture(id);
		}
		this._full_upload_queue.length = 0;
	}

	async _cache_texture_data(file_data_id) {
		const entry = this._texture_cache.get(file_data_id);
		if (entry) {
			if (entry.promise)
				await entry.promise;

			entry.ref_count++;
			return;
		}

		const placeholder = { raw_data: null, blp: null, texture: null, ref_count: 1, promise: null };
		this._texture_cache.set(file_data_id, placeholder);

		placeholder.promise = this._casc.getFile(file_data_id).then(data => {
			placeholder.raw_data = data;
			placeholder.promise = null;
		}).catch(() => {
			placeholder.promise = null;
		});

		await placeholder.promise;
	}

	_parse_cached_texture(file_data_id) {
		const entry = this._texture_cache.get(file_data_id);
		if (!entry?.raw_data || entry.blp)
			return false;

		entry.blp = new BLPFile(entry.raw_data);
		entry.raw_data = null;
		return true;
	}

	_upload_cached_texture(file_data_id) {
		const entry = this._texture_cache.get(file_data_id);
		if (!entry?.blp || entry.texture)
			return false;

		entry.texture = new GLTexture(this.ctx);
		entry.texture.set_blp(entry.blp, { wrap_s: true, wrap_t: true });
		entry.blp = null;
		return true;
	}

	_process_full_uploads(budget = FULL_UPLOAD_BUDGET) {
		while (budget > 0 && this._full_upload_queue.length > 0) {
			const job = this._full_upload_queue[0];

			if (!this._full_loading.has(job.key)) {
				this._full_upload_queue.shift();
				if (job.alpha_tex)
					this.gl.deleteTexture(job.alpha_tex);
				for (const id of job.texture_refs)
					this._release_texture(id);
				continue;
			}

			// phase 1: parse BLPs from raw data, then upload to GPU
			while (budget > 0 && job.texture_index < job.pending_textures.length) {
				const id = job.pending_textures[job.texture_index];

				if (this._parse_cached_texture(id)) {
					budget--;
					continue;
				}

				if (this._upload_cached_texture(id))
					budget--;

				job.texture_index++;
			}

			if (job.texture_index < job.pending_textures.length)
				break;

			// phase 2: allocate and upload alpha layers
			if (job.total_alpha_layers > 0) {
				const gl = this.gl;

				if (!job.alpha_tex) {
					job.alpha_tex = gl.createTexture();
					gl.bindTexture(gl.TEXTURE_2D_ARRAY, job.alpha_tex);
					gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.R8, 64, 64, job.total_alpha_layers, 0, gl.RED, gl.UNSIGNED_BYTE, null);
					gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
					gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
					gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
					gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
					budget--;

					if (budget <= 0)
						break;
				}

				gl.bindTexture(gl.TEXTURE_2D_ARRAY, job.alpha_tex);

				while (budget > 0 && job.alpha_upload_index < job.alpha_uploads.length) {
					const batch_end = Math.min(job.alpha_upload_index + ALPHA_UPLOAD_BATCH, job.alpha_uploads.length);

					for (let i = job.alpha_upload_index; i < batch_end; i++) {
						if (job.alpha_uploads[i])
							gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, i, 64, 64, 1, gl.RED, gl.UNSIGNED_BYTE, job.alpha_uploads[i]);
					}

					job.alpha_upload_index = batch_end;
					budget--;
				}

				if (job.alpha_upload_index < job.alpha_uploads.length)
					break;

				gl.bindTexture(gl.TEXTURE_2D_ARRAY, job.alpha_tex);
				gl.generateMipmap(gl.TEXTURE_2D_ARRAY);

				if (this.ctx.ext_aniso)
					gl.texParameterf(gl.TEXTURE_2D_ARRAY, this.ctx.ext_aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, this.ctx.max_anisotropy));
			}

			this._finalize_full_upload(job);
			this._full_upload_queue.shift();
			budget--;
		}
	}

	_finalize_full_upload(job) {
		const tile = this._tiles.get(job.key);
		if (!tile) {
			if (job.alpha_tex)
				this.gl.deleteTexture(job.alpha_tex);
			for (const id of job.texture_refs)
				this._release_texture(id);
			this._full_loading.delete(job.key);
			return;
		}

		tile.full = { alpha_tex: job.alpha_tex, chunk_meta: job.chunk_meta, texture_refs: job.texture_refs };
		this._full_loading.delete(job.key);
	}

	_release_texture(file_data_id) {
		const entry = this._texture_cache.get(file_data_id);
		if (!entry)
			return;

		entry.ref_count--;
		if (entry.ref_count <= 0) {
			if (entry.texture)
				entry.texture.dispose();

			this._texture_cache.delete(file_data_id);
		}
	}

	_set_light_uniforms(shader) {
		shader.set_uniform_1i('u_lighting_enabled', this.lighting_enabled ? 1 : 0);

		const lu = this.light_uniforms;
		if (!lu)
			return;

		shader.set_uniform_3fv('u_light_dir', lu.light_dir);
		shader.set_uniform_3fv('u_ambient_color', lu.ambient_color);
		shader.set_uniform_3fv('u_horizon_ambient_color', lu.horizon_ambient_color);
		shader.set_uniform_3fv('u_ground_ambient_color', lu.ground_ambient_color);
		shader.set_uniform_3fv('u_direct_color', lu.direct_color);
	}

	_set_fog_uniforms(shader) {
		shader.set_uniform_3fv('u_camera_pos', this.camera_pos);

		const fog = this.fog_uniforms;
		if (this.fog_enabled && fog) {
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

	render_full(view_matrix, projection_matrix) {
		this._compute_frustum(view_matrix, projection_matrix);

		const gl = this.gl;
		const planes = this._frustum_planes;
		let visible = 0;
		const cam_cx = this._last_cx;
		const cam_cy = this._last_cy;
		const lod = this.full_lod_distance;
		const has_height = (this._wdt?.flags & 0x80) === 0x80;

		// pass 1: tex0 fallback for tiles without full data + out-of-range chunks on full tiles
		if (this.minimap_shader.is_valid()) {
			this.minimap_shader.use();
			this.minimap_shader.set_uniform_mat4('u_view', false, view_matrix);
			this.minimap_shader.set_uniform_mat4('u_projection', false, projection_matrix);
			this.minimap_shader.set_uniform_1i('u_minimap', 0);
			this._set_light_uniforms(this.minimap_shader);
			this._set_fog_uniforms(this.minimap_shader);

			for (const tile of this._tiles.values()) {
				if (!tile.adt_tex || !this._is_tile_visible(tile, planes))
					continue;

				if (!tile.full) {
					// no full data, draw entire tile with tex0
					tile.adt_tex.bind(0);
					tile.vao.bind();
					gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.render_holes ? tile.vao.holes_ebo : tile.vao.ebo);
					visible += this._draw_tile_batched(tile, planes, gl.TRIANGLES, false);
					continue;
				}

				// tile has full data: draw only out-of-range chunks with tex0
				const full_draw = this.render_holes ? tile.chunk_draw_holes : tile.chunk_draw;
				tile.adt_tex.bind(0);
				tile.vao.bind();
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.render_holes ? tile.vao.holes_ebo : tile.vao.ebo);

				let batch_start = -1;
				let batch_count = 0;

				for (let i = 0; i < tile.chunk_count; i++) {
					const bo = i * 6;
					const dw = i * 2;

					const chunk_cx = tile.x * 16 + tile.chunk_grid_pos[i * 2];
					const chunk_cy = tile.y * 16 + tile.chunk_grid_pos[i * 2 + 1];
					const in_lod = Math.abs(chunk_cx - cam_cx) <= lod && Math.abs(chunk_cy - cam_cy) <= lod;

					if (!in_lod && this._is_aabb_visible(tile.chunk_bounds, bo, planes)) {
						const offset = full_draw[dw];
						const idx_count = full_draw[dw + 1];

						if (batch_start === -1) {
							batch_start = offset;
							batch_count = idx_count;
						} else if (offset === batch_start + batch_count) {
							batch_count += idx_count;
						} else {
							gl.drawElements(gl.TRIANGLES, batch_count, gl.UNSIGNED_SHORT, batch_start * 2);
							batch_start = offset;
							batch_count = idx_count;
						}
						visible++;
					} else if (batch_start !== -1) {
						gl.drawElements(gl.TRIANGLES, batch_count, gl.UNSIGNED_SHORT, batch_start * 2);
						batch_start = -1;
					}
				}

				if (batch_start !== -1)
					gl.drawElements(gl.TRIANGLES, batch_count, gl.UNSIGNED_SHORT, batch_start * 2);
			}
		}

		// pass 2: full textured rendering for in-range chunks
		const shader = has_height ? this.full_shader : this.full_legacy_shader;
		if (!shader?.is_valid())
			return visible;

		shader.use();
		shader.set_uniform_mat4('u_view', false, view_matrix);
		shader.set_uniform_mat4('u_projection', false, projection_matrix);
		this._set_light_uniforms(shader);
		this._set_fog_uniforms(shader);

		for (let i = 0; i < 8; i++)
			shader.set_uniform_1i('u_tex' + i, i);
		shader.set_uniform_1i('u_alpha_maps', 8);

		for (const tile of this._tiles.values()) {
			if (!tile.full || !this._is_tile_visible(tile, planes))
				continue;

			if (tile.full.alpha_tex)
				this.ctx.bind_texture(8, tile.full.alpha_tex, gl.TEXTURE_2D_ARRAY);

			const chunk_draw = this.render_holes ? tile.chunk_draw_holes : tile.chunk_draw;
			tile.vao.bind();
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.render_holes ? tile.vao.holes_ebo : tile.vao.ebo);

			for (let i = 0; i < tile.chunk_count; i++) {
				const meta = tile.full.chunk_meta[i];
				if (!meta)
					continue;

				const chunk_cx = tile.x * 16 + meta.chunk_x;
				const chunk_cy = tile.y * 16 + meta.chunk_y;
				if (Math.abs(chunk_cx - cam_cx) > lod || Math.abs(chunk_cy - cam_cy) > lod)
					continue;

				const bo = i * 6;
				if (!this._is_aabb_visible(tile.chunk_bounds, bo, planes))
					continue;

				for (let s = 0; s < meta.slot_ids.length; s++) {
					const entry = this._texture_cache.get(meta.slot_ids[s]);
					if (entry)
						entry.texture.bind(s);
				}

				shader.set_uniform_1i('u_layer_count', meta.layer_count);
				shader.set_uniform_1i('u_alpha_offset', meta.alpha_offset);
				shader.set_uniform_2f('u_chunk_offset', meta.chunk_y, meta.chunk_x);
				shader.set_uniform_1iv('u_diffuse_slot', meta.diffuse_slots);
				shader.set_uniform_1fv('u_layer_scale', meta.layer_scales);

				if (has_height) {
					shader.set_uniform_1iv('u_height_slot', meta.height_slots);
					shader.set_uniform_1fv('u_height_scale', meta.height_scales);
					shader.set_uniform_1fv('u_height_offset', meta.height_offsets);
				}

				const dw = i * 2;
				gl.drawElements(gl.TRIANGLES, chunk_draw[dw + 1], gl.UNSIGNED_SHORT, chunk_draw[dw] * 2);
				visible++;
			}
		}

		return visible;
	}

	render(view_matrix, projection_matrix, terrain_color) {
		if (!this.shader.is_valid() || this._tiles.size === 0)
			return 0;

		this.shader.use();
		this.shader.set_uniform_mat4('u_view', false, view_matrix);
		this.shader.set_uniform_mat4('u_projection', false, projection_matrix);
		this.shader.set_uniform_3fv('u_terrain_color', terrain_color);
		this._set_light_uniforms(this.shader);
		this._set_fog_uniforms(this.shader);

		this._compute_frustum(view_matrix, projection_matrix);

		const gl = this.gl;
		const planes = this._frustum_planes;
		let visible = 0;

		for (const tile of this._tiles.values()) {
			if (!this._is_tile_visible(tile, planes))
				continue;

			tile.vao.bind();
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.render_holes ? tile.vao.holes_ebo : tile.vao.ebo);
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
				gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.render_holes ? tile.vao.holes_ebo : tile.vao.ebo);
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
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.render_holes ? tile.vao.holes_wireframe_ebo : tile.vao.wireframe_ebo);
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
		this._set_light_uniforms(this.minimap_shader);
		this._set_fog_uniforms(this.minimap_shader);

		this._compute_frustum(view_matrix, projection_matrix);

		const gl = this.gl;
		const planes = this._frustum_planes;
		let visible = 0;

		for (const tile of this._tiles.values()) {
			if (!tile.minimap_tex || !this._is_tile_visible(tile, planes))
				continue;

			tile.minimap_tex.bind(0);
			tile.vao.bind();
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.render_holes ? tile.vao.holes_ebo : tile.vao.ebo);
			visible += this._draw_tile_batched(tile, planes, gl.TRIANGLES, false);
		}

		return visible;
	}

	render_adt_tex(view_matrix, projection_matrix) {
		if (!this.minimap_shader.is_valid() || this._tiles.size === 0)
			return 0;

		this.minimap_shader.use();
		this.minimap_shader.set_uniform_mat4('u_view', false, view_matrix);
		this.minimap_shader.set_uniform_mat4('u_projection', false, projection_matrix);
		this.minimap_shader.set_uniform_1i('u_minimap', 0);
		this._set_light_uniforms(this.minimap_shader);
		this._set_fog_uniforms(this.minimap_shader);

		this._compute_frustum(view_matrix, projection_matrix);

		const gl = this.gl;
		const planes = this._frustum_planes;
		let visible = 0;

		for (const tile of this._tiles.values()) {
			if (!tile.adt_tex || !this._is_tile_visible(tile, planes))
				continue;

			tile.adt_tex.bind(0);
			tile.vao.bind();
			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.render_holes ? tile.vao.holes_ebo : tile.vao.ebo);
			visible += this._draw_tile_batched(tile, planes, gl.TRIANGLES, false);
		}

		return visible;
	}

	_draw_tile_batched(tile, planes, mode, is_wireframe) {
		const gl = this.gl;
		const bounds = tile.chunk_bounds;
		const draw = this.render_holes ? tile.chunk_draw_holes : tile.chunk_draw;
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

	get_height_at(world_x, world_z) {
		const tx = Math.floor(32 - world_z / TILE_SIZE);
		const ty = Math.floor(32 - world_x / TILE_SIZE);
		const key = tx + '_' + ty;
		const tile = this._tiles.get(key);

		if (!tile?.height_data)
			return null;

		const hd = tile.height_data;
		const gx = (world_x - hd.origin_x) / hd.step_x;
		const gz = (world_z - hd.origin_z) / hd.step_z;

		const ix = Math.floor(gx);
		const iz = Math.floor(gz);

		if (ix < 0 || ix >= hd.cols - 1 || iz < 0 || iz >= hd.rows - 1)
			return null;

		const fx = gx - ix;
		const fz = gz - iz;

		const h00 = hd.grid[iz * hd.cols + ix];
		const h10 = hd.grid[iz * hd.cols + ix + 1];
		const h01 = hd.grid[(iz + 1) * hd.cols + ix];
		const h11 = hd.grid[(iz + 1) * hd.cols + ix + 1];

		// any NaN means we're over a missing chunk
		if (isNaN(h00) || isNaN(h10) || isNaN(h01) || isNaN(h11))
			return null;

		const h0 = h00 + (h10 - h00) * fx;
		const h1 = h01 + (h11 - h01) * fx;
		return h0 + (h1 - h0) * fz;
	}

	render_grid(view_matrix, projection_matrix, grid_color) {
		if (!this.wire_shader.is_valid())
			return;

		if (this._grid_dirty)
			this._rebuild_grid();

		if (this._grid_vertex_count === 0)
			return;

		this.wire_shader.use();
		this.wire_shader.set_uniform_mat4('u_view', false, view_matrix);
		this.wire_shader.set_uniform_mat4('u_projection', false, projection_matrix);
		this.wire_shader.set_uniform_3fv('u_terrain_color', grid_color);

		this._grid_vao.bind();
		this.gl.drawArrays(this.gl.LINES, 0, this._grid_vertex_count);
	}

	_rebuild_grid() {
		this._grid_dirty = false;

		const rd = this.render_distance;
		const ctx = this._last_tx;
		const cty = this._last_ty;

		if (isNaN(ctx) || isNaN(cty)) {
			this._grid_vertex_count = 0;
			return;
		}

		const min_tx = ctx - rd;
		const max_tx = ctx + rd;
		const min_ty = cty - rd;
		const max_ty = cty + rd;

		// collect tiles: all tile_info entries + 1-cell border, clipped to render distance
		const grid_set = new Set();
		for (const info of this._tile_info.values()) {
			for (let dx = -1; dx <= 1; dx++) {
				for (let dy = -1; dy <= 1; dy++) {
					const nx = info.x + dx;
					const ny = info.y + dy;
					if (nx >= 0 && nx < MAP_SIZE && ny >= 0 && ny < MAP_SIZE &&
						nx >= min_tx && nx <= max_tx && ny >= min_ty && ny <= max_ty)
						grid_set.add(nx + '_' + ny);
				}
			}
		}

		if (grid_set.size === 0) {
			this._grid_vertex_count = 0;
			return;
		}

		// pre-count vertices
		let total_verts = 0;
		for (const key of grid_set) {
			if (this._tiles.get(key)?.height_data)
				total_verts += 4 * 128 * 2;
			else
				total_verts += 4 * 2;
		}

		const positions = new Float32Array(total_verts * 3);
		let vi = 0;

		for (const key of grid_set) {
			const sep = key.indexOf('_');
			const tx = parseInt(key.substring(0, sep));
			const ty = parseInt(key.substring(sep + 1));
			vi = this._grid_write_tile(positions, vi, tx, ty);
		}

		this._grid_vertex_count = vi;

		if (vi === 0)
			return;

		const gl = this.gl;
		if (!this._grid_vao) {
			this._grid_vao = new VertexArray(this.ctx);
			this._grid_vao.bind();
			this._grid_vao.vbo = gl.createBuffer();
			gl.bindBuffer(gl.ARRAY_BUFFER, this._grid_vao.vbo);
			gl.enableVertexAttribArray(0);
			gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
		} else {
			this._grid_vao.bind();
			gl.bindBuffer(gl.ARRAY_BUFFER, this._grid_vao.vbo);
		}

		gl.bufferData(gl.ARRAY_BUFFER, positions.subarray(0, vi * 3), gl.DYNAMIC_DRAW);
	}

	_grid_write_tile(positions, vi, tx, ty) {
		const ox = (32 - ty) * TILE_SIZE;
		const oz = (32 - tx) * TILE_SIZE;
		const ex = ox - TILE_SIZE;
		const ez = oz - TILE_SIZE;

		const tile = this._tiles.get(tx + '_' + ty);

		if (tile?.height_data) {
			const grid = tile.height_data.grid;

			// top edge (row 0): z = oz, x varies
			vi = this._grid_write_h_edge(positions, vi, grid, 0, ox, oz);
			// bottom edge (row 128): z = ez, x varies
			vi = this._grid_write_h_edge(positions, vi, grid, 128 * 129, ox, ez);
			// left edge (col 0): x = ox, z varies
			vi = this._grid_write_v_edge(positions, vi, grid, 0, ox, oz);
			// right edge (col 128): x = ex, z varies
			vi = this._grid_write_v_edge(positions, vi, grid, 128, ex, oz);
		} else {
			// flat edges, skip if adjacent tile is loaded (it draws the shared edge)
			const adj_top = this._tiles.get((tx - 1) + '_' + ty);
			const adj_bot = this._tiles.get((tx + 1) + '_' + ty);
			const adj_left = this._tiles.get(tx + '_' + (ty - 1));
			const adj_right = this._tiles.get(tx + '_' + (ty + 1));

			if (!adj_top?.height_data)
				vi = this._grid_write_flat(positions, vi, ox, oz, ex, oz);

			if (!adj_bot?.height_data)
				vi = this._grid_write_flat(positions, vi, ox, ez, ex, ez);

			if (!adj_left?.height_data)
				vi = this._grid_write_flat(positions, vi, ox, oz, ox, ez);

			if (!adj_right?.height_data)
				vi = this._grid_write_flat(positions, vi, ex, oz, ex, ez);
		}

		return vi;
	}

	_grid_write_h_edge(positions, vi, grid, row_offset, ox, z) {
		const step = -TILE_SIZE / 128;
		for (let i = 0; i < 128; i++) {
			const h0 = grid[row_offset + i];
			const h1 = grid[row_offset + i + 1];
			const pi = vi * 3;
			positions[pi] = ox + i * step;
			positions[pi + 1] = (isNaN(h0) ? 0 : h0) + GRID_Y_BIAS;
			positions[pi + 2] = z;
			positions[pi + 3] = ox + (i + 1) * step;
			positions[pi + 4] = (isNaN(h1) ? 0 : h1) + GRID_Y_BIAS;
			positions[pi + 5] = z;
			vi += 2;
		}
		return vi;
	}

	_grid_write_v_edge(positions, vi, grid, col_offset, x, oz) {
		const step = -TILE_SIZE / 128;
		for (let i = 0; i < 128; i++) {
			const h0 = grid[col_offset + i * 129];
			const h1 = grid[col_offset + (i + 1) * 129];
			const pi = vi * 3;
			positions[pi] = x;
			positions[pi + 1] = (isNaN(h0) ? 0 : h0) + GRID_Y_BIAS;
			positions[pi + 2] = oz + i * step;
			positions[pi + 3] = x;
			positions[pi + 4] = (isNaN(h1) ? 0 : h1) + GRID_Y_BIAS;
			positions[pi + 5] = oz + (i + 1) * step;
			vi += 2;
		}
		return vi;
	}

	_grid_write_flat(positions, vi, x0, z0, x1, z1) {
		const pi = vi * 3;
		positions[pi] = x0;
		positions[pi + 1] = GRID_Y_BIAS;
		positions[pi + 2] = z0;
		positions[pi + 3] = x1;
		positions[pi + 4] = GRID_Y_BIAS;
		positions[pi + 5] = z1;
		return vi + 2;
	}

	render_chunk_grid(view_matrix, projection_matrix, grid_color) {
		if (!this.wire_shader.is_valid())
			return;

		if (this._chunk_grid_dirty)
			this._rebuild_chunk_grid();

		if (this._chunk_grid_vertex_count === 0)
			return;

		this.wire_shader.use();
		this.wire_shader.set_uniform_mat4('u_view', false, view_matrix);
		this.wire_shader.set_uniform_mat4('u_projection', false, projection_matrix);
		this.wire_shader.set_uniform_3fv('u_terrain_color', grid_color);

		this._chunk_grid_vao.bind();
		this.gl.drawArrays(this.gl.LINES, 0, this._chunk_grid_vertex_count);
	}

	_rebuild_chunk_grid() {
		this._chunk_grid_dirty = false;

		let total_verts = 0;
		for (const tile of this._tiles.values()) {
			if (tile.height_data)
				total_verts += 30 * 128 * 2;
		}

		if (total_verts === 0) {
			this._chunk_grid_vertex_count = 0;
			return;
		}

		const positions = new Float32Array(total_verts * 3);
		let vi = 0;

		for (const tile of this._tiles.values()) {
			if (!tile.height_data)
				continue;

			const grid = tile.height_data.grid;
			const ox = (32 - tile.y) * TILE_SIZE;
			const oz = (32 - tile.x) * TILE_SIZE;

			// horizontal chunk boundaries (15 internal rows)
			for (let cr = 1; cr < 16; cr++) {
				const row_offset = cr * 8 * 129;
				const z = oz - cr * CHUNK_SIZE;
				vi = this._grid_write_h_edge(positions, vi, grid, row_offset, ox, z);
			}

			// vertical chunk boundaries (15 internal columns)
			for (let cc = 1; cc < 16; cc++) {
				const col_offset = cc * 8;
				const x = ox - cc * CHUNK_SIZE;
				vi = this._grid_write_v_edge(positions, vi, grid, col_offset, x, oz);
			}
		}

		this._chunk_grid_vertex_count = vi;

		if (vi === 0)
			return;

		const gl = this.gl;
		if (!this._chunk_grid_vao) {
			this._chunk_grid_vao = new VertexArray(this.ctx);
			this._chunk_grid_vao.bind();
			this._chunk_grid_vao.vbo = gl.createBuffer();
			gl.bindBuffer(gl.ARRAY_BUFFER, this._chunk_grid_vao.vbo);
			gl.enableVertexAttribArray(0);
			gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
		} else {
			this._chunk_grid_vao.bind();
			gl.bindBuffer(gl.ARRAY_BUFFER, this._chunk_grid_vao.vbo);
		}

		gl.bufferData(gl.ARRAY_BUFFER, positions.subarray(0, vi * 3), gl.DYNAMIC_DRAW);
	}

	dispose() {
		this._disposed = true;
		this._loading.clear();
		this._load_queue.length = 0;
		this._upload_queue.length = 0;
		this._full_loading.clear();
		this._full_load_queue.length = 0;

		for (const job of this._full_upload_queue) {
			if (job.alpha_tex)
				this.gl.deleteTexture(job.alpha_tex);
			for (const id of job.texture_refs)
				this._release_texture(id);
		}
		this._full_upload_queue.length = 0;

		for (const tile of this._tiles.values()) {
			if (tile.minimap_tex)
				tile.minimap_tex.dispose();

			if (tile.adt_tex)
				tile.adt_tex.dispose();

			this._unload_full_tile(tile);
			tile.vao.dispose();
		}

		this._tiles.clear();

		for (const entry of this._texture_cache.values()) {
			if (entry.texture)
				entry.texture.dispose();
		}
		this._texture_cache.clear();

		for (const vao of this._vao_pool)
			vao.dispose();

		this._vao_pool.length = 0;

		if (this._tex_loader) {
			this._tex_loader.dispose();
			this._tex_loader = null;
		}

		if (this._grid_vao) {
			this._grid_vao.dispose();
			this._grid_vao = null;
		}

		if (this._chunk_grid_vao) {
			this._chunk_grid_vao.dispose();
			this._chunk_grid_vao = null;
		}

		const shaders = ['shader', 'wire_shader', 'minimap_shader', 'full_shader', 'full_legacy_shader'];
		for (const name of shaders) {
			if (this[name]) {
				Shaders.unregister(this[name]);
				this[name].dispose();
				this[name] = null;
			}
		}
	}
}

module.exports = TerrainRenderer;

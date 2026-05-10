const Shaders = require('../3D/Shaders');
const VertexArray = require('../3D/gl/VertexArray');

const RING_SEGMENTS = 24;

// ring elevation angles (radians from horizon toward zenith)
// ring 0: zenith (single vertex)
// rings 1-4: concentric bands
// ring 5: horizon
const RING_ELEVATIONS = [
	1.2217,  // ~70 deg (zenith area)
	0.6981,  // ~40 deg
	0.3491,  // ~20 deg
	0.1396,  // ~8 deg
	0.0349   // ~2 deg (horizon)
];

// color band assignments per ring
// zenith=0, rings 1-5 map to bands 0-4, horizon=5
const RING_BANDS = [0, 1, 2, 3, 4];
const ZENITH_BAND = 0;
const HORIZON_BAND = 5;

class SkyRenderer {
	constructor(gl_ctx) {
		this._ctx = gl_ctx;
		this._shader = null;
		this._vao = null;
		this._index_count = 0;
		this._sky_colors = new Float32Array(18); // 6 colors × 3 components

		this._init();
	}

	_init() {
		this._shader = Shaders.create_program(this._ctx, 'mpv_sky');
		this._build_dome();
	}

	_build_dome() {
		const vertices = [];
		const indices = [];

		// vertex 0: zenith point
		vertices.push(0, 1.0, 0, ZENITH_BAND);
		let vertex_idx = 1;

		// rings 1-5
		for (let r = 0; r < RING_ELEVATIONS.length; r++) {
			const elev = RING_ELEVATIONS[r];
			const y = Math.sin(elev);
			const xz_radius = Math.cos(elev);
			const band = RING_BANDS[r];

			for (let s = 0; s < RING_SEGMENTS; s++) {
				const angle = (s / RING_SEGMENTS) * Math.PI * 2;
				const x = Math.cos(angle) * xz_radius;
				const z = Math.sin(angle) * xz_radius;
				vertices.push(x, y, z, band);
			}

			if (r === 0) {
				// connect zenith to first ring
				for (let s = 0; s < RING_SEGMENTS; s++) {
					const next = (s + 1) % RING_SEGMENTS;
					indices.push(0, vertex_idx + s, vertex_idx + next);
				}
			} else {
				// connect ring r-1 to ring r
				const prev_start = vertex_idx - RING_SEGMENTS;
				for (let s = 0; s < RING_SEGMENTS; s++) {
					const next = (s + 1) % RING_SEGMENTS;
					indices.push(prev_start + s, vertex_idx + s, vertex_idx + next);
					indices.push(prev_start + s, vertex_idx + next, prev_start + next);
				}
			}

			vertex_idx += RING_SEGMENTS;
		}

		// horizon ring (ring 6) - slightly below horizon
		const horizon_start = vertex_idx;
		for (let s = 0; s < RING_SEGMENTS; s++) {
			const angle = (s / RING_SEGMENTS) * Math.PI * 2;
			vertices.push(Math.cos(angle), -0.05, Math.sin(angle), HORIZON_BAND);
		}

		// connect last elevation ring to horizon
		const prev_start = horizon_start - RING_SEGMENTS;
		for (let s = 0; s < RING_SEGMENTS; s++) {
			const next = (s + 1) % RING_SEGMENTS;
			indices.push(prev_start + s, horizon_start + s, horizon_start + next);
			indices.push(prev_start + s, horizon_start + next, prev_start + next);
		}

		const vert_data = new Float32Array(vertices);
		const idx_data = new Uint16Array(indices);

		this._vao = new VertexArray(this._ctx);
		this._vao.bind();
		this._vao.set_vertex_buffer(vert_data);
		this._vao.set_index_buffer(idx_data);

		// position(3f) + band(1f) = 16 bytes stride
		const gl = this._ctx.gl;
		this._vao.set_attribute(0, 4, gl.FLOAT, false, 16, 0);

		this._index_count = idx_data.length;
	}

	set_sky_colors(colors) {
		if (!colors)
			return;

		// colors is array of 6 [r, g, b] arrays
		for (let i = 0; i < 6; i++) {
			const c = colors[i];
			this._sky_colors[i * 3] = c[0];
			this._sky_colors[i * 3 + 1] = c[1];
			this._sky_colors[i * 3 + 2] = c[2];
		}
	}

	render(view_matrix, projection_matrix) {
		const ctx = this._ctx;
		const gl = ctx.gl;
		const shader = this._shader;

		// sky renders behind everything: no depth write, no depth test
		ctx.set_depth_test(false);
		ctx.set_depth_write(false);
		ctx.set_blend(false);
		ctx.set_cull_face(false);

		shader.use();
		shader.set_uniform_mat4('u_view', false, view_matrix);
		shader.set_uniform_mat4('u_projection', false, projection_matrix);
		shader.set_uniform_3fv('u_sky_colors', this._sky_colors);

		this._vao.bind();
		this._vao.draw(gl.TRIANGLES);

		// restore depth state
		ctx.set_depth_test(true);
		ctx.set_depth_write(true);
	}

	dispose() {
		if (this._vao) {
			this._vao.dispose();
			this._vao = null;
		}

		if (this._shader) {
			Shaders.unregister(this._shader);
			this._shader = null;
		}
	}
}

module.exports = SkyRenderer;

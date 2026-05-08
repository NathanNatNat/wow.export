const core = require('../core');
const GLContext = require('../3D/gl/GLContext');
const PerspectiveCamera = require('./PerspectiveCamera');
const FreeCameraControls = require('./FreeCameraControls');
const TerrainRenderer = require('./TerrainRenderer');

module.exports = {
	template: `<div class="map-viewer-screen">
		<canvas ref="canvas" tabindex="0"></canvas>
		<div class="map-viewer-hud">
			<span class="map-viewer-status">{{ status_text }}</span>
			<button class="map-viewer-close" @click="close" title="Close (Esc)">✕</button>
		</div>
	</div>`,

	data() {
		return {
			status_text: 'Initializing...'
		};
	},

	async mounted() {
		this._init_gl();
		this._init_camera();
		this._start_render_loop();

		this._esc_handler = e => {
			if (e.key === 'Escape')
				this.close();
		};
		document.addEventListener('keydown', this._esc_handler);

		await this._load_terrain();

		// focus canvas after load so keyboard controls work immediately
		this.$refs.canvas?.focus();
	},

	beforeUnmount() {
		this._stop_render_loop();
		document.removeEventListener('keydown', this._esc_handler);
		this._cleanup();
	},

	methods: {
		close() {
			core.view.mapViewerActive = false;
		},

		_init_gl() {
			const canvas = this.$refs.canvas;
			const dpr = window.devicePixelRatio || 1;
			canvas.width = canvas.clientWidth * dpr;
			canvas.height = canvas.clientHeight * dpr;

			this._gl_ctx = new GLContext(canvas, {
				antialias: true,
				alpha: false,
				preserveDrawingBuffer: false
			});

			this._gl_ctx.set_viewport(canvas.width, canvas.height);
			this._gl_ctx.set_clear_color(0.08, 0.08, 0.12, 1);
			this._gl_ctx.set_depth_test(true);

			this._resize_handler = () => {
				const dpr = window.devicePixelRatio || 1;
				canvas.width = canvas.clientWidth * dpr;
				canvas.height = canvas.clientHeight * dpr;
				this._gl_ctx.set_viewport(canvas.width, canvas.height);
				this._camera.aspect = canvas.width / canvas.height;
				this._camera.update_projection();
			};
			window.addEventListener('resize', this._resize_handler);
		},

		_init_camera() {
			const canvas = this.$refs.canvas;
			this._camera = new PerspectiveCamera(60, canvas.width / canvas.height, 1, 100000);
			this._controls = new FreeCameraControls(this._camera, canvas);
		},

		_start_render_loop() {
			this._rendering = true;
			this._last_time = performance.now() * 0.001;

			const frame = () => {
				if (!this._rendering)
					return;

				const now = performance.now() * 0.001;
				const dt = Math.min(now - this._last_time, 0.1);
				this._last_time = now;

				this._controls.update(dt);
				this._gl_ctx.clear(true, true);

				if (this._terrain) {
					const visible = this._terrain.render(this._camera.view_matrix, this._camera.projection_matrix);
					this.status_text = this._terrain.tile_count + ' ADT, render (' + visible + '/' + this._terrain._chunk_count + ')';
				}

				requestAnimationFrame(frame);
			};

			requestAnimationFrame(frame);
		},

		_stop_render_loop() {
			this._rendering = false;
		},

		async _load_terrain() {
			const map_dir = core.view.mapViewerMapDir;
			if (!map_dir) {
				this.status_text = 'No map selected';
				return;
			}

			this._terrain = new TerrainRenderer(this._gl_ctx);

			try {
				this.status_text = 'Loading WDT...';

				await this._terrain.load_map(map_dir, (loaded, total) => {
					this.status_text = 'Loading tiles: ' + loaded + '/' + total;
				});

				this._position_camera();
			} catch (e) {
				this.status_text = 'Error: ' + e.message;
			}
		},

		_position_camera() {
			const center = this._terrain.get_center();
			const b = this._terrain.bounds;
			const extent_y = b.max[1] - b.min[1];

			this._camera.position[0] = center[0];
			this._camera.position[1] = center[1] + Math.max(extent_y * 2, 500);
			this._camera.position[2] = center[2];

			this._controls.pitch = -0.6;
			this._controls.yaw = 0;
		},

		_cleanup() {
			window.removeEventListener('resize', this._resize_handler);

			if (this._controls) {
				this._controls.dispose();
				this._controls = null;
			}

			if (this._terrain) {
				this._terrain.dispose();
				this._terrain = null;
			}

			if (this._gl_ctx) {
				this._gl_ctx.dispose();
				this._gl_ctx = null;
			}
		}
	}
};

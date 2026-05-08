const core = require('../core');
const GLContext = require('../3D/gl/GLContext');
const PerspectiveCamera = require('./PerspectiveCamera');
const FreeCameraControls = require('./FreeCameraControls');
const TerrainRenderer = require('./TerrainRenderer');

const SECTIONS = [
	{
		id: 'rendering',
		label: 'Rendering',
		controls: [
			{ type: 'slider', key: 'mapViewerRenderDistance', label: 'Render Distance', min: 1, max: 256, step: 1 }
		]
	},
	{
		id: 'camera',
		label: 'Camera',
		controls: [
			{ type: 'slider', key: 'mapViewerFlySpeed', label: 'Fly Speed', min: 10, max: 2000, step: 10 }
		]
	}
];

module.exports = {
	template: `<div class="map-viewer-screen">
		<canvas ref="canvas" tabindex="0"></canvas>
		<div class="map-viewer-hud">
			<span class="map-viewer-status">{{ status_text }}</span>
			<button class="map-viewer-close" @click="close" title="Close (Esc)">&#x2715;</button>
		</div>
		<div class="mv-panel">
			<div v-for="section in sections" :key="section.id" class="mv-panel-section">
				<div class="mv-panel-header" @click="toggle_section(section.id)">
					<span class="mv-panel-arrow" :class="{ open: open_section === section.id }">&#x25B6;</span>
					{{ section.label }}
				</div>
				<div v-if="open_section === section.id" class="mv-panel-body">
					<div v-for="ctrl in section.controls" :key="ctrl.key" class="mv-panel-control">
						<label class="mv-panel-label">{{ ctrl.label }}</label>
						<div v-if="ctrl.type === 'slider'" class="mv-panel-slider-row">
							<input
								type="range"
								class="mv-panel-slider"
								:min="ctrl.min"
								:max="ctrl.max"
								:step="ctrl.step"
								:value="config[ctrl.key]"
								@input="config[ctrl.key] = Number($event.target.value)"
							/>
							<span class="mv-panel-value">{{ config[ctrl.key] }}</span>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>`,

	data() {
		return {
			status_text: 'Initializing...',
			open_section: null,
			sections: SECTIONS
		};
	},

	computed: {
		config() {
			return core.view.config;
		}
	},

	watch: {
		'config.mapViewerRenderDistance'(val) {
			if (this._terrain)
				this._terrain.set_render_distance(val);
		}
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

		await this._init_terrain();
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

		toggle_section(id) {
			this.open_section = this.open_section === id ? null : id;
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
			this._controls = new FreeCameraControls(this._camera, canvas, () => core.view.config.mapViewerFlySpeed);
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
					this._terrain.update(this._camera.position);
					const visible = this._terrain.render(this._camera.view_matrix, this._camera.projection_matrix);
					const loaded = this._terrain.tile_count;
					const loading = this._terrain.loading_count;
					this.status_text = loaded + ' ADT (' + loading + ' queued), render (' + visible + '/' + this._terrain.chunk_count + ')';
				}

				requestAnimationFrame(frame);
			};

			requestAnimationFrame(frame);
		},

		_stop_render_loop() {
			this._rendering = false;
		},

		async _init_terrain() {
			const map_dir = core.view.mapViewerMapDir;
			if (!map_dir) {
				this.status_text = 'No map selected';
				return;
			}

			const terrain = new TerrainRenderer(this._gl_ctx);

			try {
				this.status_text = 'Loading WDT...';
				await terrain.init(map_dir);
				terrain.set_render_distance(core.view.config.mapViewerRenderDistance);
				this._terrain = terrain;
				this._position_camera();
			} catch (e) {
				this.status_text = 'Error: ' + e.message;
				terrain.dispose();
			}
		},

		_position_camera() {
			const center = this._terrain.map_center;
			this._camera.position[0] = center[0];
			this._camera.position[1] = 500;
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

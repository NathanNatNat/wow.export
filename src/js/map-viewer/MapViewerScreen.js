const core = require('../core');
const GLContext = require('../3D/gl/GLContext');
const PerspectiveCamera = require('./PerspectiveCamera');
const FreeCameraControls = require('./FreeCameraControls');
const TerrainRenderer = require('./TerrainRenderer');

const SECTIONS = [
	{
		id: 'interface',
		label: 'Interface',
		controls: [
			{ type: 'checkbox', key: 'mapViewerShowStats', label: 'Show Technical Stats' }
		]
	},
	{
		id: 'rendering',
		label: 'Rendering',
		controls: [
			{ type: 'slider', key: 'mapViewerRenderDistance', label: 'Render Distance', min: 1, max: 256, step: 1 },
			{ type: 'color', key: 'mapViewerSkyColor', label: 'Sky Colour' },
			{ type: 'color', key: 'mapViewerTerrainColor', label: 'Terrain Colour' }
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

function hex_to_rgb(hex) {
	const r = parseInt(hex.slice(1, 3), 16) / 255;
	const g = parseInt(hex.slice(3, 5), 16) / 255;
	const b = parseInt(hex.slice(5, 7), 16) / 255;
	return new Float32Array([r, g, b]);
}

module.exports = {
	template: `<div class="map-viewer-screen">
		<canvas ref="canvas"></canvas>
		<template v-if="show_ui">
			<div class="map-viewer-hud">
				<span v-if="config.mapViewerShowStats" class="map-viewer-status">{{ status_text }}</span>
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
							<label v-if="ctrl.type === 'checkbox'" class="mv-panel-checkbox-row">
								<input
									type="checkbox"
									:checked="config[ctrl.key]"
									@change="config[ctrl.key] = $event.target.checked"
								/>
								<span class="mv-panel-label">{{ ctrl.label }}</span>
							</label>
							<template v-else-if="ctrl.type === 'slider'">
								<label class="mv-panel-label">{{ ctrl.label }}</label>
							</template>
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
							<div v-if="ctrl.type === 'color'" class="mv-panel-color-row">
								<label class="mv-panel-label">{{ ctrl.label }}</label>
								<div class="mv-panel-color-swatch" :style="{ background: config[ctrl.key] }" @click="open_picker($event)">
									<input
										type="color"
										:value="config[ctrl.key]"
										@input="config[ctrl.key] = $event.target.value"
									/>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
			<div class="mv-shortcuts">
				<span class="mv-shortcut"><kbd>Esc</kbd> Exit Map</span>
				<span class="mv-shortcut"><kbd>Alt+Z</kbd> Hide UI</span>
			</div>
		</template>
	</div>`,

	data() {
		return {
			status_text: 'Initializing...',
			show_ui: true,
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
		},

		'config.mapViewerSkyColor'(val) {
			if (this._gl_ctx) {
				const c = hex_to_rgb(val);
				this._gl_ctx.set_clear_color(c[0], c[1], c[2], 1);
			}
		},

		'config.mapViewerTerrainColor'(val) {
			this._terrain_color = hex_to_rgb(val);
		}
	},

	async mounted() {
		this._init_gl();
		this._init_camera();
		this._start_render_loop();

		this._key_handler = e => {
			if (e.key === 'Escape')
				this.close();
			else if (e.altKey && e.key === 'z')
				this.show_ui = !this.show_ui;
		};
		document.addEventListener('keydown', this._key_handler);

		await this._init_terrain();
	},

	beforeUnmount() {
		this._stop_render_loop();
		document.removeEventListener('keydown', this._key_handler);
		this._cleanup();
	},

	methods: {
		close() {
			core.view.mapViewerActive = false;
		},

		toggle_section(id) {
			this.open_section = this.open_section === id ? null : id;
		},

		open_picker(e) {
			e.currentTarget.querySelector('input[type="color"]').click();
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

			const sky = hex_to_rgb(core.view.config.mapViewerSkyColor);
			this._gl_ctx.set_clear_color(sky[0], sky[1], sky[2], 1);
			this._terrain_color = hex_to_rgb(core.view.config.mapViewerTerrainColor);

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
					const visible = this._terrain.render(this._camera.view_matrix, this._camera.projection_matrix, this._terrain_color);
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

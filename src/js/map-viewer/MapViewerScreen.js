const core = require('../core');
const constants = require('../constants');
const GLContext = require('../3D/gl/GLContext');
const PerspectiveCamera = require('./PerspectiveCamera');
const FreeCameraControls = require('./FreeCameraControls');
const TerrainRenderer = require('./TerrainRenderer');
const Minimap = require('./Minimap');

const TILE_SIZE = constants.GAME.TILE_SIZE;

const SECTIONS = [
	{
		id: 'interface',
		label: 'Interface',
		controls: [
			{ type: 'checkbox', key: 'mapViewerShowStats', label: 'Show Technical Stats' },
			{ type: 'checkbox', key: 'mapViewerShowMinimap', label: 'Show Minimap' }
		]
	},
	{
		id: 'rendering',
		label: 'Rendering',
		controls: [
			{ type: 'slider', key: 'mapViewerRenderDistance', label: 'Render Distance', min: 1, max: 256, step: 1 },
			{ type: 'color', key: 'mapViewerSkyColor', label: 'Sky Colour' }
		]
	},
	{
		id: 'lighting',
		label: 'Lighting',
		controls: [
			{ type: 'slider', key: 'mapViewerSunAzimuth', label: 'Sun Azimuth', min: 0, max: 360, step: 1 },
			{ type: 'slider', key: 'mapViewerSunElevation', label: 'Sun Elevation', min: 0, max: 90, step: 1 },
			{ type: 'slider', key: 'mapViewerSunIntensity', label: 'Sun Intensity', min: 0, max: 100, step: 1 },
			{ type: 'color', key: 'mapViewerSunColor', label: 'Sun Colour' }
		]
	},
	{
		id: 'terrain',
		label: 'Terrain',
		controls: [
			{ type: 'dropdown', data_key: 'texture_mode', label: 'Texture Mode', options: ['Flat', 'Wireframe', 'Minimap'] },
			{ type: 'color', key: 'mapViewerTerrainColor', label: 'Terrain Colour', visible_mode: 'Flat' },
			{ type: 'color', key: 'mapViewerWireframeColor', label: 'Wireframe Colour', visible_mode: 'Wireframe' },
			{ type: 'checkbox', key: 'mapViewerWireframeOcclusion', label: 'Depth Occlusion', visible_mode: 'Wireframe' }
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

function compute_light_dir(azimuth_deg, elevation_deg) {
	const az = azimuth_deg * Math.PI / 180;
	const el = elevation_deg * Math.PI / 180;
	const cos_el = Math.cos(el);
	return new Float32Array([cos_el * Math.sin(az), Math.sin(el), cos_el * Math.cos(az)]);
}

module.exports = {
	template: `<div class="map-viewer-screen">
		<canvas ref="canvas"></canvas>
		<template v-if="show_ui">
			<div class="map-viewer-hud">
				<div v-if="config.mapViewerShowStats" class="map-viewer-hud-stats">
					<span v-if="map_info" class="map-viewer-status">{{ map_info }}</span>
					<span class="map-viewer-status">{{ status_text }}</span>
					<span v-if="coord_text" class="map-viewer-status">{{ coord_text }}</span>
				</div>
			</div>
			<div class="mv-panel">
				<div v-for="section in sections" :key="section.id" class="mv-panel-section">
					<div class="mv-panel-header" @click="toggle_section(section.id)">
						<span class="mv-panel-arrow" :class="{ open: open_section === section.id }">&#x25B6;</span>
						{{ section.label }}
					</div>
					<div v-if="open_section === section.id" class="mv-panel-body">
						<template v-for="ctrl in section.controls" :key="ctrl.key || ctrl.data_key">
							<div v-if="is_ctrl_visible(ctrl)" class="mv-panel-control">
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
								<div v-if="ctrl.type === 'dropdown'" class="mv-panel-dropdown-row">
									<label class="mv-panel-label">{{ ctrl.label }}</label>
									<select class="mv-panel-dropdown" :value="get_ctrl_value(ctrl)" @change="set_ctrl_value(ctrl, $event.target.value); $event.target.blur()">
										<option v-for="opt in ctrl.options" :key="opt" :value="opt">{{ opt }}</option>
									</select>
								</div>
							</div>
						</template>
					</div>
				</div>
			</div>
			<div v-show="config.mapViewerShowMinimap" ref="minimap_container" class="mv-minimap-container"></div>
			<div class="mv-shortcuts">
				<span class="mv-shortcut"><kbd>Esc</kbd> Exit Map</span>
				<span class="mv-shortcut"><kbd>Alt+Z</kbd> Hide UI</span>
			</div>
		</template>
	</div>`,

	data() {
		return {
			status_text: 'Initializing...',
			map_info: null,
			coord_text: null,
			show_ui: true,
			open_section: null,
			sections: SECTIONS,
			texture_mode: 'Flat'
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
		},

		'config.mapViewerWireframeColor'(val) {
			this._wireframe_color = hex_to_rgb(val);
		},

		'config.mapViewerSunAzimuth'() {
			this._update_light_dir();
		},

		'config.mapViewerSunElevation'() {
			this._update_light_dir();
		},

		'config.mapViewerSunIntensity'(val) {
			if (this._terrain)
				this._terrain.sun_intensity = val / 100;
		},

		'config.mapViewerSunColor'(val) {
			if (this._terrain)
				this._terrain.sun_color = hex_to_rgb(val);
		},

		'config.mapViewerShowMinimap'(val) {
			if (val)
				this._init_minimap();
			else
				this._dispose_minimap();
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

		is_ctrl_visible(ctrl) {
			if (!ctrl.visible_mode)
				return true;

			return ctrl.visible_mode === this.texture_mode;
		},

		get_ctrl_value(ctrl) {
			if (ctrl.data_key)
				return this[ctrl.data_key];

			return this.config[ctrl.key];
		},

		set_ctrl_value(ctrl, value) {
			if (ctrl.data_key)
				this[ctrl.data_key] = value;
			else
				this.config[ctrl.key] = value;
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
			this._wireframe_color = hex_to_rgb(core.view.config.mapViewerWireframeColor);

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
					const cam = this._camera.position;
					this._terrain.update(cam);

					// track height above terrain
					const terrain_h = this._terrain.get_height_at(cam[0], cam[2]);
					const ground = terrain_h !== null ? terrain_h : 0;
					this._height_above_terrain = cam[1] - ground;

					let visible;
					if (this.texture_mode === 'Wireframe') {
						const sky = hex_to_rgb(core.view.config.mapViewerSkyColor);
						visible = this._terrain.render_wireframe(this._camera.view_matrix, this._camera.projection_matrix, this._wireframe_color, sky, core.view.config.mapViewerWireframeOcclusion);
					} else if (this.texture_mode === 'Minimap') {
						visible = this._terrain.render_minimap(this._camera.view_matrix, this._camera.projection_matrix);
					} else {
						visible = this._terrain.render(this._camera.view_matrix, this._camera.projection_matrix, this._terrain_color);
					}
					const loaded = this._terrain.tile_count;
					const loading = this._terrain.loading_count;
					this.status_text = loaded + ' ADT (' + loading + ' queued), render (' + visible + '/' + this._terrain.chunk_count + ')';

					const adt_x = Math.floor(32 - cam[2] / TILE_SIZE);
					const adt_y = Math.floor(32 - cam[0] / TILE_SIZE);
					this.coord_text = 'X: ' + cam[0].toFixed(1) + ' Y: ' + cam[2].toFixed(1) + ' Z: ' + cam[1].toFixed(1) + ' [' + adt_x + ', ' + adt_y + ']';

					// update minimap
					if (this._minimap && core.view.config.mapViewerShowMinimap) {
						this._minimap.set_loaded_tiles(this._terrain.loaded_tiles);
						this._minimap.set_camera(cam[0], cam[2]);
						this._minimap.draw();
					}
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

			const map_name = core.view.mapViewerMapName;
			const map_id = core.view.mapViewerMapId;

			if (map_name && map_id != null)
				this.map_info = map_name + ' (' + map_dir + ') [' + map_id + ']';
			else
				this.map_info = map_dir;

			const terrain = new TerrainRenderer(this._gl_ctx);

			try {
				this.status_text = 'Loading WDT...';
				await terrain.init(map_dir);
				terrain.set_render_distance(core.view.config.mapViewerRenderDistance);
				this._terrain = terrain;
				this._apply_sun_settings();
				this._position_camera();
				this._init_minimap();
			} catch (e) {
				this.status_text = 'Error: ' + e.message;
				terrain.dispose();
			}
		},

		_update_light_dir() {
			if (this._terrain)
				this._terrain.light_dir = compute_light_dir(this.config.mapViewerSunAzimuth, this.config.mapViewerSunElevation);
		},

		_apply_sun_settings() {
			if (!this._terrain)
				return;

			this._terrain.light_dir = compute_light_dir(this.config.mapViewerSunAzimuth, this.config.mapViewerSunElevation);
			this._terrain.sun_color = hex_to_rgb(this.config.mapViewerSunColor);
			this._terrain.sun_intensity = this.config.mapViewerSunIntensity / 100;
		},

		_position_camera() {
			const center = this._terrain.map_center;
			this._camera.position[0] = center[0];
			this._camera.position[1] = 500;
			this._camera.position[2] = center[2];
			this._controls.pitch = -0.6;
			this._controls.yaw = 0;
			this._height_above_terrain = 500;
		},

		_init_minimap() {
			if (!this.$refs.minimap_container || !this._terrain || !core.view.config.mapViewerShowMinimap)
				return;

			this._minimap = new Minimap(this.$refs.minimap_container);
			this._minimap.set_tile_info(this._terrain.tile_info);

			this._minimap.set_move_callback((world_x, world_z) => {
				const cam = this._camera.position;

				// compute height above terrain at new position
				const terrain_h = this._terrain.get_height_at(world_x, world_z);
				const ground = terrain_h !== null ? terrain_h : 0;

				cam[0] = world_x;
				cam[1] = ground + this._height_above_terrain;
				cam[2] = world_z;
			});
		},

		_dispose_minimap() {
			if (this._minimap) {
				this._minimap.dispose();
				this._minimap = null;
			}
		},

		_cleanup() {
			window.removeEventListener('resize', this._resize_handler);
			this._dispose_minimap();

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

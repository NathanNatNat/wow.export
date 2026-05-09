const constants = require('../constants');

const MAP_SIZE = constants.GAME.MAP_SIZE;
const TILE_SIZE = constants.GAME.TILE_SIZE;

const MINIMAP_SIZE = 200;

const COLOR_BG = 'rgba(0, 0, 0, 0.6)';
const COLOR_TILE = '#2a6e3f';
const COLOR_LOADED = '#3da85a';
const COLOR_CAMERA = '#ff4444';
const COLOR_BORDER = 'rgba(255, 255, 255, 0.15)';

class Minimap {
	constructor(container) {
		this._canvas = document.createElement('canvas');
		this._canvas.width = MINIMAP_SIZE;
		this._canvas.height = MINIMAP_SIZE;
		this._canvas.className = 'mv-minimap';
		container.appendChild(this._canvas);

		this._ctx = this._canvas.getContext('2d');
		this._tile_info = null;
		this._loaded_tiles = null;
		this._camera_tx = 0;
		this._camera_ty = 0;
		this._dragging = false;

		// visible bounds (auto-fit to tile extent)
		this._min_x = 0;
		this._max_x = MAP_SIZE;
		this._min_y = 0;
		this._max_y = MAP_SIZE;

		this._on_move = null;

		this._bind_events();
	}

	set_tile_info(tile_info) {
		this._tile_info = tile_info;
		this._compute_bounds();
	}

	_compute_bounds() {
		let min_x = MAP_SIZE, min_y = MAP_SIZE, max_x = 0, max_y = 0;

		for (const info of this._tile_info.values()) {
			if (info.x < min_x) min_x = info.x;
			if (info.y < min_y) min_y = info.y;
			if (info.x > max_x) max_x = info.x;
			if (info.y > max_y) max_y = info.y;
		}

		// add 1 padding so the outermost tiles aren't clipped
		const pad = 1;
		this._min_x = Math.max(0, min_x - pad);
		this._min_y = Math.max(0, min_y - pad);
		this._max_x = Math.min(MAP_SIZE, max_x + 1 + pad);
		this._max_y = Math.min(MAP_SIZE, max_y + 1 + pad);

		// ensure square aspect by expanding the smaller axis
		const range_x = this._max_x - this._min_x;
		const range_y = this._max_y - this._min_y;

		if (range_x > range_y) {
			const diff = range_x - range_y;
			this._min_y = Math.max(0, this._min_y - Math.floor(diff / 2));
			this._max_y = this._min_y + range_x;
		} else if (range_y > range_x) {
			const diff = range_y - range_x;
			this._min_x = Math.max(0, this._min_x - Math.floor(diff / 2));
			this._max_x = this._min_x + range_y;
		}
	}

	set_loaded_tiles(loaded_tiles) {
		this._loaded_tiles = loaded_tiles;
	}

	set_camera(world_x, world_z) {
		this._camera_tx = 32 - world_z / TILE_SIZE;
		this._camera_ty = 32 - world_x / TILE_SIZE;
	}

	set_move_callback(fn) {
		this._on_move = fn;
	}

	_bind_events() {
		this._canvas.addEventListener('mousedown', e => {
			e.preventDefault();
			e.stopPropagation();
			this._dragging = true;
			this._handle_click(e);
		});

		this._mouse_move = e => {
			if (!this._dragging)
				return;

			e.preventDefault();
			e.stopPropagation();
			this._handle_click(e);
		};

		this._mouse_up = () => {
			this._dragging = false;
		};

		document.addEventListener('mousemove', this._mouse_move);
		document.addEventListener('mouseup', this._mouse_up);
	}

	_handle_click(e) {
		const rect = this._canvas.getBoundingClientRect();
		const px = e.clientX - rect.left;
		const py = e.clientY - rect.top;

		const range_x = this._max_x - this._min_x;
		const range_y = this._max_y - this._min_y;

		// minimap x-axis = reversed tile_y, y-axis = reversed tile_x
		const tile_y = this._min_y + (px / MINIMAP_SIZE) * range_y;
		const tile_x = this._min_x + (py / MINIMAP_SIZE) * range_x;

		// tile coords to world coords (inverse of camera formula)
		const world_z = (32 - tile_x) * TILE_SIZE;
		const world_x = (32 - tile_y) * TILE_SIZE;

		if (this._on_move)
			this._on_move(world_x, world_z);
	}

	draw() {
		const ctx = this._ctx;
		const size = MINIMAP_SIZE;
		const range_x = this._max_x - this._min_x;
		const range_y = this._max_y - this._min_y;
		const cell_w = size / range_x;
		const cell_h = size / range_y;

		ctx.clearRect(0, 0, size, size);

		// background
		ctx.fillStyle = COLOR_BG;
		ctx.fillRect(0, 0, size, size);

		// draw tiles (minimap x = reversed tile_y, minimap y = reversed tile_x)
		if (this._tile_info) {
			for (const info of this._tile_info.values()) {
				const px = (info.y - this._min_y) * cell_w;
				const py = (info.x - this._min_x) * cell_h;

				const key = info.x + '_' + info.y;
				const is_loaded = this._loaded_tiles && this._loaded_tiles.has(key);

				ctx.fillStyle = is_loaded ? COLOR_LOADED : COLOR_TILE;
				ctx.fillRect(px, py, cell_w, cell_h);
			}
		}

		// camera indicator
		const cam_px = (this._camera_ty - this._min_y) * cell_w;
		const cam_py = (this._camera_tx - this._min_x) * cell_h;

		ctx.fillStyle = COLOR_CAMERA;
		ctx.beginPath();
		ctx.arc(cam_px, cam_py, 4, 0, Math.PI * 2);
		ctx.fill();

		// border
		ctx.strokeStyle = COLOR_BORDER;
		ctx.lineWidth = 1;
		ctx.strokeRect(0.5, 0.5, size - 1, size - 1);
	}

	dispose() {
		document.removeEventListener('mousemove', this._mouse_move);
		document.removeEventListener('mouseup', this._mouse_up);

		if (this._canvas.parentNode)
			this._canvas.parentNode.removeChild(this._canvas);
	}
}

module.exports = Minimap;

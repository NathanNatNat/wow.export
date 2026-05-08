const KEY_W = 87;
const KEY_S = 83;
const KEY_A = 65;
const KEY_D = 68;
const KEY_Q = 81;
const KEY_E = 69;

const MOVE_SPEED = 200;
const MOVE_SPEED_FAST = 800;
const MOVE_SPEED_SLOW = 40;

const MOUSE_SENSITIVITY = 0.003;
const SCROLL_SPEED = 150;
const PITCH_LIMIT = Math.PI / 2 - 0.01;

class FreeCameraControls {
	constructor(camera, dom_element) {
		this.camera = camera;
		this.dom_element = dom_element;

		this.yaw = 0;
		this.pitch = -0.4;

		this._keys = new Set();
		this._is_dragging = false;
		this._last_mouse_x = 0;
		this._last_mouse_y = 0;
		this._shift = false;
		this._alt = false;

		this._bind_events();
	}

	_bind_events() {
		const el = this.dom_element;

		el.addEventListener('mousedown', e => this._on_mouse_down(e));
		el.addEventListener('contextmenu', e => e.preventDefault());
		el.addEventListener('wheel', e => this._on_wheel(e), { passive: false });

		this._on_move = e => this._on_mouse_move(e);
		this._on_up = e => this._on_mouse_up(e);
		this._on_keydown = e => this._on_key_down(e);
		this._on_keyup = e => this._on_key_up(e);

		document.addEventListener('mousemove', this._on_move);
		document.addEventListener('mouseup', this._on_up);
		el.addEventListener('keydown', this._on_keydown);
		el.addEventListener('keyup', this._on_keyup);

		if (el.tabIndex === -1)
			el.tabIndex = 0;
	}

	_on_mouse_down(e) {
		e.preventDefault();
		this.dom_element.focus();
		this._is_dragging = true;
		this._last_mouse_x = e.clientX;
		this._last_mouse_y = e.clientY;
	}

	_on_mouse_move(e) {
		if (!this._is_dragging)
			return;

		const dx = e.clientX - this._last_mouse_x;
		const dy = e.clientY - this._last_mouse_y;
		this._last_mouse_x = e.clientX;
		this._last_mouse_y = e.clientY;

		this.yaw -= dx * MOUSE_SENSITIVITY;
		this.pitch -= dy * MOUSE_SENSITIVITY;
		this.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.pitch));
	}

	_on_mouse_up() {
		this._is_dragging = false;
	}

	_on_wheel(e) {
		e.preventDefault();

		const dir = e.deltaY < 0 ? 1 : -1;
		const cos_pitch = Math.cos(this.pitch);

		this.camera.position[0] += Math.sin(this.yaw) * cos_pitch * dir * SCROLL_SPEED;
		this.camera.position[1] += Math.sin(this.pitch) * dir * SCROLL_SPEED;
		this.camera.position[2] += Math.cos(this.yaw) * cos_pitch * dir * SCROLL_SPEED;
	}

	_on_key_down(e) {
		this._keys.add(e.keyCode);
		this._shift = e.shiftKey;
		this._alt = e.altKey;
	}

	_on_key_up(e) {
		this._keys.delete(e.keyCode);
		this._shift = e.shiftKey;
		this._alt = e.altKey;
	}

	update(dt) {
		// movement
		let speed = MOVE_SPEED;
		if (this._shift)
			speed = MOVE_SPEED_FAST;
		else if (this._alt)
			speed = MOVE_SPEED_SLOW;

		const move = speed * dt;

		// forward/right on XZ plane (no pitch influence for WASD)
		const fwd_x = Math.sin(this.yaw);
		const fwd_z = Math.cos(this.yaw);
		const right_x = Math.cos(this.yaw);
		const right_z = -Math.sin(this.yaw);

		const pos = this.camera.position;

		if (this._keys.has(KEY_W)) {
			pos[0] += fwd_x * move;
			pos[2] += fwd_z * move;
		}

		if (this._keys.has(KEY_S)) {
			pos[0] -= fwd_x * move;
			pos[2] -= fwd_z * move;
		}

		if (this._keys.has(KEY_A)) {
			pos[0] += right_x * move;
			pos[2] += right_z * move;
		}

		if (this._keys.has(KEY_D)) {
			pos[0] -= right_x * move;
			pos[2] -= right_z * move;
		}

		if (this._keys.has(KEY_Q))
			pos[1] += move;

		if (this._keys.has(KEY_E))
			pos[1] -= move;

		// update look target from yaw/pitch
		const cos_pitch = Math.cos(this.pitch);
		this.camera.lookAt(
			pos[0] + Math.sin(this.yaw) * cos_pitch,
			pos[1] + Math.sin(this.pitch),
			pos[2] + Math.cos(this.yaw) * cos_pitch
		);
	}

	dispose() {
		document.removeEventListener('mousemove', this._on_move);
		document.removeEventListener('mouseup', this._on_up);
		this.dom_element.removeEventListener('keydown', this._on_keydown);
		this.dom_element.removeEventListener('keyup', this._on_keyup);
	}
}

module.exports = FreeCameraControls;

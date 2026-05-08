class PerspectiveCamera {
	constructor(fov, aspect, near, far) {
		this.fov = fov;
		this.aspect = aspect;
		this.near = near;
		this.far = far;

		this.position = [0, 0, 0];
		this.target = [0, 0, 0];
		this.up = [0, 1, 0];

		this.view_matrix = new Float32Array(16);
		this.projection_matrix = new Float32Array(16);

		this.update_projection();
	}

	update_projection() {
		const f = 1.0 / Math.tan(this.fov * 0.5 * Math.PI / 180);
		const nf = 1 / (this.near - this.far);

		const m = this.projection_matrix;
		m[0] = f / this.aspect;
		m[1] = 0;
		m[2] = 0;
		m[3] = 0;
		m[4] = 0;
		m[5] = f;
		m[6] = 0;
		m[7] = 0;
		m[8] = 0;
		m[9] = 0;
		m[10] = (this.far + this.near) * nf;
		m[11] = -1;
		m[12] = 0;
		m[13] = 0;
		m[14] = 2 * this.far * this.near * nf;
		m[15] = 0;
	}

	update_view() {
		const px = this.position[0], py = this.position[1], pz = this.position[2];
		const tx = this.target[0], ty = this.target[1], tz = this.target[2];
		const ux = this.up[0], uy = this.up[1], uz = this.up[2];

		// forward (camera looks from position toward target)
		let fx = px - tx, fy = py - ty, fz = pz - tz;
		let fl = Math.sqrt(fx * fx + fy * fy + fz * fz);
		if (fl > 0) { fx /= fl; fy /= fl; fz /= fl; }

		// right = up x forward
		let rx = uy * fz - uz * fy;
		let ry = uz * fx - ux * fz;
		let rz = ux * fy - uy * fx;
		let rl = Math.sqrt(rx * rx + ry * ry + rz * rz);
		if (rl > 0) { rx /= rl; ry /= rl; rz /= rl; }

		// true up = forward x right
		const nux = fy * rz - fz * ry;
		const nuy = fz * rx - fx * rz;
		const nuz = fx * ry - fy * rx;

		const m = this.view_matrix;
		m[0] = rx;
		m[1] = nux;
		m[2] = fx;
		m[3] = 0;
		m[4] = ry;
		m[5] = nuy;
		m[6] = fy;
		m[7] = 0;
		m[8] = rz;
		m[9] = nuz;
		m[10] = fz;
		m[11] = 0;
		m[12] = -(rx * px + ry * py + rz * pz);
		m[13] = -(nux * px + nuy * py + nuz * pz);
		m[14] = -(fx * px + fy * py + fz * pz);
		m[15] = 1;
	}

	lookAt(x, y, z) {
		this.target[0] = x;
		this.target[1] = y;
		this.target[2] = z;
		this.update_view();
	}
}

module.exports = PerspectiveCamera;

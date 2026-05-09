/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const DXT_TYPE_DXT1 = 0;
const DXT_TYPE_DXT3 = 1;
const DXT_TYPE_DXT5 = 2;

function unpack_565(v) {
	return [
		((v >> 11) & 0x1F) * (255 / 31),
		((v >> 5) & 0x3F) * (255 / 63),
		(v & 0x1F) * (255 / 31)
	];
}

/**
 * decode raw dxt data into rgba uint8array
 * @param {Buffer|Uint8Array} data
 * @param {number} width
 * @param {number} height
 * @param {number} dxt_type - 0=DXT1, 1=DXT3, 2=DXT5
 * @returns {Uint8Array}
 */
function decode_dxt(data, width, height, dxt_type) {
	const out = new Uint8Array(width * height * 4);
	const block_bytes = dxt_type === DXT_TYPE_DXT1 ? 8 : 16;
	const blocks_x = Math.max(1, (width + 3) >> 2);
	const blocks_y = Math.max(1, (height + 3) >> 2);

	let pos = 0;

	for (let by = 0; by < blocks_y; by++) {
		for (let bx = 0; bx < blocks_x; bx++) {
			let color_ofs = pos;
			if (dxt_type !== DXT_TYPE_DXT1)
				color_ofs += 8;

			const c0 = data[color_ofs] | (data[color_ofs + 1] << 8);
			const c1 = data[color_ofs + 2] | (data[color_ofs + 3] << 8);

			const r0 = unpack_565(c0);
			const r1 = unpack_565(c1);

			const colors = new Array(16);
			colors[0] = r0[0]; colors[1] = r0[1]; colors[2] = r0[2]; colors[3] = 255;
			colors[4] = r1[0]; colors[5] = r1[1]; colors[6] = r1[2]; colors[7] = 255;

			if (dxt_type === DXT_TYPE_DXT1 && c0 <= c1) {
				colors[8] = (r0[0] + r1[0]) / 2;
				colors[9] = (r0[1] + r1[1]) / 2;
				colors[10] = (r0[2] + r1[2]) / 2;
				colors[11] = 255;
				colors[12] = 0; colors[13] = 0; colors[14] = 0; colors[15] = 0;
			} else {
				colors[8] = (2 * r0[0] + r1[0]) / 3;
				colors[9] = (2 * r0[1] + r1[1]) / 3;
				colors[10] = (2 * r0[2] + r1[2]) / 3;
				colors[11] = 255;
				colors[12] = (r0[0] + 2 * r1[0]) / 3;
				colors[13] = (r0[1] + 2 * r1[1]) / 3;
				colors[14] = (r0[2] + 2 * r1[2]) / 3;
				colors[15] = 255;
			}

			// decode color indices
			const idx_data = new Array(16);
			for (let i = 0; i < 4; i++) {
				const packed = data[color_ofs + 4 + i];
				idx_data[i * 4] = packed & 0x3;
				idx_data[i * 4 + 1] = (packed >> 2) & 0x3;
				idx_data[i * 4 + 2] = (packed >> 4) & 0x3;
				idx_data[i * 4 + 3] = (packed >> 6) & 0x3;
			}

			// decode alpha
			const alpha = new Array(16);
			if (dxt_type === DXT_TYPE_DXT3) {
				for (let i = 0; i < 8; i++) {
					const q = data[pos + i];
					alpha[i * 2] = (q & 0x0F) | ((q & 0x0F) << 4);
					alpha[i * 2 + 1] = (q & 0xF0) | ((q & 0xF0) >> 4);
				}
			} else if (dxt_type === DXT_TYPE_DXT5) {
				const a0 = data[pos];
				const a1 = data[pos + 1];
				const a_lut = new Array(8);
				a_lut[0] = a0;
				a_lut[1] = a1;

				if (a0 <= a1) {
					for (let i = 1; i < 5; i++)
						a_lut[i + 1] = ((5 - i) * a0 + i * a1) / 5;
					a_lut[6] = 0;
					a_lut[7] = 255;
				} else {
					for (let i = 1; i < 7; i++)
						a_lut[i + 1] = ((7 - i) * a0 + i * a1) / 7;
				}

				let a_pos = 2;
				let a_idx = 0;
				for (let i = 0; i < 2; i++) {
					let value = 0;
					for (let j = 0; j < 3; j++)
						value |= data[pos + a_pos++] << (8 * j);
					for (let j = 0; j < 8; j++)
						alpha[a_idx++] = a_lut[(value >> (3 * j)) & 0x7];
				}
			}

			// write pixels
			for (let py = 0; py < 4; py++) {
				for (let px = 0; px < 4; px++) {
					const sx = bx * 4 + px;
					const sy = by * 4 + py;

					if (sx >= width || sy >= height)
						continue;

					const src = idx_data[py * 4 + px] * 4;
					const dst = (sy * width + sx) * 4;
					out[dst] = colors[src];
					out[dst + 1] = colors[src + 1];
					out[dst + 2] = colors[src + 2];

					if (dxt_type === DXT_TYPE_DXT1)
						out[dst + 3] = colors[src + 3];
					else
						out[dst + 3] = alpha[py * 4 + px];
				}
			}

			pos += block_bytes;
		}
	}

	return out;
}

module.exports = { decode_dxt, DXT_TYPE_DXT1, DXT_TYPE_DXT3, DXT_TYPE_DXT5 };

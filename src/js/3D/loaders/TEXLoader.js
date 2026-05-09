/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const { decode_dxt, DXT_TYPE_DXT1, DXT_TYPE_DXT3, DXT_TYPE_DXT5 } = require('./DXTDecoder');

const CHUNK_TXVR = 0x54585652;
const CHUNK_TXBT = 0x54584254;
const CHUNK_TXFN = 0x5458464E;
const CHUNK_TXMD = 0x54584D44;

const DXT_BLOCK_BYTES = [8, 16, 16]; // DXT1, DXT3, DXT5

class TEXLoader {
	constructor(data) {
		this.data = data;
		this.version = 0;
		this.entries = new Map();
		this._base_offset = 0;
		this._decoded_cache = new Map();
	}

	load() {
		const data = this.data;

		while (data.remainingBytes > 0) {
			const chunk_id = data.readUInt32LE();
			const chunk_size = data.readUInt32LE();
			const next_pos = data.offset + chunk_size;

			if (chunk_id === CHUNK_TXVR)
				this.version = data.readUInt32LE();
			else if (chunk_id === CHUNK_TXBT)
				this._parse_txbt(data, chunk_size);

			// txmd_offset is relative to end of TXFN (v0) or TXBT (v1+)
			if (chunk_id === CHUNK_TXBT || chunk_id === CHUNK_TXFN)
				this._base_offset = next_pos;

			data.seek(next_pos);
		}
	}

	_parse_txbt(data, chunk_size) {
		const count = chunk_size / 12;

		for (let i = 0; i < count; i++) {
			const file_id = data.readUInt32LE();
			const txmd_offset = data.readUInt32LE();
			const size_x = data.readUInt8();
			const size_y = data.readUInt8();
			const level_byte = data.readUInt8();
			const fmt_byte = data.readUInt8();

			const num_levels = level_byte & 0x7F;
			const dxt_type = fmt_byte & 0x0F;

			if (file_id > 0) {
				this.entries.set(file_id, {
					txmd_offset,
					size_x,
					size_y,
					num_levels,
					dxt_type
				});
			}
		}
	}

	/**
	 * get decoded rgba pixels for a file data id (mip 0 only)
	 * @param {number} file_id
	 * @returns {object|null} { pixels: Uint8Array, width, height }
	 */
	get_texture(file_id) {
		const cached = this._decoded_cache.get(file_id);
		if (cached)
			return cached;

		const entry = this.entries.get(file_id);
		if (!entry || this._base_offset === 0)
			return null;

		const { size_x, size_y, dxt_type, txmd_offset } = entry;
		if (size_x === 0 || size_y === 0)
			return null;

		const block_bytes = DXT_BLOCK_BYTES[dxt_type] ?? 8;
		const blocks_x = Math.max(1, (size_x + 3) >> 2);
		const blocks_y = Math.max(1, (size_y + 3) >> 2);
		const mip0_size = blocks_x * blocks_y * block_bytes;

		// txmd_offset points to the TXMD chunk header, +8 skips id+size
		const abs_offset = this._base_offset + txmd_offset + 8;
		this.data.seek(abs_offset);

		const raw = this.data.readBuffer(mip0_size, false);
		const pixels = decode_dxt(raw, size_x, size_y, dxt_type);

		const result = { pixels, width: size_x, height: size_y };
		this._decoded_cache.set(file_id, result);
		return result;
	}

	dispose() {
		this._decoded_cache.clear();
		this.entries.clear();
		this.data = null;
	}
}

module.exports = TEXLoader;

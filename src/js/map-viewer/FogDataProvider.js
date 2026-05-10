const db2 = require('../casc/db2');
const constants = require('../constants');

const MAP_COORD_BASE = constants.GAME.MAP_COORD_BASE;
const ZONE_BLEND_DIST = 50.0;
const FOG_UPDATE_THRESHOLD = 32; // world units movement before recalc
const DENSITY_MULT = 0.0005;

class FogDataProvider {
	constructor(map_id) {
		this._map_id = map_id;
		this._loaded = false;
		this._loading = false;

		// db2 data (indexed for fast lookup)
		this._lights = [];        // Light records for this map + global default
		this._default_light = null;
		this._zone_lights = [];   // ZoneLight polygons for this map
		this._light_data_cache = new Map(); // lightParamId -> sorted LightData rows

		// last computed state
		this._last_pos = [NaN, NaN, NaN];
		this._last_time = -1;
		this._fog_uniforms = create_default_uniforms();
		this._sky_colors = create_default_sky_colors();
		this._light_uniforms = create_default_light_uniforms();

		// public state
		this.time_of_day = 720; // noon (0-2880, half-minutes)
	}

	get loaded() {
		return this._loaded;
	}

	get fog_uniforms() {
		return this._fog_uniforms;
	}

	get sky_colors() {
		return this._sky_colors;
	}

	get light_uniforms() {
		return this._light_uniforms;
	}

	async load() {
		if (this._loaded || this._loading || this._map_id == null)
			return;

		this._loading = true;

		try {
			const [light_table, light_data_table, light_params_table, zone_light_table, zone_light_point_table] = await Promise.all([
				db2.preload.Light(),
				db2.preload.LightData(),
				db2.preload.LightParams(),
				db2.preload.ZoneLight(),
				db2.preload.ZoneLightPoint()
			]);

			this._index_lights(light_table);
			this._index_zone_lights(zone_light_table, zone_light_point_table);
			this._light_data_table = light_data_table;
			this._light_params_table = light_params_table;

			this._loaded = true;
		} catch (e) {
			console.error('[FogDataProvider] Failed to load DB2 tables:', e);
		}

		this._loading = false;
	}

	update(viewer_cam) {
		if (!this._loaded)
			return;

		const wow_x = MAP_COORD_BASE - viewer_cam[0];
		const wow_y = MAP_COORD_BASE - viewer_cam[2];
		const wow_z = viewer_cam[1];

		const dx = wow_x - this._last_pos[0];
		const dy = wow_y - this._last_pos[1];
		const dz = wow_z - this._last_pos[2];
		const dist_sq = dx * dx + dy * dy + dz * dz;

		if (dist_sq < FOG_UPDATE_THRESHOLD * FOG_UPDATE_THRESHOLD && this.time_of_day === this._last_time)
			return;

		this._last_pos[0] = wow_x;
		this._last_pos[1] = wow_y;
		this._last_pos[2] = wow_z;
		this._last_time = this.time_of_day;

		this._compute_fog(wow_x, wow_y, wow_z);
	}

	_index_lights(light_table) {
		const rows = light_table.rows;
		for (const [id, row] of rows) {
			const continent_id = row.ContinentID;
			const coords = row.GameCoords;
			const x = coords?.[0] ?? 0;
			const y = coords?.[1] ?? 0;
			const z = coords?.[2] ?? 0;
			const falloff_start = row.GameFalloffStart ?? 0;
			const falloff_end = row.GameFalloffEnd ?? 0;

			const raw_params = row.LightParamsID;
			const light_params = Array.isArray(raw_params) ? [...raw_params] : [raw_params ?? 0];
			while (light_params.length < 8)
				light_params.push(0);

			const is_default = (x === 0 && y === 0 && z === 0);
			const entry = { id, continent_id, x, y, z, falloff_start, falloff_end, light_params, is_default };

			if (is_default) {
				if (continent_id === this._map_id)
					this._default_light = entry;
				else if (continent_id === 0 && !this._default_light)
					this._default_light = entry;
			} else if (continent_id === this._map_id) {
				this._lights.push(entry);
			}
		}
	}

	_index_zone_lights(zone_light_table, zone_light_point_table) {
		const rows = zone_light_table.rows;
		for (const [id, row] of rows) {
			if (row.MapID !== this._map_id)
				continue;

			const points = [];
			const point_rows = zone_light_point_table.getRelationRows(id);
			if (point_rows) {
				// sort by PointOrder
				const sorted = [...point_rows].sort((a, b) => (a.PointOrder ?? 0) - (b.PointOrder ?? 0));
				for (const pt of sorted) {
					const pos = pt.Pos;
					if (Array.isArray(pos))
						points.push([pos[0] ?? 0, pos[1] ?? 0]);
					else
						points.push([0, 0]);
				}
			}

			if (points.length < 3)
				continue;

			// compute AABB
			let min_x = Infinity, max_x = -Infinity;
			let min_y = Infinity, max_y = -Infinity;
			for (const [px, py] of points) {
				min_x = Math.min(min_x, px);
				max_x = Math.max(max_x, px);
				min_y = Math.min(min_y, py);
				max_y = Math.max(max_y, py);
			}

			this._zone_lights.push({
				id,
				light_id: row.LightID ?? 0,
				z_min: row.Zmin ?? -Infinity,
				z_max: row.Zmax ?? Infinity,
				points,
				aabb: { min_x, max_x, min_y, max_y }
			});
		}
	}

	_get_light_data(light_param_id) {
		if (this._light_data_cache.has(light_param_id))
			return this._light_data_cache.get(light_param_id);

		const rows = this._light_data_table.getRelationRows(light_param_id);
		if (!rows || rows.length === 0) {
			this._light_data_cache.set(light_param_id, null);
			return null;
		}

		const sorted = [...rows].sort((a, b) => (a.Time ?? 0) - (b.Time ?? 0));
		this._light_data_cache.set(light_param_id, sorted);
		return sorted;
	}

	_compute_fog(wow_x, wow_y, wow_z) {
		const time = this.time_of_day;
		const blends = this._calculate_light_blends(wow_x, wow_y, wow_z);

		if (blends.length === 0) {
			this._fog_uniforms = create_default_uniforms();
			this._sky_colors = create_default_sky_colors();
			this._light_uniforms = create_default_light_uniforms();
			return;
		}

		// compute fog + sky + lighting result for each blend, then combine
		let result = null;
		let sky_result = null;
		let light_result = null;
		for (const { light_param_id, blend } of blends) {
			if (light_param_id <= 0)
				continue;

			const fog = this._calc_fog_for_param(light_param_id, time);
			if (!fog)
				continue;

			if (!result) {
				result = scale_fog_result(fog, blend);
				sky_result = scale_sky_colors(fog.sky_colors, blend);
				light_result = scale_light_colors(fog.light_colors, blend);
			} else {
				add_scaled_fog_result(result, fog, blend);
				add_scaled_sky_colors(sky_result, fog.sky_colors, blend);
				add_scaled_light_colors(light_result, fog.light_colors, blend);
			}
		}

		if (!result) {
			this._fog_uniforms = create_default_uniforms();
			this._sky_colors = create_default_sky_colors();
			this._light_uniforms = create_default_light_uniforms();
			return;
		}

		this._fog_uniforms = fog_result_to_uniforms(result);
		this._sky_colors = sky_result;
		this._light_uniforms = light_colors_to_uniforms(light_result, time);
	}

	_calculate_light_blends(wow_x, wow_y, wow_z) {
		const blends = [];

		// default light always contributes
		if (this._default_light)
			blends.push({ light_param_id: this._default_light.light_params[0], blend: 1.0 });

		// zone lights (polygon-based)
		for (const zl of this._zone_lights) {
			const z_inside = wow_z >= zl.z_min && wow_z <= zl.z_max;
			const z_dist = z_inside ? -Math.min(wow_z - zl.z_min, zl.z_max - wow_z) : Math.min(Math.abs(wow_z - zl.z_min), Math.abs(wow_z - zl.z_max));

			const inside_poly = point_in_polygon(wow_x, wow_y, zl.points);
			const border_dist = distance_to_polygon_border(wow_x, wow_y, zl.points);
			const poly_dist = inside_poly ? -border_dist : border_dist;

			const blend_dist = Math.max(poly_dist, z_inside ? -Math.min(wow_z - zl.z_min, zl.z_max - wow_z) : z_dist) - ZONE_BLEND_DIST;

			if (blend_dist < 0) {
				const blend_factor = Math.min(Math.max((-blend_dist) / (2 * ZONE_BLEND_DIST), 0), 1);
				const light = this._find_light_by_id(zl.light_id);
				if (light)
					blends.push({ light_param_id: light.light_params[0], blend: blend_factor });
			}
		}

		// position-based lights (radius spheres)
		for (const light of this._lights) {
			const dx = wow_x - light.x;
			const dy = wow_y - light.y;
			const dz = wow_z - light.z;
			const dist_sq = dx * dx + dy * dy + dz * dz;
			const end_sq = light.falloff_end * light.falloff_end;

			if (dist_sq >= end_sq)
				continue;

			const dist = Math.sqrt(dist_sq);
			const start = light.falloff_start;
			const end = light.falloff_end;

			let alpha;
			if (end <= start)
				alpha = 1.0;
			else
				alpha = Math.min(Math.max(((start - dist) / (end - start)) + 1.0, 0), 1);

			if (alpha > 0)
				blends.push({ light_param_id: light.light_params[0], blend: alpha });
		}

		return blends;
	}

	_find_light_by_id(light_id) {
		for (const light of this._lights) {
			if (light.id === light_id)
				return light;
		}

		if (this._default_light && this._default_light.id === light_id)
			return this._default_light;

		return null;
	}

	_calc_fog_for_param(light_param_id, time) {
		const data_rows = this._get_light_data(light_param_id);
		if (!data_rows || data_rows.length === 0)
			return null;

		// find the two time slots bracketing current time
		let idx_a = 0;
		let idx_b = 0;

		for (let i = 0; i < data_rows.length; i++) {
			if ((data_rows[i].Time ?? 0) <= time)
				idx_a = i;
		}

		idx_b = (idx_a + 1) % data_rows.length;

		const row_a = data_rows[idx_a];
		const row_b = data_rows[idx_b];
		const time_a = row_a.Time ?? 0;
		const time_b = row_b.Time ?? 0;

		let blend_t = 0;
		if (time_b !== time_a) {
			if (time_b > time_a)
				blend_t = Math.min(Math.max((time - time_a) / (time_b - time_a), 0), 1);
			else
				// wrap around midnight
				blend_t = Math.min(Math.max((time - time_a) / (2880 - time_a + time_b), 0), 1);
		}

		return interpolate_light_data(row_a, row_b, blend_t);
	}
}

function interpolate_light_data(a, b, t) {
	const lerp = (x, y) => x + (y - x) * t;
	const lerp_color = (field) => int_to_rgb(lerp_int_color(a[field] ?? 0, b[field] ?? 0, t));

	const fog_end = lerp(a.FogEnd ?? 10000, b.FogEnd ?? 10000);
	const fog_scaler = Math.min(Math.max(lerp(a.FogScaler ?? 0, b.FogScaler ?? 0), -1), 1);
	let fog_density = lerp(a.FogDensity ?? 0, b.FogDensity ?? 0);
	const fog_height = Math.max(lerp(a.FogHeight ?? -10000, b.FogHeight ?? -10000), -10000);
	const fog_height_scaler = Math.min(Math.max(lerp(a.FogHeightScaler ?? 0, b.FogHeightScaler ?? 0), -1), 1);
	let fog_height_density = lerp(a.FogHeightDensity ?? 0, b.FogHeightDensity ?? 0);
	const fog_z_scalar = lerp(a.FogZScalar ?? 0, b.FogZScalar ?? 0);
	const main_fog_start_dist = lerp(a.MainFogStartDist ?? 0, b.MainFogStartDist ?? 0);
	const main_fog_end_dist = lerp(a.MainFogEndDist ?? 0, b.MainFogEndDist ?? 0);
	const sun_fog_angle = lerp(a.SunFogAngle ?? 1, b.SunFogAngle ?? 1);
	const end_fog_color_distance = lerp(a.EndFogColorDistance ?? 0, b.EndFogColorDistance ?? 0);
	const sun_fog_strength = lerp(a.SunFogStrength ?? 0, b.SunFogStrength ?? 0);
	const fog_start_offset = lerp(a.FogStartOffset ?? 0, b.FogStartOffset ?? 0);

	const fog_color = lerp_color('SkyFogColor');
	const end_fog_color = lerp_color('EndFogColor');
	const sun_fog_color = lerp_color('SunFogColor');
	const fog_height_color = lerp_color('FogHeightColor');
	const end_fog_height_color = lerp_color('EndFogHeightColor');

	// fog coefficients
	const fog_height_coeff = lerp_coeff(a, b, 'FogHeightCoefficients', t);
	const main_fog_coeff = lerp_coeff(a, b, 'MainFogCoefficients', t);
	const height_density_fog_coeff = lerp_coeff(a, b, 'HeightDensityFogCoeff', t);

	// auto-calculate density if not provided
	if (fog_density <= 0) {
		const far_clamped = Math.min(50000, 700) - 200;
		const difference = fog_end - fog_end * fog_scaler;
		if (difference > far_clamped || far_clamped <= 0)
			fog_density = 1.5;
		else
			fog_density = ((1.0 - (difference / far_clamped)) * 5.5) + 1.5;
	}

	fog_density = Math.max(fog_density, 0.9);

	if (fog_height_scaler === 0)
		fog_height_density = fog_density;

	// determine legacy vs modern fog
	const coeff_mag_main = vec4_length_sq(main_fog_coeff);
	const coeff_mag_height = vec4_length_sq(height_density_fog_coeff);
	const legacy_fog_scalar = (coeff_mag_main > 0.00000012 || coeff_mag_height > 0.00000012) ? 0.0 : 1.0;

	// default height coefficients if zero
	const hc_mag = vec4_length_sq(fog_height_coeff);
	if (hc_mag <= 0.00000012) {
		fog_height_coeff[0] = 0;
		fog_height_coeff[1] = 0;
		fog_height_coeff[2] = 1;
		fog_height_coeff[3] = 0;
	}

	// compute fog start
	const fog_start = Math.min(50000, 3000) * fog_scaler;

	// sun angle blend
	let sun_angle_blend = 1.0;
	if (sun_fog_angle >= 1.0)
		sun_angle_blend = 0.0;

	const sky_colors = [
		lerp_color('SkyTopColor'),
		lerp_color('SkyMiddleColor'),
		lerp_color('SkyBand1Color'),
		lerp_color('SkyBand2Color'),
		lerp_color('SkySmogColor'),
		lerp_color('SkyFogColor')
	];

	const light_colors = {
		ambient: lerp_color('AmbientColor'),
		horizon_ambient: lerp_color('HorizonAmbientColor'),
		ground_ambient: lerp_color('GroundAmbientColor'),
		direct: lerp_color('DirectColor')
	};

	return {
		sky_colors,
		light_colors,
		fog_start,
		fog_end: Math.max(fog_end, 10),
		fog_density,
		fog_height,
		fog_height_scaler,
		fog_height_density,
		fog_z_scalar,
		main_fog_start_dist,
		main_fog_end_dist,
		sun_fog_angle,
		end_fog_color_distance: end_fog_color_distance > 0 ? end_fog_color_distance : 50000,
		sun_fog_strength,
		fog_start_offset,
		fog_color,
		end_fog_color,
		sun_fog_color,
		fog_height_color,
		end_fog_height_color,
		fog_height_coeff,
		main_fog_coeff,
		height_density_fog_coeff,
		legacy_fog_scalar,
		sun_angle_blend
	};
}

function lerp_coeff(a, b, field, t) {
	const result = [0, 0, 0, 0];
	const arr_a = a[field];
	const arr_b = b[field];

	for (let i = 0; i < 4; i++) {
		const va = (Array.isArray(arr_a) ? arr_a[i] : 0) ?? 0;
		const vb = (Array.isArray(arr_b) ? arr_b[i] : 0) ?? 0;
		result[i] = va + (vb - va) * t;
	}
	return result;
}

function vec4_length_sq(v) {
	return v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
}

// wow stores colors as packed BGRA integers
function int_to_rgb(packed) {
	if (typeof packed === 'object')
		return packed;

	const b = (packed & 0xFF) / 255;
	const g = ((packed >> 8) & 0xFF) / 255;
	const r = ((packed >> 16) & 0xFF) / 255;
	return [r, g, b];
}

function lerp_int_color(a, b, t) {
	const ca = int_to_rgb(a);
	const cb = int_to_rgb(b);
	return [
		ca[0] + (cb[0] - ca[0]) * t,
		ca[1] + (cb[1] - ca[1]) * t,
		ca[2] + (cb[2] - ca[2]) * t
	];
}

const FOG_RESULT_SKIP_KEYS = new Set(['sky_colors', 'light_colors']);

function scale_fog_result(fog, scale) {
	const result = {};
	for (const key of Object.keys(fog)) {
		if (FOG_RESULT_SKIP_KEYS.has(key))
			continue;

		const val = fog[key];
		if (Array.isArray(val))
			result[key] = val.map(v => v * scale);
		else if (typeof val === 'number')
			result[key] = val * scale;
		else
			result[key] = val;
	}
	return result;
}

function add_scaled_fog_result(dest, src, scale) {
	for (const key of Object.keys(src)) {
		if (FOG_RESULT_SKIP_KEYS.has(key))
			continue;

		const val = src[key];
		if (Array.isArray(val)) {
			for (let i = 0; i < val.length; i++)
				dest[key][i] += val[i] * scale;
		} else if (typeof val === 'number') {
			dest[key] += val * scale;
		}
	}
}

function fog_result_to_uniforms(fog) {
	return {
		enabled: 1.0,
		density_params: new Float32Array([fog.fog_start, fog.fog_end, fog.fog_density * DENSITY_MULT, 0]),
		height_plane: new Float32Array([0, 1, 0, -fog.fog_height]),
		color_height_rate: new Float32Array([fog.fog_color[0], fog.fog_color[1], fog.fog_color[2], fog.fog_height_scaler]),
		hdensity_end_color: new Float32Array([fog.fog_height_density * DENSITY_MULT, fog.end_fog_color[0], fog.end_fog_color[1], fog.end_fog_color[2]]),
		sun_angle_color: new Float32Array([fog.sun_fog_angle, fog.sun_fog_color[0], fog.sun_fog_color[1], fog.sun_fog_color[2]]),
		hcolor_end_dist: new Float32Array([fog.fog_height_color[0], fog.fog_height_color[1], fog.fog_height_color[2], fog.end_fog_color_distance]),
		sun_pct_str: new Float32Array([fog.sun_angle_blend * fog.sun_fog_strength, fog.sun_fog_strength, 0, 0]),
		sun_dir_z_scalar: new Float32Array([0, 0, 0, fog.fog_z_scalar]), // sun dir set externally
		height_coeff: new Float32Array(fog.fog_height_coeff),
		main_coeff: new Float32Array(fog.main_fog_coeff),
		hdensity_coeff: new Float32Array(fog.height_density_fog_coeff),
		distances: new Float32Array([
			fog.main_fog_end_dist > fog.main_fog_start_dist + 0.001 ? fog.main_fog_end_dist : fog.main_fog_start_dist + 0.001,
			Math.max(fog.main_fog_start_dist, 0),
			fog.legacy_fog_scalar,
			1.0
		]),
		hend_color_offset: new Float32Array([fog.end_fog_height_color[0], fog.end_fog_height_color[1], fog.end_fog_height_color[2], fog.fog_start_offset])
	};
}

function create_default_sky_colors() {
	return [
		[0.12, 0.12, 0.12],
		[0.12, 0.12, 0.12],
		[0.12, 0.12, 0.12],
		[0.12, 0.12, 0.12],
		[0.12, 0.12, 0.12],
		[0.12, 0.12, 0.12]
	];
}

function scale_sky_colors(colors, scale) {
	return colors.map(c => [c[0] * scale, c[1] * scale, c[2] * scale]);
}

function add_scaled_sky_colors(dest, src, scale) {
	for (let i = 0; i < 6; i++) {
		dest[i][0] += src[i][0] * scale;
		dest[i][1] += src[i][1] * scale;
		dest[i][2] += src[i][2] * scale;
	}
}

function create_default_uniforms() {
	return {
		enabled: 0.0,
		density_params: new Float32Array([0, 100000, 0, 0]),
		height_plane: new Float32Array([0, 1, 0, 10000]),
		color_height_rate: new Float32Array([0.5, 0.5, 0.5, 0]),
		hdensity_end_color: new Float32Array([0, 0.5, 0.5, 0.5]),
		sun_angle_color: new Float32Array([1, 0, 0, 0]),
		hcolor_end_dist: new Float32Array([0.5, 0.5, 0.5, 100000]),
		sun_pct_str: new Float32Array([0, 0, 0, 0]),
		sun_dir_z_scalar: new Float32Array([0, 1, 0, 0]),
		height_coeff: new Float32Array([0, 0, 1, 0]),
		main_coeff: new Float32Array([0, 0, 0, 0]),
		hdensity_coeff: new Float32Array([0, 0, 0, 0]),
		distances: new Float32Array([100000, 0, 1, 1]),
		hend_color_offset: new Float32Array([0.5, 0.5, 0.5, 0])
	};
}

// point-in-polygon test (ray casting)
function point_in_polygon(x, y, points) {
	let inside = false;
	const n = points.length;

	for (let i = 0, j = n - 1; i < n; j = i++) {
		const xi = points[i][0], yi = points[i][1];
		const xj = points[j][0], yj = points[j][1];

		if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi))
			inside = !inside;
	}

	return inside;
}

// minimum distance from point to polygon edges
function distance_to_polygon_border(x, y, points) {
	let min_dist = Infinity;
	const n = points.length;

	for (let i = 0; i < n; i++) {
		const j = (i + 1) % n;
		const dist = point_to_segment_dist(x, y, points[i][0], points[i][1], points[j][0], points[j][1]);
		if (dist < min_dist)
			min_dist = dist;
	}

	return min_dist;
}

function point_to_segment_dist(px, py, ax, ay, bx, by) {
	const dx = bx - ax;
	const dy = by - ay;
	const len_sq = dx * dx + dy * dy;

	if (len_sq === 0)
		return Math.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay));

	let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
	t = Math.max(0, Math.min(1, t));

	const proj_x = ax + t * dx;
	const proj_y = ay + t * dy;
	return Math.sqrt((px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y));
}

function scale_light_colors(colors, scale) {
	return {
		ambient: colors.ambient.map(v => v * scale),
		horizon_ambient: colors.horizon_ambient.map(v => v * scale),
		ground_ambient: colors.ground_ambient.map(v => v * scale),
		direct: colors.direct.map(v => v * scale)
	};
}

function add_scaled_light_colors(dest, src, scale) {
	for (const key of ['ambient', 'horizon_ambient', 'ground_ambient', 'direct']) {
		for (let i = 0; i < 3; i++)
			dest[key][i] += src[key][i] * scale;
	}
}

function compute_sun_dir_from_time(time) {
	// time 0=midnight, 720=6am, 1440=noon, 2160=6pm, 2880=midnight
	const day_angle = (time / 2880) * 2 * Math.PI;
	const elevation = -Math.cos(day_angle) * (Math.PI / 2);
	const azimuth = Math.PI * 0.25;
	const cos_el = Math.cos(elevation);
	return new Float32Array([
		cos_el * Math.sin(azimuth),
		Math.sin(elevation),
		cos_el * Math.cos(azimuth)
	]);
}

function light_colors_to_uniforms(colors, time) {
	return {
		light_dir: compute_sun_dir_from_time(time),
		ambient_color: new Float32Array(colors.ambient),
		horizon_ambient_color: new Float32Array(colors.horizon_ambient),
		ground_ambient_color: new Float32Array(colors.ground_ambient),
		direct_color: new Float32Array(colors.direct)
	};
}

function create_default_light_uniforms() {
	return {
		light_dir: new Float32Array([-0.4394, 0.8192, 0.3687]),
		ambient_color: new Float32Array([0.5, 0.5, 0.5]),
		horizon_ambient_color: new Float32Array([0.5, 0.5, 0.5]),
		ground_ambient_color: new Float32Array([0.35, 0.3, 0.25]),
		direct_color: new Float32Array([0.5, 0.475, 0.425])
	};
}

module.exports = FogDataProvider;

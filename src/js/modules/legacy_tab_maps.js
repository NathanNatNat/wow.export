const util = require('util');
const crypto = require('crypto');
const path = require('path');
const core = require('../core');
const log = require('../log');
const constants = require('../constants');
const InstallType = require('../install-type');
const DBCReader = require('../db/DBCReader');
const BufferWrapper = require('../buffer');
const BLPFile = require('../casc/blp');
const WDTLoader = require('../3D/loaders/WDTLoader');
const ADTLoader = require('../3D/loaders/ADTLoader');
const ExportHelper = require('../casc/export-helper');
const OBJWriter = require('../3D/writers/OBJWriter');
const MTLWriter = require('../3D/writers/MTLWriter');
const CSVWriter = require('../3D/writers/CSVWriter');
const PNGWriter = require('../png-writer');
const TiledPNGWriter = require('../tiled-png-writer');
const generics = require('../generics');

const MAP_SIZE = constants.GAME.MAP_SIZE;
const TILE_SIZE = constants.GAME.TILE_SIZE;
const CHUNK_SIZE = TILE_SIZE / 16;
const UNIT_SIZE = CHUNK_SIZE / 8;
const UNIT_SIZE_HALF = UNIT_SIZE / 2;

let selected_map_id = null;
let selected_map_dir = null;
let selected_map_name = null;
let selected_wdt = null;
let minimap_translate = null;

const get_mpq_file = (file_path) => {
	const raw = core.view.mpq.getFile(file_path);
	if (!raw)
		return null;

	return new BufferWrapper(Buffer.from(raw));
};

const load_minimap_translate = () => {
	const trs_paths = [
		'textures\\Minimap\\md5translate.trs',
		'textures\\minimap\\md5translate.trs'
	];

	let raw = null;
	for (const trs_path of trs_paths) {
		raw = core.view.mpq.getFile(trs_path);
		if (raw)
			break;
	}

	if (!raw) {
		log.write('md5translate.trs not found in MPQ');
		return new Map();
	}

	const text = new TextDecoder('utf-8').decode(raw);
	const lines = text.split(/[\r\n]+/);
	const lookup = new Map();

	for (const line of lines) {
		if (line.length === 0 || line.startsWith('dir'))
			continue;

		const tab_idx = line.indexOf('\t');
		if (tab_idx === -1)
			continue;

		const target = line.substring(0, tab_idx).trim().toLowerCase();
		const source = line.substring(tab_idx + 1).trim();

		if (target.length > 0 && source.length > 0)
			lookup.set(target, source);
	}

	log.write('loaded minimap translate table: %d entries', lookup.size);
	return lookup;
};

const parse_map_entry = (entry) => {
	const match = entry.match(/\[(\d+)\]\31([^\31]+)\31\(([^)]+)\)/);
	if (!match)
		throw new Error('unexpected map entry');

	return { id: parseInt(match[1]), name: match[2], dir: match[3] };
};

const load_map_tile = async (x, y, size) => {
	if (!selected_map_dir)
		return false;

	try {
		const padded_x = x.toString().padStart(2, '0');
		const padded_y = y.toString().padStart(2, '0');
		const tile_key = util.format('%s\\map%s_%s.blp', selected_map_dir, padded_x, padded_y);

		let data = null;

		// modern path (world\minimaps\<dir>\mapXX_YY.blp)
		const direct_path = 'world\\minimaps\\' + tile_key;
		data = get_mpq_file(direct_path);

		// md5translate fallback for pre-cata clients
		if (!data && minimap_translate) {
			const hash_file = minimap_translate.get(tile_key);
			if (hash_file)
				data = get_mpq_file('textures\\minimap\\' + hash_file);
		}

		if (!data)
			return false;

		const blp = new BLPFile(data);
		const canvas = blp.toCanvas(0b0111);

		const scale = size / blp.scaledWidth;
		const scaled = document.createElement('canvas');
		scaled.width = size;
		scaled.height = size;

		const ctx = scaled.getContext('2d');
		ctx.scale(scale, scale);
		ctx.drawImage(canvas, 0, 0);

		return ctx.getImageData(0, 0, size, size);
	} catch (e) {
		return false;
	}
};

const load_legacy_adt = (map_dir, tile_y, tile_x) => {
	const adt_path = util.format('world\\maps\\%s\\%s_%d_%d.adt', map_dir, map_dir, tile_y, tile_x);
	const data = get_mpq_file(adt_path);
	if (!data)
		return null;

	const adt = new ADTLoader(data);

	// monolithic pre-cata ADT: load root chunks (geometry)
	adt.loadRoot();

	// rewind and load obj chunks (M2/WMO placements)
	data.seek(0);
	adt.loadObj();

	return adt;
};

const resolve_m2_filename = (adt, model) => {
	if (!adt.m2Names || !adt.m2Offsets)
		return null;

	return adt.m2Names[adt.m2Offsets[model.mmidEntry]] ?? null;
};

const resolve_wmo_filename = (adt, model) => {
	if (!adt.wmoNames || !adt.wmoOffsets)
		return null;

	return adt.wmoNames[adt.wmoOffsets[model.mwidEntry]] ?? null;
};

const export_terrain_obj = async (map_dir, tile_index, dir, config, helper) => {
	const tile_x = tile_index % MAP_SIZE;
	const tile_y = Math.floor(tile_index / MAP_SIZE);
	const tile_id = tile_y + '_' + tile_x;

	const adt = load_legacy_adt(map_dir, tile_y, tile_x);
	if (!adt || !adt.chunks)
		throw new Error('failed to load ADT');

	const obj_out = path.join(dir, 'adt_' + tile_id + '.obj');
	const obj = new OBJWriter(obj_out);
	const mtl = new MTLWriter(path.join(dir, 'adt_' + tile_id + '.mtl'));

	const vertices = [];
	const normals = [];
	const uvs = [];

	const first_chunk = adt.chunks[0];
	const first_chunk_x = first_chunk.position[0];
	const first_chunk_y = first_chunk.position[1];
	const include_holes = config.mapsIncludeHoles;

	let ofs = 0;
	let chunk_id = 0;

	for (let x = 0, mid_x = 0; x < 16; x++) {
		for (let y = 0; y < 16; y++) {
			const indices = [];
			const chunk_index = x * 16 + y;
			const chunk = adt.chunks[chunk_index];

			if (!chunk || !chunk.vertices)
				continue;

			const chunk_x = chunk.position[0];
			const chunk_y = chunk.position[1];
			const chunk_z = chunk.position[2];

			for (let row = 0, idx = 0; row < 17; row++) {
				const is_short = !!(row % 2);
				const col_count = is_short ? 8 : 9;

				for (let col = 0; col < col_count; col++) {
					let vx = chunk_y - (col * UNIT_SIZE);
					let vy = chunk.vertices[idx] + chunk_z;
					let vz = chunk_x - (row * UNIT_SIZE_HALF);

					if (is_short)
						vx -= UNIT_SIZE_HALF;

					vertices.push(vx, vy, vz);

					const normal = chunk.normals[idx];
					normals.push(normal[0] / 127, normal[1] / 127, normal[2] / 127);

					const uv_raw_u = -(vx - first_chunk_x) / TILE_SIZE;
					const uv_raw_v = (vz - first_chunk_y) / TILE_SIZE;
					uvs.push(uv_raw_u, uv_raw_v);

					idx++;
					mid_x++;
				}
			}

			const holes_high_res = chunk.holesHighRes;
			for (let j = 9, xx = 0, yy = 0; j < 145; j++, xx++) {
				if (xx >= 8) {
					xx = 0;
					yy++;
				}

				let is_hole = false;
				if (include_holes) {
					if (!(chunk.flags & 0x10000)) {
						const current = Math.trunc(Math.pow(2, Math.floor(xx / 2) + Math.floor(yy / 2) * 4));
						if (chunk.holesLowRes & current)
							is_hole = true;
					} else {
						if ((holes_high_res[yy] >> xx) & 1)
							is_hole = true;
					}
				}

				if (!is_hole) {
					const ind_ofs = ofs + j;
					indices.push(ind_ofs, ind_ofs - 9, ind_ofs + 8);
					indices.push(ind_ofs, ind_ofs - 8, ind_ofs - 9);
					indices.push(ind_ofs, ind_ofs + 9, ind_ofs - 8);
					indices.push(ind_ofs, ind_ofs + 8, ind_ofs + 9);
				}

				if (!((j + 1) % (9 + 8)))
					j += 9;
			}

			ofs = mid_x;
			obj.addMesh(chunk_id, indices, 'tex_' + tile_id);
			chunk_id++;
		}
	}

	// minimap-based texture
	const padded_x = tile_y.toString().padStart(2, '0');
	const padded_y = tile_x.toString().padStart(2, '0');
	const tile_key = util.format('%s\\map%s_%s.blp', map_dir, padded_x, padded_y);

	let minimap_data = get_mpq_file('world\\minimaps\\' + tile_key);
	if (!minimap_data && minimap_translate) {
		const hash_file = minimap_translate.get(tile_key);
		if (hash_file)
			minimap_data = get_mpq_file('textures\\minimap\\' + hash_file);
	}

	if (minimap_data) {
		const tex_file = 'tex_' + tile_id + '.png';
		const tex_out_path = path.join(dir, tex_file);

		if (config.overwriteFiles || !await generics.fileExists(tex_out_path)) {
			const blp = new BLPFile(minimap_data);
			await blp.saveToPNG(tex_out_path);
		}

		mtl.addMaterial('tex_' + tile_id, tex_file);
	}

	obj.setVertArray(vertices);
	obj.setNormalArray(normals);
	obj.addUVArray(uvs);

	if (!mtl.isEmpty)
		obj.setMaterialLibrary(path.basename(mtl.out));

	await obj.write(config.overwriteFiles);
	await mtl.write(config.overwriteFiles);

	// export M2/WMO placements
	if (config.mapsIncludeWMO || config.mapsIncludeM2)
		await export_placements(adt, map_dir, tile_id, dir, config, helper);

	return obj_out;
};

const export_placements = async (adt, map_dir, tile_id, dir, config, helper) => {
	const csv_path = path.join(dir, 'adt_' + tile_id + '_ModelPlacementInformation.csv');
	if (!config.overwriteFiles && await generics.fileExists(csv_path))
		return;

	const csv = new CSVWriter(csv_path);
	csv.addField('ModelFile', 'PositionX', 'PositionY', 'PositionZ', 'RotationX', 'RotationY', 'RotationZ', 'RotationW', 'ScaleFactor', 'ModelId', 'Type', 'SourceFile');

	const exported_files = new Set();
	const use_posix = config.pathFormat === 'posix';

	if (config.mapsIncludeM2 && adt.models) {
		helper?.setCurrentTaskName('Tile ' + tile_id + ', doodads');
		helper?.setCurrentTaskMax(adt.models.length);

		for (let i = 0; i < adt.models.length; i++) {
			helper?.setCurrentTaskValue(i);

			const model = adt.models[i];
			const filename = resolve_m2_filename(adt, model);
			if (!filename)
				continue;

			// extract raw M2 file
			if (!exported_files.has(filename)) {
				const raw = core.view.mpq.getFile(filename);
				if (raw) {
					const out_name = path.basename(filename);
					const out_path = path.join(dir, out_name);

					if (config.overwriteFiles || !await generics.fileExists(out_path)) {
						const buf = new BufferWrapper(Buffer.from(raw));
						await buf.writeToFile(out_path);
					}

					exported_files.add(filename);
				}
			}

			let model_file = path.basename(filename);
			if (use_posix)
				model_file = ExportHelper.win32ToPosix(model_file);

			csv.addRow({
				ModelFile: model_file,
				PositionX: model.position[0],
				PositionY: model.position[1],
				PositionZ: model.position[2],
				RotationX: model.rotation[0],
				RotationY: model.rotation[1],
				RotationZ: model.rotation[2],
				RotationW: 0,
				ScaleFactor: model.scale / 1024,
				ModelId: model.uniqueId,
				Type: 'm2',
				SourceFile: filename
			});
		}
	}

	if (config.mapsIncludeWMO && adt.worldModels) {
		helper?.setCurrentTaskName('Tile ' + tile_id + ', WMOs');
		helper?.setCurrentTaskMax(adt.worldModels.length);

		for (let i = 0; i < adt.worldModels.length; i++) {
			helper?.setCurrentTaskValue(i);

			const model = adt.worldModels[i];
			const filename = resolve_wmo_filename(adt, model);
			if (!filename)
				continue;

			// extract raw WMO root file
			if (!exported_files.has(filename)) {
				const raw = core.view.mpq.getFile(filename);
				if (raw) {
					const out_name = path.basename(filename);
					const out_path = path.join(dir, out_name);

					if (config.overwriteFiles || !await generics.fileExists(out_path)) {
						const buf = new BufferWrapper(Buffer.from(raw));
						await buf.writeToFile(out_path);
					}

					exported_files.add(filename);
				}
			}

			let model_file = path.basename(filename);
			if (use_posix)
				model_file = ExportHelper.win32ToPosix(model_file);

			csv.addRow({
				ModelFile: model_file,
				PositionX: model.position[0],
				PositionY: model.position[1],
				PositionZ: model.position[2],
				RotationX: model.rotation[0],
				RotationY: model.rotation[1],
				RotationZ: model.rotation[2],
				RotationW: 0,
				ScaleFactor: model.scale / 1024,
				ModelId: model.uniqueId,
				Type: 'wmo',
				SourceFile: filename
			});
		}
	}

	await csv.write();
};

module.exports = {
	register() {
		this.registerNavButton('Maps', 'map.svg', InstallType.MPQ);
	},

	template: `
		<div class="tab list-tab" id="tab-maps">
			<div class="map-placeholder"></div>
			<div class="list-container" id="maps-list-container">
				<component :is="$components.ListboxMaps" id="listbox-maps" class="listbox-icons" v-model:selection="$core.view.selectionMaps" :items="$core.view.mapViewerMaps" :filter="$core.view.userInputFilterMaps" :expansion-filter="-1" :keyinput="true" :single="true" :regex="$core.view.config.regexFilters" :copymode="$core.view.config.copyMode" :pasteselection="$core.view.config.pasteSelection" :copytrimwhitespace="$core.view.config.removePathSpacesCopy" :includefilecount="true" unittype="map" persistscrollkey="maps" @contextmenu="handle_map_context"></component>
				<component :is="$components.ContextMenu" :node="$core.view.contextMenus.nodeMap" v-slot:default="context" @close="$core.view.contextMenus.nodeMap = null">
					<span @click.self="copy_map_names(context.node.selection)">Copy map name{{ context.node.count > 1 ? 's' : '' }}</span>
					<span @click.self="copy_map_internal_names(context.node.selection)">Copy internal name{{ context.node.count > 1 ? 's' : '' }}</span>
					<span @click.self="copy_map_ids(context.node.selection)">Copy map ID{{ context.node.count > 1 ? 's' : '' }}</span>
					<span @click.self="copy_map_export_paths(context.node.selection)">Copy export path{{ context.node.count > 1 ? 's' : '' }}</span>
					<span @click.self="open_map_export_directory(context.node.selection)">Open export directory</span>
				</component>
			</div>
			<div class="filter">
				<div class="regex-info" v-if="$core.view.config.regexFilters" :title="$core.view.regexTooltip">Regex Enabled</div>
				<input type="text" v-model="$core.view.userInputFilterMaps" placeholder="Filter maps..."/>
			</div>
			<component :is="$components.MapViewer" :map="$core.view.mapViewerSelectedMap" :loader="$core.view.mapViewerTileLoader" :tile-size="512" :zoom="12" :mask="$core.view.mapViewerChunkMask" :grid-size="$core.view.mapViewerGridSize" v-model:selection="$core.view.mapViewerSelection" :selectable="true"></component>
			<div class="spaced-preview-controls">
				<component :is="$components.MenuButton" :options="menuButtonExport" :default="$core.view.config.exportMapFormat" @change="$core.view.config.exportMapFormat = $event" :disabled="$core.view.isBusy || $core.view.mapViewerSelection.length === 0" @click="export_map"></component>
			</div>

			<div id="maps-sidebar" class="sidebar">
				<span class="header">Export Options</span>
				<label class="ui-checkbox" title="Include WMO objects (large objects such as buildings)">
					<input type="checkbox" v-model="$core.view.config.mapsIncludeWMO"/>
					<span>Export WMO (Raw)</span>
				</label>
				<label class="ui-checkbox" v-if="$core.view.config.mapsIncludeWMO" title="Include objects inside WMOs (interior decorations)">
					<input type="checkbox" v-model="$core.view.config.mapsIncludeWMOSets"/>
					<span>Export WMO Sets</span>
				</label>
				<label class="ui-checkbox" title="Export M2 objects on this tile (smaller objects such as trees)">
					<input type="checkbox" v-model="$core.view.config.mapsIncludeM2"/>
					<span>Export M2 (Raw)</span>
				</label>
				<label class="ui-checkbox" title="Include terrain holes for WMOs">
					<input type="checkbox" v-model="$core.view.config.mapsIncludeHoles"/>
					<span>Include Holes</span>
				</label>
			</div>
		</div>
	`,

	data() {
		return {
			menuButtonExport: [
				{ label: 'Export OBJ', value: 'OBJ' },
				{ label: 'Export PNG', value: 'PNG' },
				{ label: 'Export Minimap Tiles', value: 'MINIMAP' }
			]
		};
	},

	methods: {
		handle_map_context(data) {
			this.$core.view.contextMenus.nodeMap = {
				selection: data.selection,
				count: data.selection.length
			};
		},

		copy_map_names(selection) {
			const names = selection.map(e => parse_map_entry(e).name);
			nw.Clipboard.get().set(names.join('\n'), 'text');
		},

		copy_map_internal_names(selection) {
			const names = selection.map(e => parse_map_entry(e).dir);
			nw.Clipboard.get().set(names.join('\n'), 'text');
		},

		copy_map_ids(selection) {
			const ids = selection.map(e => parse_map_entry(e).id);
			nw.Clipboard.get().set(ids.join('\n'), 'text');
		},

		copy_map_export_paths(selection) {
			const paths = selection.map(e => {
				const map = parse_map_entry(e);
				return ExportHelper.getExportPath(path.join('maps', map.dir));
			});
			nw.Clipboard.get().set(paths.join('\n'), 'text');
		},

		open_map_export_directory(selection) {
			if (selection.length === 0)
				return;

			const map = parse_map_entry(selection[0]);
			const dir = ExportHelper.getExportPath(path.join('maps', map.dir));
			nw.Shell.openItem(dir);
		},

		async load_map(map_id, map_dir, map_name) {
			const map_dir_lower = map_dir.toLowerCase();

			this.$core.hideToast();

			selected_map_id = map_id;
			selected_map_dir = map_dir_lower;
			selected_map_name = map_name ?? null;
			selected_wdt = null;

			this.$core.view.mapViewerHasWorldModel = false;
			this.$core.view.mapViewerIsWMOMinimap = false;
			this.$core.view.mapViewerGlobalWMO = null;
			this.$core.view.mapViewerGridSize = null;
			this.$core.view.mapViewerSelection.splice(0);

			const wdt_path = util.format('world\\maps\\%s\\%s.wdt', map_dir_lower, map_dir_lower);
			log.write('loading map preview for %s (%d)', map_dir_lower, map_id);

			try {
				const data = get_mpq_file(wdt_path);
				if (!data)
					throw new Error('WDT not found in MPQ');

				const wdt = selected_wdt = new WDTLoader(data);
				wdt.load();

				this.$core.view.mapViewerTileLoader = load_map_tile;
				this.$core.view.mapViewerChunkMask = wdt.tiles;
				this.$core.view.mapViewerSelectedMap = map_id;
				this.$core.view.mapViewerSelectedDir = map_dir;
			} catch (e) {
				log.write('cannot load %s, defaulting to all chunks enabled', wdt_path);
				this.$core.view.mapViewerTileLoader = load_map_tile;
				this.$core.view.mapViewerChunkMask = null;
				this.$core.view.mapViewerSelectedMap = map_id;
				this.$core.view.mapViewerSelectedDir = map_dir;
			}
		},

		async export_map() {
			const format = this.$core.view.config.exportMapFormat;

			if (format === 'OBJ')
				await this.export_selected_map();
			else if (format === 'PNG')
				await this.export_selected_map_as_png();
			else if (format === 'MINIMAP')
				await this.export_selected_minimap_tiles();
		},

		async export_selected_map() {
			const export_tiles = this.$core.view.mapViewerSelection;

			if (export_tiles.length === 0)
				return this.$core.setToast('error', 'You haven\'t selected any tiles; hold shift and click on a map tile to select it.', null, -1);

			const helper = new ExportHelper(export_tiles.length, 'tile');
			helper.start();

			const dir = ExportHelper.getExportPath(path.join('maps', selected_map_dir));
			const export_paths = this.$core.openLastExportStream();
			const mark_path = path.join('maps', selected_map_dir, selected_map_dir);

			for (const index of export_tiles) {
				if (helper.isCancelled())
					break;

				try {
					const obj_path = await export_terrain_obj(selected_map_dir, index, dir, this.$core.view.config, helper);
					await export_paths?.writeLine('ADT_OBJ:' + obj_path);
					helper.mark(mark_path, true);
				} catch (e) {
					helper.mark(mark_path, false, e.message, e.stack);
				}
			}

			export_paths?.close();
			helper.finish();
		},

		async export_selected_map_as_png() {
			const export_tiles = this.$core.view.mapViewerSelection;

			if (export_tiles.length === 0)
				return this.$core.setToast('error', 'You haven\'t selected any tiles; hold shift and click on a map tile to select it.', null, -1);

			const helper = new ExportHelper(export_tiles.length + 1, 'tile');
			helper.start();

			try {
				const tile_coords = export_tiles.map(index => ({
					index,
					x: Math.floor(index / MAP_SIZE),
					y: index % MAP_SIZE
				}));

				const min_x = Math.min(...tile_coords.map(t => t.x));
				const max_x = Math.max(...tile_coords.map(t => t.x));
				const min_y = Math.min(...tile_coords.map(t => t.y));
				const max_y = Math.max(...tile_coords.map(t => t.y));

				const first_tile = await load_map_tile(tile_coords[0].x, tile_coords[0].y, 512);
				if (!first_tile)
					throw new Error('unable to load first tile to determine tile size');

				const tile_size = first_tile.width;
				const tiles_wide = (max_x - min_x) + 1;
				const tiles_high = (max_y - min_y) + 1;
				const final_width = tiles_wide * tile_size;
				const final_height = tiles_high * tile_size;

				const writer = new TiledPNGWriter(final_width, final_height, tile_size);

				for (const tile_coord of tile_coords) {
					if (helper.isCancelled())
						break;

					const tile_data = await load_map_tile(tile_coord.x, tile_coord.y, tile_size);
					if (tile_data) {
						writer.addTile(tile_coord.x - min_x, tile_coord.y - min_y, tile_data);
						helper.mark(util.format('Tile %d %d', tile_coord.x, tile_coord.y), true);
					} else {
						helper.mark(util.format('Tile %d %d', tile_coord.x, tile_coord.y), false, 'Tile not available');
					}
				}

				const sorted_tiles = [...export_tiles].sort((a, b) => a - b);
				const tile_hash = crypto.createHash('md5').update(sorted_tiles.join(',')).digest('hex').substring(0, 8);

				const filename = selected_map_dir + '_' + tile_hash + '.png';
				const out_path = ExportHelper.getExportPath(path.join('maps', selected_map_dir, filename));

				await writer.write(out_path);

				const export_paths = this.$core.openLastExportStream();
				await export_paths?.writeLine('png:' + out_path);
				export_paths?.close();

				helper.mark(path.join('maps', selected_map_dir, filename), true);
			} catch (e) {
				helper.mark('PNG export', false, e.message, e.stack);
			}

			helper.finish();
		},

		async export_selected_minimap_tiles() {
			const export_tiles = this.$core.view.mapViewerSelection;

			if (export_tiles.length === 0)
				return this.$core.setToast('error', 'You haven\'t selected any tiles; hold shift and click on a map tile to select it.', null, -1);

			const helper = new ExportHelper(export_tiles.length, 'tile');
			helper.start();

			const dir = ExportHelper.getExportPath(path.join('maps', selected_map_dir, 'minimap'));
			const export_paths = this.$core.openLastExportStream();

			for (const index of export_tiles) {
				if (helper.isCancelled())
					break;

				const x = Math.floor(index / MAP_SIZE);
				const y = index % MAP_SIZE;
				const tile_name = util.format('map%s_%s', x.toString().padStart(2, '0'), y.toString().padStart(2, '0'));
				const mark_path = path.join('maps', selected_map_dir, 'minimap', tile_name);

				try {
					const tile_data = await load_map_tile(x, y, 512);
					if (!tile_data)
						throw new Error('minimap tile not available');

					const png = new PNGWriter(tile_data.width, tile_data.height);
					png.getPixelData().set(tile_data.data);

					const out_path = path.join(dir, tile_name + '.png');
					await png.write(out_path);

					await export_paths?.writeLine('png:' + out_path);
					helper.mark(mark_path, true);
				} catch (e) {
					helper.mark(mark_path, false, e.message, e.stack);
				}
			}

			export_paths?.close();
			helper.finish();
		},

		async initialize() {
			this.$core.showLoadingScreen(2);
			await this.$core.progressLoadingScreen('Loading map database...');

			try {
				const mpq = this.$core.view.mpq;
				const build_id = mpq.build_id ?? '1.12.1.5875';

				minimap_translate = load_minimap_translate();

				const raw_data = mpq.getFile('DBFilesClient\\Map.dbc');
				if (!raw_data)
					throw new Error('Map.dbc not found in MPQ archives');

				const data = new BufferWrapper(Buffer.from(raw_data));
				const dbc = new DBCReader('Map.dbc', build_id);
				await dbc.parse(data);

				const rows = await dbc.getAllRows();
				const maps = [];

				for (const [id, row] of rows) {
					const dir = row.Directory;
					const name = row.MapName_lang ?? row.MapName ?? ('Map ' + id);

					if (!dir || dir.length === 0)
						continue;

					// verify WDT exists in MPQ via listfile lookup (avoids full extraction)
					const wdt_key = util.format('world\\maps\\%s\\%s.wdt', dir.toLowerCase(), dir.toLowerCase());
					if (!mpq.listfile.has(wdt_key))
						continue;

					maps.push(util.format('0\x19[%d]\x19%s\x19(%s)', id, name, dir));
				}

				this.$core.view.mapViewerMaps = maps;
				log.write('loaded %d maps from Map.dbc', maps.length);
			} catch (e) {
				log.write('failed to load maps: %s', e.message);
				this.$core.setToast('error', 'Failed to load map list. Check the log for details.', { 'View Log': () => log.openRuntimeLog() }, -1);
			}

			this.$core.hideLoadingScreen();
		}
	},

	async mounted() {
		this.$core.view.mapViewerTileLoader = load_map_tile;

		this.$core.view.$watch('selectionMaps', async selection => {
			const first = selection[0];

			if (!this.$core.view.isBusy && first) {
				const map = parse_map_entry(first);
				if (selected_map_id !== map.id)
					this.load_map(map.id, map.dir, map.name);
			}
		});

		await this.initialize();
	}
};

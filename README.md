# EASPF

Four SAR processing programs, one self-contained Python file each, no local imports.
Each module folder holds its program together with `raw/` sample inputs and the
`results/` outputs produced from them.

## Programs

`01ST-GVD/stgvd.py`
Subcommands: `json-filter`, `par-filter`, `select-image`, `kml2shp`.
`json-filter` filters archive JSON by AOI overlap and groups scenes into physical
tracks by heading, incidence angle and cross-track distance, then writes one KML
footprint per scene plus an overlap list for each track.
`par-filter` scans a whole working directory for pairwise footprint overlap and
rewrites the `itab` on/off column from the result.
`select-image` keeps the slaves that can interfere with one master.
`kml2shp` merges footprints into a single Shapefile, optionally carrying overlap.

`02RW-EKE/rweke.py`
Subcommands: `correct-orbit`, `delta-lonlat`.
`correct-orbit` rebuilds the orbit inside a product XML: temporally weighted RANSAC
picks a consensus set, weighted least squares fits a cubic polynomial, positions
inside the observed span are interpolated and outside it extrapolated along the
boundary tangent so the fit cannot diverge, velocities come from the analytic
derivative, and the result is written back as a fixed number of epochs.
`delta-lonlat` pairs the import/raw/correct parameter files of one scene and
reports how far their centre coordinates moved.

`03GC-CPR/gccpr.py`
Subcommands: `register`, `stat-tif`, `stat-fft`.
`register` runs two passes on the GPU: normalised cross-correlation with spectral
zero-padding yields the full offset field, the slave window is then shifted by the
integer part of that offset so only the residual is searched, and a local
Matrix-DFT zoom refines it down to sub-pixel.
`stat-tif` and `stat-fft` give a binned histogram, or power-spectrum metrics
(low/high band energy ratio and spectral entropy), for a region you cut out.

`04FO-BEAM/fobeam.py`
No subcommands; it runs one nine-step chain: symlink the workspace, estimate the
initial offset, multi-look, form the differential interferogram, adaptive
filtering, unwrapping, trend removal, atmosphere removal, geocoding.
Every step checks its outputs and is skipped when they already exist.

## Install

```bash
python -m pip install -r requirements.txt
```

`stgvd.py` and `rweke.py` need `numpy` + `shapely`; `gccpr.py register` needs
`scipy` + `cupy`; its two statistic subcommands need `rasterio` + `matplotlib`.
`fobeam.py` and part of `stgvd.py` call external interferometry tools
(`create_offset` … `data2geotiff`, `mk_tab`, `base_calc`, `SLC_corners`, `ogr2ogr`).

## Example

```bash
unzip 01ST-GVD/raw/json.zip -d 01ST-GVD/raw      # once

python 01ST-GVD/stgvd.py json-filter \
    --aoi 01ST-GVD/results/MT-json/border.kml \
    --json-dir 01ST-GVD/raw/MT-json/json \
    --out-dir /tmp/out --min-overlap 0.05
```

Every subcommand answers `--help`; those calling external tools also accept
`--dry-run`.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-GVD: scene filtering, physical track clustering and vector footprints

Subcommands
------
json-filter    filter archive JSON by AOI overlap, then group scenes into
               physical tracks; emit per-scene KML footprints and overlap list
par-filter     full-matrix overlap scan from .slc.par / .slc.kml footprints;
               writes bperp_file_overlap.txt and the itab on/off flags
select-image   pick slaves with the same heading, close incidence and a real
               intersection with the master; build master-<id>.slc.par-list/
kml2shp        merge a KML directory into one Shapefile (overlap as attribute)

Dependencies
----
python >= 3.8, shapely
par-filter / select-image need external commands mk_tab, base_calc, SLC_corners
kml2shp needs ogr2ogr

Examples
----
python stgvd.py json-filter --aoi border.kml --json-dir raw/json \
    --out-dir results --min-overlap 0.05
python stgvd.py par-filter --workdir ./Path_01_ASC --ratio 0.4
python stgvd.py select-image --dir ./Path_01_ASC --master 20230626_0000140954
python stgvd.py kml2shp --indir results/DEC_STRIP1_Track1_0.05 --prefix ZX
"""


from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from shapely.geometry import Polygon
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "Error: missing dependency shapely. Install it first:  pip install shapely\n"
    )
    raise SystemExit(2)


VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WGS84_A_KM = 6378.137          # WGS84 semi-major axis (km)
DEG_TO_KM = math.pi / 180.0 * WGS84_A_KM
SPEED_OF_LIGHT = 299792458.0   # m/s

KML_TAIL = ".tar.gz"           # default archive suffix in the scene list


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def die(message: str, code: int = 2) -> "None":
    """Print an error and exit."""
    sys.stderr.write("Error: {}\n".format(message))
    raise SystemExit(code)


def safe_float(value, default: float = 0.0) -> float:
    """Convert 'null' / empty / scientific-notation text to float, else default."""
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", "nan"):
        return default
    try:
        return float(text)
    except ValueError:
        return default


def strip_suffix(name: str, suffix: str) -> str:
    """Strip a fixed trailing suffix."""
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def run(cmd, cwd=None, dry_run=False, check=True, quiet=True) -> str:
    """Run an external command; with dry_run only print it. Returns stdout."""
    argv = [str(item) for item in cmd]
    printable = " ".join(argv)
    if dry_run:
        print("[dry-run] {}".format(printable))
        return ""
    if not quiet:
        print("[run] {}".format(printable))
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        die("executable not found: {} (make sure the tools are on PATH)".format(argv[0]))
    if check and proc.returncode != 0:
        die(
            "command failed (exit code {}): {}\n{}".format(
                proc.returncode, printable, proc.stderr.strip()
            )
        )
    return proc.stdout


def which_or_die(program: str, hint: str = "") -> None:
    """Verify that an external program exists."""
    if shutil.which(program) is None:
        tail = " " + hint if hint else ""
        die("executable '{}' not found; install it or add it to PATH.{}".format(program, tail))


# ---------------------------------------------------------------------------
# Geometry kernel
# ---------------------------------------------------------------------------
def make_polygon(points):
    """Build a valid polygon from [(lon, lat), ...]; None on failure."""
    if len(points) < 3:
        return None
    try:
        poly = Polygon(points)
    except Exception:
        return None
    if poly.is_empty:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.is_empty:
            return None
    return poly


def kml_polygon(kml_path):
    """Return the largest Polygon in a KML (namespace and MultiGeometry safe)."""
    path = Path(kml_path)
    if not path.is_file():
        return None
    try:
        root = ET.parse(str(path)).getroot()
    except (ET.ParseError, OSError):
        return None

    best = None
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag != "coordinates" or not node.text:
            continue
        points = []
        for token in node.text.split():
            parts = token.split(",")
            if len(parts) < 2:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
            except ValueError:
                continue
            points.append((lon, lat))
        if len(points) > 1 and points[0] == points[-1]:
            points = points[:-1]
        poly = make_polygon(points)
        if poly is None:
            continue
        if best is None or poly.area > best.area:
            best = poly
    return best


def calculate_heading(lon1, lat1, lon2, lat2, orbit_dir: str = "") -> float:
    """Track azimuth (deg) from start -> end. Endpoints are swapped for DEC."""
    direction = (orbit_dir or "").upper()
    if direction.startswith("DEC") or direction.startswith("DESC"):
        lon1, lat1, lon2, lat2 = lon2, lat2, lon1, lat1
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.cos(dlon)
    )
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def calculate_incidence_angle(bandwidth_hz: float, ground_res_m: float) -> float:
    """Incidence angle (deg) from pulse bandwidth and ground range resolution.

    Slant resolution slant = c / (2 * B), so sin(inc) = slant / ground_res.
    """
    if bandwidth_hz <= 0 or ground_res_m <= 0:
        return 0.0
    slant_res = SPEED_OF_LIGHT / (2.0 * bandwidth_hz)
    ratio = slant_res / ground_res_m
    ratio = max(-1.0, min(1.0, ratio))
    try:
        return math.degrees(math.asin(ratio))
    except ValueError:
        return 0.0


def cross_track_km(ref_lon, ref_lat, ref_heading, cur_lon, cur_lat) -> float:
    """Cross-track distance (km) relative to the reference heading."""
    heading_rad = math.radians(ref_heading)
    dx = (cur_lon - ref_lon) * math.cos(math.radians((ref_lat + cur_lat) / 2.0))
    dy = cur_lat - ref_lat
    dist_deg = abs(dx * math.cos(heading_rad) - dy * math.sin(heading_rad))
    return dist_deg * DEG_TO_KM


def overlap_ratio(aoi_poly, img_poly, calc_mode: int = 0) -> float:
    """Image/AOI overlap. calc_mode=0 divides by image area, 1 by AOI area."""
    if aoi_poly is None or img_poly is None:
        return 0.0
    try:
        inter = img_poly.intersection(aoi_poly).area
        denom = img_poly.area if calc_mode == 0 else aoi_poly.area
        if denom <= 0:
            return 0.0
        return round(inter / denom, 4)
    except Exception:
        return 0.0


def polygon_iou(poly_a, poly_b) -> float:
    """Intersection over union of two footprints."""
    if poly_a is None or poly_b is None:
        return 0.0
    try:
        union = poly_a.union(poly_b).area
        if union <= 0:
            return 0.0
        return round(poly_a.intersection(poly_b).area / union, 4)
    except Exception:
        return 0.0


def track_tolerance_km(imaging_mode: str) -> float:
    """Allowed cross-track distance for one track (STRIP5 / STRIP2 / other)."""
    mode = (imaging_mode or "").upper()
    if "STRIP5" in mode:
        return 40.0
    if "STRIP2" in mode:
        return 25.0
    return 12.0


# ---------------------------------------------------------------------------
# Subcommand 1: json-filter
# ---------------------------------------------------------------------------
CORNER_KEYS = (
    ("topleftlongitude", "topleftlatitude"),
    ("toprightlongitude", "toprightlatitude"),
    ("bottomrightlongitude", "bottomrightlatitude"),
    ("bottomleftlongitude", "bottomleftlatitude"),
)


def load_lt1_meta(json_path: Path):
    """Parse one LT-1 JSON; return None when a required field is missing."""
    try:
        with open(str(json_path), "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    def get(key):
        value = raw.get(key)
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in ("null", "none"):
            return None
        return text

    archive = get("dataarchivefile") or get("dataarchivefileurl")
    if archive is None:
        return None
    fname = re.split(r"[\\/]", archive.rstrip("/"))[-1]
    if not fname:
        return None

    center_lon, center_lat = get("centerlongitude"), get("centerlatitude")
    bandwidth, ground_res = get("pulsebandwidth"), get("groundrangeresolution")
    if center_lon is None or center_lat is None or bandwidth is None:
        return None

    corners = []
    for lon_key, lat_key in CORNER_KEYS:
        lon, lat = get(lon_key), get(lat_key)
        if lon is None or lat is None or ground_res is None:
            return None
        corners.append((float(lon), float(lat)))

    return {
        "fname": fname,
        "mode": get("imagingmode") or "NA",
        "direction": (get("orbitdirection") or "ASC").upper(),
        "bandwidth": float(bandwidth),
        "ground_res": float(ground_res),
        "center": (float(center_lon), float(center_lat)),
        "corners": corners,
    }


def footprint_kml(name: str, overlap: float, incidence: float, corners) -> str:
    """Build the footprint KML for one scene."""
    ring = " ".join("{:.10f},{:.10f}".format(lon, lat) for lon, lat in corners)
    first = "{:.10f},{:.10f}".format(corners[0][0], corners[0][1])
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        "  <Document>\n"
        "    <Placemark>\n"
        "      <name>{name}</name>\n"
        "      <description>Overlap: {overlap:.4f}, IncAngle: {incidence:.2f}</description>\n"
        "      <Style>\n"
        "        <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>\n"
        "        <PolyStyle><fill>0</fill></PolyStyle>\n"
        "      </Style>\n"
        "      <Polygon>\n"
        "        <outerBoundaryIs><LinearRing><coordinates>{ring} {first}"
        "</coordinates></LinearRing></outerBoundaryIs>\n"
        "      </Polygon>\n"
        "    </Placemark>\n"
        "  </Document>\n"
        "</kml>\n"
    ).format(name=name, overlap=overlap, incidence=incidence, ring=ring, first=first)


def clean_json_filter_outputs(out_dir: Path, suffix: str) -> None:
    """Remove previous outputs produced with the same threshold."""
    for entry in sorted(out_dir.glob("*_Track*_{}".format(suffix))):
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
            print("[clean] removing directory {}".format(entry.name))
    for entry in sorted(out_dir.glob("*_Track*_{}.txt".format(suffix))):
        entry.unlink()
        print("[clean] removing file {}".format(entry.name))


def cmd_json_filter(args) -> int:
    aoi_poly = kml_polygon(args.aoi)
    if aoi_poly is None:
        die("cannot parse a polygon from the AOI file: {}".format(args.aoi))

    json_dir = Path(args.json_dir).expanduser().resolve()
    if not json_dir.is_dir():
        die("JSON directory not found: {}".format(json_dir))

    json_files = sorted(p for p in json_dir.glob(args.pattern) if p.is_file())
    if not json_files:
        die("no file matching '{}' in: {}".format(args.pattern, json_dir))

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else json_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "{:.2f}".format(args.min_overlap)
    if args.clean:
        clean_json_filter_outputs(out_dir, suffix)

    pools = {"ASC": [], "DEC": []}
    buckets = {}
    skipped = 0

    print("Classifying scenes by physical vectors and overlap ...")
    print("-" * 64)

    for json_file in json_files:
        meta = load_lt1_meta(json_file)
        if meta is None:
            skipped += 1
            continue

        img_poly = make_polygon(meta["corners"])
        if img_poly is None:
            skipped += 1
            continue

        ratio = overlap_ratio(aoi_poly, img_poly, args.calc_mode)
        if ratio < args.min_overlap:
            continue

        print("Match: {} ({:.4f})".format(meta["fname"], ratio))

        bl = meta["corners"][3]
        tl = meta["corners"][0]
        heading = calculate_heading(bl[0], bl[1], tl[0], tl[1], meta["direction"])
        incidence = calculate_incidence_angle(meta["bandwidth"], meta["ground_res"])

        direction_tag = "ASC" if meta["direction"].startswith("ASC") else "DEC"
        tolerance = track_tolerance_km(meta["mode"])
        pool = pools[direction_tag]

        track_id = None
        for index, ref in enumerate(pool, start=1):
            dist = cross_track_km(
                ref["lon"], ref["lat"], ref["heading"],
                meta["center"][0], meta["center"][1],
            )
            if dist < tolerance and abs(incidence - ref["incidence"]) < 1.0:
                track_id = index
                break
        if track_id is None:
            pool.append(
                {
                    "lon": meta["center"][0],
                    "lat": meta["center"][1],
                    "heading": heading,
                    "incidence": incidence,
                }
            )
            track_id = len(pool)
            print(
                ">> new physical track: {} Track_{} (incidence ~{:.2f} deg)".format(
                    direction_tag, track_id, incidence
                )
            )

        tag = "{}_{}_Track{}_{}".format(direction_tag, meta["mode"], track_id, suffix)
        buckets.setdefault(tag, []).append(
            {
                "fname": meta["fname"],
                "ratio": ratio,
                "incidence": incidence,
                "corners": meta["corners"],
            }
        )

    print("-" * 64)
    if skipped:
        print("skipped {} JSON files that failed to parse or lack fields.".format(skipped))
    if not buckets:
        print("no scene reaches the overlap threshold {:.2f}.".format(args.min_overlap))
        return 0

    print("Writing KML footprints and lists ...")
    for tag in sorted(buckets):
        folder = out_dir / tag
        folder.mkdir(parents=True, exist_ok=True)
        records = buckets[tag]
        lines = []
        for record in records:
            basis = strip_suffix(record["fname"], KML_TAIL)
            lines.append("{:<75} {:.4f}".format(record["fname"], record["ratio"]))
            (folder / "{}.kml".format(basis)).write_text(
                footprint_kml(
                    basis, record["ratio"], record["incidence"], record["corners"]
                ),
                encoding="utf-8",
            )
        list_path = out_dir / "{}.txt".format(tag)
        list_path.write_text(
            "\n".join(sorted(set(lines))) + "\n", encoding="utf-8"
        )
        print(
            "[{}] {:>4d} scenes -> {}  (list: {})".format(
                tag, len(records), folder.name, list_path.name
            )
        )
    print("Selection finished.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand 2: par-filter
# ---------------------------------------------------------------------------
def scene_polygon(work_dir: Path, scene_id: str, dry_run: bool = False):
    """Footprint polygon of a scene; generated with SLC_corners when missing."""
    kml_path = work_dir / "{}.slc.kml".format(scene_id)
    if not kml_path.is_file():
        run(
            ["SLC_corners", "{}.slc.par".format(scene_id), "-", kml_path.name],
            cwd=work_dir,
            dry_run=dry_run,
            check=False,
        )
    return kml_polygon(kml_path)


def cmd_par_filter(args) -> int:
    if not args.dry_run:
        which_or_die("mk_tab", "This subcommand needs mk_tab/base_calc/SLC_corners.")
    work_dir = Path(args.workdir).expanduser().resolve()
    if not work_dir.is_dir():
        die("working directory not found: {}".format(work_dir))

    par_files = sorted(p.name for p in work_dir.glob("*.slc.par"))
    if not par_files:
        die("no *.slc.par in: {}".format(work_dir))
    count = len(par_files)
    print("found {} scenes (*.slc.par) in {}".format(count, work_dir))

    for name in ("bperp_file_overlap.txt", "itab.new", "itab", "slc_tab", "bperp_file", "par.list"):
        target = work_dir / name
        if target.exists() and not args.dry_run:
            target.unlink()

    run(["mk_tab", ".", "slc", "slc.par", "slc_tab"], cwd=work_dir, dry_run=args.dry_run)

    entries = []
    pair_id = 1
    for i in range(1, count + 1):
        for j in range(i + 1, count + 1):
            entries.append((i, j, pair_id))
            pair_id += 1
    itab_path = work_dir / "itab"
    if not args.dry_run:
        itab_path.write_text(
            "".join("{:4d} {:4d} {:4d} {:2d}\n".format(m, s, p, 1) for m, s, p in entries),
            encoding="utf-8",
        )

    run(
        [
            "base_calc", "slc_tab", par_files[0], "bperp_file", "itab", "1", "1",
            "-", args.max_bperp, "-", args.max_btemp, str(count),
        ],
        cwd=work_dir,
        dry_run=args.dry_run,
    )

    bperp_lines = {}
    bperp_path = work_dir / "bperp_file"
    if bperp_path.is_file():
        for line in bperp_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if fields:
                bperp_lines[fields[0]] = line.rstrip("\n")

    print("Starting full-matrix overlap scan (KML footprints) ...")
    overlap_lines = []
    itab_lines = []
    ratios = []
    enabled = 0

    for m_idx, s_idx, p_idx in entries:
        m_id = strip_suffix(par_files[m_idx - 1], ".slc.par")
        s_id = strip_suffix(par_files[s_idx - 1], ".slc.par")
        ratio = polygon_iou(
            scene_polygon(work_dir, m_id, args.dry_run),
            scene_polygon(work_dir, s_id, args.dry_run),
        )
        ratios.append(ratio)
        overlap_lines.append("{} {:8.4f}".format(bperp_lines.get(str(p_idx), ""), ratio))
        flag = 1 if ratio >= args.ratio else 0
        enabled += flag
        itab_lines.append("{:4d} {:4d} {:4d} {:2d}".format(m_idx, s_idx, p_idx, flag))

    if not args.dry_run:
        (work_dir / "bperp_file_overlap.txt").write_text(
            "\n".join(overlap_lines) + "\n", encoding="utf-8"
        )
        itab_path.write_text("\n".join(itab_lines) + "\n", encoding="utf-8")

    print("-" * 64)
    print("image pairs: {}".format(len(entries)))
    if ratios:
        print(
            "overlap min/mean/max: {:.4f} / {:.4f} / {:.4f}".format(
                min(ratios), sum(ratios) / len(ratios), max(ratios)
            )
        )
    print(
        "pairs enabled at threshold {:.2f}: {} / {} (written to itab column 4)".format(
            args.ratio, enabled, len(entries)
        )
    )
    print("bperp_file_overlap.txt updated (Overlap column appended).")
    return 0


# ---------------------------------------------------------------------------
# Subcommand 3: select-image
# ---------------------------------------------------------------------------
def parse_scene_par(par_path: Path) -> dict:
    """Parse a .par file ('key: value' lines); value is the first token."""
    params = {}
    path = Path(par_path)
    if not path.is_file():
        return params
    with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields = value.strip().split()
            params[key.strip()] = fields[0] if fields else ""
    return params


def same_heading_sign(head_a: float, head_b: float) -> bool:
    """Orbit-side test: only headings with the same sign can interfere."""
    if head_a == 0 or head_b == 0:
        return False
    return (head_a > 0) == (head_b > 0)


def link_or_copy(src: Path, dst: Path, as_symlink: bool) -> None:
    """Create a symlink; fall back to a copy when unsupported."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if as_symlink:
        try:
            dst.symlink_to(src)
            return
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(str(src), str(dst))


def cmd_select_image(args) -> int:
    if not args.dry_run:
        which_or_die("mk_tab", "This subcommand needs mk_tab/base_calc/SLC_corners.")

    work_dir = Path(args.dir).expanduser().resolve()
    if not work_dir.is_dir():
        die("working directory not found: {}".format(work_dir))

    master_id = args.master
    master_par = work_dir / "{}.slc.par".format(master_id)
    if not master_par.is_file():
        die("master parameter file not found: {}".format(master_par))

    out_root = (
        Path(args.outdir).expanduser().resolve() if args.outdir else work_dir
    )
    list_dir = out_root / "master-{}.slc.par-list".format(master_id)
    list_dir.mkdir(parents=True, exist_ok=True)

    master_params = parse_scene_par(master_par)
    m_heading = safe_float(master_params.get("heading"), 0.0)
    m_incidence = safe_float(master_params.get("incidence_angle"), 0.0)
    master_poly = scene_polygon(work_dir, master_id, args.dry_run)
    if master_poly is None:
        print("warning: master footprint unavailable, overlap cannot be computed.")

    # Add the master's own assets
    link_or_copy(work_dir / "{}.slc".format(master_id), list_dir / "{}.slc".format(master_id), True)
    for ext in (".slc.par", ".slc.kml", ".slc.bmp"):
        src = work_dir / "{}{}".format(master_id, ext)
        if src.is_file():
            shutil.copy2(str(src), str(list_dir / src.name))

    log_path = out_root / "02LT_select_image.log"
    log_lines = [
        "workdir: {}".format(work_dir),
        "LT image select dir: {}".format(list_dir),
    ]

    slaves = [
        p for p in sorted(work_dir.glob("*.slc.par"))
        if strip_suffix(p.name, ".slc.par") != master_id
    ]
    print("master: {}  candidate slaves: {}".format(master_id, len(slaves)))

    overlap_lines = []
    selected = 0

    for index, par_path in enumerate(slaves, start=1):
        slave_id = strip_suffix(par_path.name, ".slc.par")
        print("process:({}/{}) {}".format(index, len(slaves), slave_id))

        params = parse_scene_par(par_path)
        s_heading = safe_float(params.get("heading"), 0.0)
        s_incidence = safe_float(params.get("incidence_angle"), 0.0)

        if not same_heading_sign(m_heading, s_heading):
            print("  {} heading differs from the master, skipped.".format(slave_id))
            continue
        if abs(round(m_incidence - s_incidence)) >= 2:
            print("  {} incidence differs too much ({:.2f} deg), skipped.".format(slave_id, m_incidence - s_incidence))
            continue

        slave_poly = scene_polygon(work_dir, slave_id, args.dry_run)
        if master_poly is None or slave_poly is None:
            print("  {} footprint missing, skipped.".format(slave_id))
            continue
        if not master_poly.intersects(slave_poly):
            print("  {} does not overlap the master.".format(slave_id))
            continue

        ratio = round(master_poly.intersection(slave_poly).area / master_poly.area, 4)
        title = params.get("title", slave_id) or slave_id
        overlap_lines.append("{}{} {:.4f}".format(title, KML_TAIL, ratio))

        if ratio >= args.overlap:
            link_or_copy(work_dir / "{}.slc".format(slave_id), list_dir / "{}.slc".format(slave_id), True)
            for ext in (".slc.par", ".slc.kml", ".slc.bmp"):
                src = work_dir / "{}{}".format(slave_id, ext)
                if src.is_file():
                    shutil.copy2(str(src), str(list_dir / src.name))
            selected += 1
            message = "result: {} is selected !!!   overlap:{:.4f}".format(slave_id, ratio)
            print("  " + message)
            log_lines.append(message)

    if overlap_lines:
        (out_root / "overlap_result.txt").write_text(
            "\n".join(overlap_lines) + "\n", encoding="utf-8"
        )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("-" * 64)
    print("selected slaves: {} / {}".format(selected, len(slaves)))
    print("output directory: {}".format(list_dir))

    run(
        ["mk_tab", str(list_dir), "slc", "slc.par", str(list_dir / "slc_tab")],
        cwd=work_dir,
        dry_run=args.dry_run,
    )
    run(
        [
            "base_calc", str(list_dir / "slc_tab"),
            "{}.slc.par".format(master_id),
            str(list_dir / "bperp_file"), str(list_dir / "itab"),
            "1", "1", "-", "-", "-", "-", str(len(slaves)),
        ],
        cwd=work_dir,
        dry_run=args.dry_run,
    )
    print("Finish processing")
    return 0


# ---------------------------------------------------------------------------
# Subcommand 4: kml2shp
# ---------------------------------------------------------------------------
def load_overlap_map(txt_path: Path) -> dict:
    """Read the small list into {lowercase basename: value}.

    Each line: `<filename>[.tar.gz] <overlap_value>`; the name may contain spaces.
    """
    mapping = {}
    path = Path(txt_path)
    if not path.is_file():
        die("overlap list not found: {}".format(path))
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        value_text = fields[-1]
        name_text = " ".join(fields[:-1])
        try:
            value = float(value_text.strip('"').strip(","))
        except ValueError:
            continue
        name_text = name_text.strip().strip('"').strip("'")
        name_text = re.split(r"[\\/]", name_text)[-1]
        for suffix in (".tar.gz", ".tar", ".gz", ".zip"):
            if name_text.lower().endswith(suffix):
                name_text = name_text[: -len(suffix)]
                break
        if name_text:
            mapping[name_text.lower()] = value
    return mapping


def cmd_kml2shp(args) -> int:
    which_or_die("ogr2ogr", "This subcommand needs GDAL.")

    in_dir = Path(args.indir).expanduser().resolve()
    if not in_dir.is_dir():
        die("KML directory not found: {}".format(in_dir))

    kml_files = sorted(p for p in in_dir.glob("*.kml") if p.is_file())
    if not kml_files:
        die("no .kml files in: {}".format(in_dir))

    folder = in_dir.name
    direction = re.search(r"ASC|DEC", folder)
    track = re.search(r"Track(\d+)", folder)
    suffix = "{}{}".format(
        direction.group(0) if direction else "",
        track.group(1) if track else "",
    )

    stem = args.name if args.name else (
        "{}_{}".format(args.prefix, suffix) if suffix else args.prefix
    )
    if args.overlap_txt:
        stem += "_with_overlap"

    out_root = Path(args.outdir).expanduser().resolve() if args.outdir else Path.cwd()
    out_root.mkdir(parents=True, exist_ok=True)
    out_shp = out_root / "{}.shp".format(stem)
    layer = stem

    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"):
        stale = out_shp.with_suffix(ext)
        if stale.exists():
            stale.unlink()

    overlap_map = load_overlap_map(args.overlap_txt) if args.overlap_txt else {}
    if args.overlap_txt:
        print("overlap map entries loaded: {}".format(len(overlap_map)))

    tmp_dir = Path(tempfile.mkdtemp(prefix="kml2shp_"))
    unmatched = []
    matched = 0
    first = True

    try:
        for kml_path in kml_files:
            props = {}
            if args.overlap_txt:
                key = kml_path.stem.lower()
                value = overlap_map.get(key)
                if value is None:
                    for candidate in (key + ".tar.gz", key + ".gz"):
                        if candidate in overlap_map:
                            value = overlap_map[candidate]
                            break
                if value is None:
                    unmatched.append(kml_path.name)
                    props["overlap"] = None
                else:
                    props["overlap"] = value
                    matched += 1

            geo_in = tmp_dir / "in.geojson"
            geo_out = tmp_dir / "out.geojson"
            run(["ogr2ogr", "-f", "GeoJSON", str(geo_in), str(kml_path)])

            source = geo_in
            if props:
                with open(str(geo_in), "r", encoding="utf-8") as handle:
                    doc = json.load(handle)
                for feature in doc.get("features", []):
                    if not isinstance(feature.get("properties"), dict):
                        feature["properties"] = {}
                    feature["properties"].update(props)
                with open(str(geo_out), "w", encoding="utf-8") as handle:
                    json.dump(doc, handle)
                source = geo_out

            cmd = ["ogr2ogr", "-f", "ESRI Shapefile", "-nln", layer]
            if not first:
                cmd += ["-update", "-append"]
            cmd += [str(out_shp), str(source)]
            run(cmd)
            first = False
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    if args.overlap_txt:
        (out_root / "unmatched.txt").write_text(
            "\n".join(unmatched) + ("\n" if unmatched else ""), encoding="utf-8"
        )

    print("Done.")
    print("  KML files: {}".format(len(kml_files)))
    if args.overlap_txt:
        print("  files matched to an overlap value: {}".format(matched))
        print("  unmatched files written to: {}".format(out_root / "unmatched.txt"))
    print("  final Shapefile: {}".format(out_shp))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXAMPLES = """\
Examples
--------
# 1) AOI overlap filtering + physical track clustering (no external commands)
python stgvd.py json-filter --aoi border.kml --json-dir raw/ZX-json/json \\
    --out-dir results/ZX-json --min-overlap 0.05 --clean

# 2) full-matrix overlap scan in a working directory, rewrite the itab flags
python stgvd.py par-filter --workdir ./Path_01_ASC --ratio 0.4 \\
    --max-bperp 500 --max-btemp 100

# 3) pick slaves that can interfere with master 20230626_0000140954
python stgvd.py select-image --dir ./Path_01_ASC --master 20230626_0000140954 \\
    --overlap 0.8

# 4) merge a KML directory into a Shapefile with overlap as an attribute
python stgvd.py kml2shp --indir results/ZX-json/DEC_STRIP1_Track1_0.05 \\
    --prefix ZX --overlap-txt results/ZX-json/DEC_STRIP1_Track1_0.05.txt
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stgvd.py",
        description="ST-GVD: LT-1 scene grouping, track clustering and vector footprints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + VERSION)
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # --- json-filter ---
    p_json = subparsers.add_parser(
        "json-filter",
        help="filter LT-1 JSON by AOI overlap and group scenes into physical tracks",
        description="Walk archive JSON, filter by AOI overlap and emit KML footprints per track.",
    )
    p_json.add_argument("--aoi", required=True, help="AOI polygon KML file")
    p_json.add_argument("--json-dir", required=True, help="directory holding the LT-1 JSON files")
    p_json.add_argument("--out-dir", default=None, help="output directory (defaults to the JSON directory)")
    p_json.add_argument(
        "--min-overlap", type=float, default=0.5, help="minimum overlap ratio (default 0.5)"
    )
    p_json.add_argument(
        "--calc-mode", type=int, choices=(0, 1), default=0,
        help="0=divide by image area (default), 1=divide by AOI area",
    )
    p_json.add_argument("--pattern", default="*.json", help="filename glob (default *.json)")
    p_json.add_argument("--clean", action="store_true", help="remove previous outputs of the same threshold first")
    p_json.set_defaults(func=cmd_json_filter)

    # --- par-filter ---
    p_par = subparsers.add_parser(
        "par-filter",
        help="full-matrix overlap scan in a working directory, rewrite itab",
        description="Compute pairwise IoU from .slc.par/.slc.kml footprints; write bperp_file_overlap.txt and the itab flags.",
    )
    p_par.add_argument("--workdir", required=True, help="working directory containing *.slc.par")
    p_par.add_argument("--ratio", type=float, default=0.4, help="overlap threshold (default 0.4)")
    p_par.add_argument("--max-bperp", default="-", help="maximum perpendicular baseline passed to base_calc")
    p_par.add_argument("--max-btemp", default="-", help="maximum temporal baseline passed to base_calc")
    p_par.add_argument("--dry-run", action="store_true", help="print external commands without running them")
    p_par.set_defaults(func=cmd_par_filter)

    # --- select-image ---
    p_sel = subparsers.add_parser(
        "select-image",
        help="select slaves that can interfere with one master scene",
        description="Filter slaves by heading, incidence and a real intersection with the master; build the master list directory.",
    )
    p_sel.add_argument("--dir", default=".", help="directory containing *.slc.par (default: current directory)")
    p_sel.add_argument("--master", required=True, help="master scene ID (without .slc.par)")
    p_sel.add_argument("--outdir", default=None, help="output root (defaults to --dir)")
    p_sel.add_argument("--overlap", type=float, default=0.8, help="overlap threshold (default 0.8)")
    p_sel.add_argument("--dry-run", action="store_true", help="print external commands without running them")
    p_sel.set_defaults(func=cmd_select_image)

    # --- kml2shp ---
    p_shp = subparsers.add_parser(
        "kml2shp",
        help="merge a KML directory into one Shapefile",
        description="Merge every KML in a directory into one Shapefile; optionally attach overlap as an attribute.",
    )
    p_shp.add_argument("--indir", required=True, help="KML directory")
    p_shp.add_argument("--prefix", default="footprints", help="output filename prefix (default footprints)")
    p_shp.add_argument("--name", default=None, help="output filename without extension")
    p_shp.add_argument("--overlap-txt", default=None, help="small overlap list (enables the attribute)")
    p_shp.add_argument("--outdir", default=None, help="output directory (default: current directory)")
    p_shp.set_defaults(func=cmd_kml2shp)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

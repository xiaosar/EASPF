#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RW-EKE: robust orbit-window reconstruction and geometric centre offset evaluation

Subcommands
------
correct-orbit  temporally weighted RANSAC + WLS cubic orbit reconstruction;
               polynomial interpolation inside the observed span, boundary
               tangent extrapolation outside it; analytic velocities; rewrite
               orbitHeader and stateVec to a fixed point count
delta-lonlat   pair .slc_import / .slc_raw / .slc_correct parameter files and
               report raw/correct centre offsets against import, in metres

Dependencies
----
python >= 3.8, numpy
xlsx output needs pandas + openpyxl; falls back to csv when missing

Examples
----
python rweke.py correct-orbit scene.xml
python rweke.py correct-orbit scene.xml --dry-run
python rweke.py correct-orbit scene.xml --points 64 --pre-roll 30 \
    --output scene_fixed.xml
python rweke.py delta-lonlat --dir ./Path_01_ASC --threshold 3.0
"""


from __future__ import annotations

import argparse
import copy
import csv
import datetime
import math
import re
import shutil
import statistics
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

VERSION = "1.0.0"

DEFAULT_POINTS = 40          # stateVec points written back
DEFAULT_PRE_ROLL = 15        # seconds before the scene start for the first epoch
DEFAULT_SIGMA_T = 20.0       # std of the temporal Gaussian weight (s)
DEFAULT_ITERS = 80           # RANSAC iterations
DEFAULT_TAU = 0.1            # RANSAC inlier threshold (m)
MIN_ORBIT_RADIUS = 1.0e6     # minimum orbit radius (m) to drop invalid telemetry
FLOAT_FMT = "{:.17E}"        # high-precision scientific notation, lossless round trip


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def die(message: str, code: int = 2) -> "None":
    sys.stderr.write("Error: {}\n".format(message))
    raise SystemExit(code)


def require_numpy():
    try:
        import numpy
    except ImportError:
        die("missing dependency numpy. Install it first:  pip install numpy")
    return numpy


def strip_suffix(name: str, suffix: str) -> str:
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def local_name(tag: str) -> str:
    """Strip the XML namespace, keeping only the local name."""
    return tag.rsplit("}", 1)[-1]


def child_by_name(parent, name: str):
    if parent is None:
        return None
    for child in list(parent):
        if local_name(child.tag) == name:
            return child
    return None


def find_by_path(node, path: str):
    """Walk down by local name, e.g. 'platform/orbit'."""
    current = node
    for name in path.split("/"):
        current = child_by_name(current, name)
        if current is None:
            return None
    return current


def text_of(node, default=None):
    if node is None or node.text is None:
        return default
    value = node.text.strip()
    return value if value else default


def set_child_text(parent, name: str, value: str):
    """Set a child element's text, creating it when absent."""
    child = child_by_name(parent, name)
    if child is None:
        child = ET.SubElement(parent, name)
    child.text = value
    return child


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z?$"
)


def parse_utc(text):
    """'2023-07-01T02:34:56.123456Z' -> Unix seconds (float); None on failure."""
    if text is None:
        return None
    match = _TIME_RE.match(str(text).strip())
    if not match:
        return None
    fraction = (match.group(3) or "")[:6].ljust(6, "0")
    try:
        moment = datetime.datetime.strptime(
            "{} {}.{}".format(match.group(1), match.group(2), fraction),
            "%Y-%m-%d %H:%M:%S.%f",
        )
    except ValueError:
        return None
    return moment.replace(tzinfo=datetime.timezone.utc).timestamp()


def format_utc(seconds) -> str:
    """Unix seconds -> 'YYYY-MM-DDTHH:MM:SS.000000' (UTC, fixed 6 decimals)."""
    moment = datetime.datetime.fromtimestamp(
        int(round(float(seconds))), tz=datetime.timezone.utc
    )
    return moment.strftime("%Y-%m-%dT%H:%M:%S") + ".000000"


# ---------------------------------------------------------------------------
# Numeric kernel: temporally weighted RANSAC + WLS cubic fit
# ---------------------------------------------------------------------------
def robust_orbit_reconstruct(
    t_obs,
    pos_obs,
    t_start: float,
    t_stop: float,
    n_points: int = DEFAULT_POINTS,
    pre_roll: float = DEFAULT_PRE_ROLL,
    sigma_t: float = DEFAULT_SIGMA_T,
    iterations: int = DEFAULT_ITERS,
    tau: float = DEFAULT_TAU,
    seed=None,
):
    """Robust orbit reconstruction.

    Returns a dict:
        t_out   (n,)        integer-second epochs
        pos_out (n, 3)      positions (m)
        vel_out (n, 3)      analytic velocities (m/s)
        stats   dict        inliers / total / rms / max / span
    """
    np = require_numpy()

    t_obs = np.asarray(t_obs, dtype=float).ravel()
    pos_obs = np.asarray(pos_obs, dtype=float)
    if pos_obs.ndim == 1:
        pos_obs = pos_obs.reshape(1, -1)

    radius = np.linalg.norm(pos_obs, axis=1)
    valid = (radius > MIN_ORBIT_RADIUS) & np.all(np.isfinite(pos_obs), axis=1)
    if int(np.sum(valid)) < 4:
        die("fewer than 4 valid telemetry vectors (non-zero/finite check); cannot fit an orbit.")

    t_clean = t_obs[valid]
    pos_clean = pos_obs[valid]

    # Temporal Gaussian weight: 1 inside the scene, exp(-dt^2 / 2 sigma^2) outside
    dt = np.minimum(np.abs(t_clean - t_start), np.abs(t_clean - t_stop))
    inside = (t_clean >= t_start) & (t_clean <= t_stop)
    dt[inside] = 0.0
    weights = np.exp(-(dt ** 2) / (2.0 * sigma_t ** 2))
    if float(np.sum(weights)) <= 0.0:
        die("all temporal weights underflowed to 0; increase --sigma-t.")

    n_obs = len(t_clean)
    if seed is not None:
        np.random.seed(int(seed))

    best_inliers = None
    best_score = -1.0
    sampling_p = weights / np.sum(weights)

    for _ in range(max(1, iterations)):
        if n_obs > 4:
            sample_idx = np.random.choice(n_obs, 4, replace=False, p=sampling_p)
        else:
            sample_idx = np.arange(n_obs)

        t_sample = t_clean[sample_idx]
        p_sample = pos_clean[sample_idx]
        try:
            design_s = np.vander(t_sample - t_start, 4)
            beta = np.linalg.solve(design_s, p_sample)
        except np.linalg.LinAlgError:
            continue

        design_all = np.vander(t_clean - t_start, 4)
        predicted = np.dot(design_all, beta)
        residuals = np.linalg.norm(pos_clean - predicted, axis=1)
        inliers = residuals <= tau
        score = float(np.sum(weights[inliers]))
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or int(np.sum(best_inliers)) < 4:
        die("RANSAC found no consistent consensus set; relax --tau or check the epochs.")

    # Final refinement: weighted least squares
    t_in = t_clean[best_inliers]
    pos_in = pos_clean[best_inliers]
    w_in = weights[best_inliers]

    design_in = np.vander(t_in - t_start, 4)
    weight_mat = np.diag(w_in)
    try:
        normal = np.dot(design_in.T, np.dot(weight_mat, design_in))
        beta_final = np.linalg.solve(
            normal, np.dot(design_in.T, np.dot(weight_mat, pos_in))
        )
    except np.linalg.LinAlgError:
        die("singular matrix in the final weighted least squares.")

    # np.vander(x, 4) columns are [x^3, x^2, x, 1]
    a3 = beta_final[0, :]
    a2 = beta_final[1, :]
    a1 = beta_final[2, :]
    a0 = beta_final[3, :]

    t_clean_min = float(np.min(t_clean))
    t_clean_max = float(np.max(t_clean))

    t_first = float(t_start) - float(pre_roll)
    t_out = np.round(t_first + np.arange(int(n_points))).astype(np.int64)

    pos_out = np.zeros((len(t_out), 3), dtype=float)
    vel_out = np.zeros((len(t_out), 3), dtype=float)

    for index, moment in enumerate(t_out):
        t_rel = float(moment) - t_start
        vel_out[index, :] = 3.0 * a3 * (t_rel ** 2) + 2.0 * a2 * t_rel + a1

        if t_clean_min <= moment <= t_clean_max:
            # inside the observed span: cubic polynomial interpolation
            pos_out[index, :] = (
                a3 * (t_rel ** 3) + a2 * (t_rel ** 2) + a1 * t_rel + a0
            )
        elif moment < t_clean_min:
            # left extrapolation: anchor at the boundary, first-order tangent
            t_edge = t_clean_min - t_start
            pos_edge = a3 * (t_edge ** 3) + a2 * (t_edge ** 2) + a1 * t_edge + a0
            vel_edge = 3.0 * a3 * (t_edge ** 2) + 2.0 * a2 * t_edge + a1
            pos_out[index, :] = pos_edge + vel_edge * (moment - t_clean_min)
        else:
            # right extrapolation: same as above
            t_edge = t_clean_max - t_start
            pos_edge = a3 * (t_edge ** 3) + a2 * (t_edge ** 2) + a1 * t_edge + a0
            vel_edge = 3.0 * a3 * (t_edge ** 2) + 2.0 * a2 * t_edge + a1
            pos_out[index, :] = pos_edge + vel_edge * (moment - t_clean_max)

    residual = np.linalg.norm(
        pos_clean[best_inliers] - np.dot(design_in, beta_final), axis=1
    )
    stats = {
        "total": int(n_obs),
        "inliers": int(np.sum(best_inliers)),
        "rms": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else 0.0,
        "max": float(np.max(residual)) if len(residual) else 0.0,
        "span": (t_clean_min, t_clean_max),
    }
    return {"t_out": t_out, "pos_out": pos_out, "vel_out": vel_out, "stats": stats}


# ---------------------------------------------------------------------------
# Subcommand 1: correct-orbit
# ---------------------------------------------------------------------------
def read_orbit_state_vectors(orbit_node):
    """Read the stateVec list -> ([t...], [[x,y,z]...], skipped)."""
    times = []
    positions = []
    skipped = 0
    for node in list(orbit_node):
        if local_name(node.tag) != "stateVec":
            continue
        moment = parse_utc(text_of(child_by_name(node, "timeUTC")))
        if moment is None:
            skipped += 1
            continue
        try:
            xyz = [
                float(text_of(child_by_name(node, axis)))
                for axis in ("posX", "posY", "posZ")
            ]
        except (TypeError, ValueError):
            skipped += 1
            continue
        times.append(moment)
        positions.append(xyz)
    return times, positions, skipped


def cmd_correct_orbit(args) -> int:
    require_numpy()
    xml_path = Path(args.xml).expanduser().resolve()
    if not xml_path.is_file():
        die("XML file not found: {}".format(xml_path))

    print(" ")
    print("#####################################################################")
    print("rweke.py correct-orbit: Scene-Covering Window Correction for LT-1")
    print("#####################################################################")
    print(" ")

    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as exc:
        die("XML parse failed: {}".format(exc))
    root = tree.getroot()

    if local_name(root.tag) == "level1Product":
        level1 = root
    else:
        level1 = child_by_name(root, "level1Product")
    if level1 is None:
        die("level1Product not found under the root element.")

    scene_info = find_by_path(level1, "productInfo/sceneInfo")
    start_text = text_of(find_by_path(scene_info, "start/timeUTC"))
    stop_text = text_of(find_by_path(scene_info, "stop/timeUTC"))
    t_start = parse_utc(start_text)
    t_stop = parse_utc(stop_text)
    if t_start is None or t_stop is None:
        die("cannot parse the scene start/stop time (productInfo/sceneInfo/{start,stop}/timeUTC).")

    orbit = find_by_path(level1, "platform/orbit")
    if orbit is None:
        die("platform/orbit not found.")

    times, positions, skipped = read_orbit_state_vectors(orbit)
    print("telemetry vectors: {} valid / {} skipped".format(len(times), skipped))
    if len(times) < 4:
        die("fewer than 4 valid stateVec entries; cannot reconstruct the orbit.")

    result = robust_orbit_reconstruct(
        times,
        positions,
        t_start=t_start,
        t_stop=t_stop,
        n_points=args.points,
        pre_roll=args.pre_roll,
        sigma_t=args.sigma_t,
        iterations=args.iterations,
        tau=args.tau,
        seed=args.seed,
    )
    t_out = result["t_out"]
    pos_out = result["pos_out"]
    vel_out = result["vel_out"]
    stats = result["stats"]

    print("Scene Range: {} -> {}".format(start_text, stop_text))
    print(
        "Selected Window: {} -> {} ({} points, RANSAC+WLS)".format(
            format_utc(t_out[0]), format_utc(t_out[-1]), len(t_out)
        )
    )
    print(
        "Fit quality: inliers {}/{}  RMS={:.4f} m  MAX={:.4f} m".format(
            stats["inliers"], stats["total"], stats["rms"], stats["max"]
        )
    )
    print(
        "Observed span: {} -> {}".format(
            format_utc(stats["span"][0]), format_utc(stats["span"][1])
        )
    )

    if args.dry_run:
        print("[dry-run] diagnostics only, nothing written.")
        return 0

    # --- write back orbitHeader ---
    header = child_by_name(orbit, "orbitHeader")
    if header is None:
        header = ET.SubElement(orbit, "orbitHeader")
    set_child_text(header, "numStateVectors", str(len(t_out)))

    first_node = child_by_name(header, "firstStateTime")
    if first_node is None:
        first_node = ET.SubElement(header, "firstStateTime")
    set_child_text(first_node, "firstStateTimeUTC", format_utc(t_out[0]))

    last_node = child_by_name(header, "lastStateTime")
    if last_node is None:
        last_node = ET.SubElement(header, "lastStateTime")
    set_child_text(last_node, "lastStateTimeUTC", format_utc(t_out[-1]))

    # --- write back stateVec (clone when short, drop when long) ---
    vectors = [node for node in list(orbit) if local_name(node.tag) == "stateVec"]
    while len(vectors) < len(t_out):
        clone = copy.deepcopy(vectors[-1])
        orbit.append(clone)
        vectors.append(clone)

    print("Updating existing XML nodes in-place ...")
    for index in range(len(t_out)):
        node = vectors[index]
        node.set("num", str(index + 1))
        set_child_text(node, "timeUTC", format_utc(t_out[index]))
        for axis_index, axis in enumerate(("posX", "posY", "posZ")):
            set_child_text(node, axis, FLOAT_FMT.format(pos_out[index][axis_index]))
        for axis_index, axis in enumerate(("velX", "velY", "velZ")):
            set_child_text(node, axis, FLOAT_FMT.format(vel_out[index][axis_index]))

    removed = 0
    for node in vectors[len(t_out):]:
        orbit.remove(node)
        removed += 1
    if removed:
        print("removed {} surplus stateVec nodes".format(removed))

    target = Path(args.output).expanduser().resolve() if args.output else xml_path
    if target == xml_path and not args.no_backup:
        stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        backup = xml_path.with_name("{}.bak.{}".format(xml_path.name, stamp))
        shutil.copy2(str(xml_path), str(backup))
        print("backup written: {}".format(backup.name))

    tree.write(str(target), encoding="UTF-8", xml_declaration=True)
    print(
        "Success: orbit trimmed to {} points covering the scene, high-precision notation kept.".format(
            len(t_out)
        )
    )
    print("output file: {}".format(target))
    return 0


# ---------------------------------------------------------------------------
# Subcommand 2: delta-lonlat
# ---------------------------------------------------------------------------
_PARSE_ROLES = ("import", "raw", "correct")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def read_par_dict(par_path: Path) -> dict:
    params = {}
    with open(str(par_path), "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            params[key.strip()] = value.strip()
    return params


def first_number(text):
    if text is None:
        return None
    match = _NUMBER_RE.search(str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def read_center(par_path: Path):
    """Return (lat, lon); (None, None) when missing."""
    params = read_par_dict(par_path)
    lat = first_number(params.get("center_latitude"))
    lon = first_number(params.get("center_longitude"))
    return lat, lon


def group_par_files(folder: Path):
    """Group by scene prefix into {prefix: {role: (lat, lon)}}."""
    groups = {}
    for par_path in sorted(folder.glob("*.par")):
        name = par_path.name
        if ".slc_" not in name:
            continue
        prefix, rest = name.split(".slc_", 1)
        role = strip_suffix(rest, ".par")
        if role not in _PARSE_ROLES:
            continue
        lat, lon = read_center(par_path)
        if lat is None or lon is None:
            continue
        groups.setdefault(prefix, {})[role] = (lat, lon)
    return groups


def deg_to_meters(dlat, dlon, ref_lat):
    """Degree differences -> metres (longitude scaled by the reference latitude)."""
    lat_m = dlat * 111320.0
    lon_m = dlon * 111320.0 * math.cos(math.radians(ref_lat))
    return lat_m, lon_m


def describe(values, label: str) -> str:
    clean = [value for value in values if value is not None]
    if not clean:
        return "  {:<20s} no data".format(label)
    if len(clean) == 1:
        return "  {:<20s} mean={:.3f} m (single sample, no std)".format(
            label, clean[0]
        )
    return "  {:<20s} mean={:.3f} m, std={:.3f} m".format(
        label, statistics.mean(clean), statistics.stdev(clean)
    )


def write_table(rows, columns, base_stem: Path, fmt: str = "auto"):
    """Prefer xlsx (needs pandas), fall back to csv. Returns the actual path."""
    pandas_module = None
    if fmt in ("auto", "xlsx"):
        try:
            import pandas as pandas_module  # noqa: F401
        except ImportError:
            pandas_module = None
    if pandas_module is not None:
        frame = pandas_module.DataFrame(rows, columns=columns)
        path = base_stem.with_suffix(".xlsx")
        frame.to_excel(str(path), index=False)
        return path
    if fmt == "xlsx":
        print("warning: pandas/openpyxl not installed, falling back to CSV.")
    path = base_stem.with_suffix(".csv")
    with open(str(path), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(column, "") for column in columns])
    return path


def cmd_delta_lonlat(args) -> int:
    folder = Path(args.dir).expanduser().resolve()
    if not folder.is_dir():
        die("directory not found: {}".format(folder))

    groups = group_par_files(folder)
    if not groups:
        die("no file matching '*.slc_<import|raw|correct>.par' in: {}".format(folder))

    columns = [
        "Group_Index",
        "File_Name",
        "Raw dLat (m)",
        "Raw dLon (m)",
        "Correct dLat (m)",
        "Correct dLon (m)",
    ]
    records = []

    all_raw_lat, all_raw_lon = [], []
    all_cor_lat, all_cor_lon = [], []
    ex_raw_lat, ex_raw_lon = [], []
    ex_cor_lat, ex_cor_lon = [], []
    raw_exceed = correct_exceed = both_exceed = 0

    for index, prefix in enumerate(sorted(groups), start=1):
        roles = groups[prefix]
        if "import" not in roles:
            continue
        ref_lat, ref_lon = roles["import"]

        raw_lat_m = raw_lon_m = None
        cor_lat_m = cor_lon_m = None
        raw_flag = cor_flag = False

        if "raw" in roles:
            raw_lat_m, raw_lon_m = deg_to_meters(
                roles["raw"][0] - ref_lat, roles["raw"][1] - ref_lon, ref_lat
            )
            all_raw_lat.append(raw_lat_m)
            all_raw_lon.append(raw_lon_m)
            if abs(raw_lat_m) >= args.threshold or abs(raw_lon_m) >= args.threshold:
                raw_flag = True
                ex_raw_lat.append(raw_lat_m)
                ex_raw_lon.append(raw_lon_m)

        if "correct" in roles:
            cor_lat_m, cor_lon_m = deg_to_meters(
                roles["correct"][0] - ref_lat, roles["correct"][1] - ref_lon, ref_lat
            )
            all_cor_lat.append(cor_lat_m)
            all_cor_lon.append(cor_lon_m)
            if abs(cor_lat_m) >= args.threshold or abs(cor_lon_m) >= args.threshold:
                cor_flag = True
                ex_cor_lat.append(cor_lat_m)
                ex_cor_lon.append(cor_lon_m)

        raw_exceed += 1 if raw_flag else 0
        correct_exceed += 1 if cor_flag else 0
        if raw_flag and cor_flag:
            both_exceed += 1
            print("\nGroup {} exceeds on both:".format(prefix))
            print("  raw:     dLat={:.3f} m, dLon={:.3f} m".format(raw_lat_m, raw_lon_m))
            print("  correct: dLat={:.3f} m, dLon={:.3f} m".format(cor_lat_m, cor_lon_m))

        records.append(
            {
                "Group_Index": index,
                "File_Name": prefix,
                "Raw dLat (m)": raw_lat_m,
                "Raw dLon (m)": raw_lon_m,
                "Correct dLat (m)": cor_lat_m,
                "Correct dLon (m)": cor_lon_m,
            }
        )

    print("\nTotal groups: {}".format(len(groups)))
    print("groups over threshold (raw): {}".format(raw_exceed))
    print("groups over threshold (correct): {}".format(correct_exceed))
    print("groups over threshold on both: {}".format(both_exceed))
    print("\n=== All offsets (metres) ===")
    print(describe(all_raw_lat, "all_raw_lat_m"))
    print(describe(all_raw_lon, "all_raw_lon_m"))
    print(describe(all_cor_lat, "all_correct_lat_m"))
    print(describe(all_cor_lon, "all_correct_lon_m"))
    print("\n=== Offsets above the threshold (metres) ===")
    print(describe(ex_raw_lat, "exceed_raw_lat_m"))
    print(describe(ex_raw_lon, "exceed_raw_lon_m"))
    print(describe(ex_cor_lat, "exceed_correct_lat_m"))
    print(describe(ex_cor_lon, "exceed_correct_lon_m"))

    out_dir = Path(args.outdir).expanduser().resolve() if args.outdir else folder
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = write_table(
        records, columns, out_dir / "{}_raw_results".format(args.prefix), args.format
    )
    print("\nRaw offset table written to {}".format(full_path.name))

    # Drop outliers: keep rows whose raw offsets stay within mean +/- std
    raw_lat_values = [
        row["Raw dLat (m)"] for row in records if row["Raw dLat (m)"] is not None
    ]
    raw_lon_values = [
        row["Raw dLon (m)"] for row in records if row["Raw dLon (m)"] is not None
    ]
    if len(raw_lat_values) >= 2 and len(raw_lon_values) >= 2:
        lat_mean = statistics.mean(raw_lat_values)
        lat_std = statistics.stdev(raw_lat_values)
        lon_mean = statistics.mean(raw_lon_values)
        lon_std = statistics.stdev(raw_lon_values)
        filtered = [
            row
            for row in records
            if row["Raw dLat (m)"] is not None
            and row["Raw dLon (m)"] is not None
            and lat_mean - lat_std <= row["Raw dLat (m)"] <= lat_mean + lat_std
            and lon_mean - lon_std <= row["Raw dLon (m)"] <= lon_mean + lon_std
        ]
        filtered_path = write_table(
            filtered,
            columns,
            out_dir / "{}_filtered_results".format(args.prefix),
            args.format,
        )
        print(
            "kept {}/{} rows after outlier removal, written to {}".format(
                len(filtered), len(records), filtered_path.name
            )
        )
    else:
        print("fewer than 2 raw samples, skipping outlier removal.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXAMPLES = """\
Examples
--------
# 1) fix the orbit window (in place, with a backup)
python rweke.py correct-orbit scene.xml

# 2) diagnostics only, nothing written
python rweke.py correct-orbit scene.xml --dry-run

# 3) custom point count and pre-roll, write to a new file
python rweke.py correct-orbit scene.xml --points 64 --pre-roll 30 \\
    --output scene_fixed.xml

# 4) report centre offsets before and after correction
python rweke.py delta-lonlat --dir ./Path_01_ASC --threshold 3.0
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rweke.py",
        description="RW-EKE: robust LT-1 orbit-window reconstruction and geometric offset evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + VERSION)
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    p_orbit = subparsers.add_parser(
        "correct-orbit",
        help="robustly reconstruct and rewrite LT-1 orbit stateVec entries",
        description="Temporally weighted RANSAC + WLS orbit fit, trimmed to a fixed point count covering the scene.",
    )
    p_orbit.add_argument("xml", help="LT-1 L1A product XML file")
    p_orbit.add_argument(
        "--points", type=int, default=DEFAULT_POINTS,
        help="number of stateVec points to write (default 40)",
    )
    p_orbit.add_argument(
        "--pre-roll", type=float, default=DEFAULT_PRE_ROLL,
        help="seconds before the scene start for the first epoch (default 15)",
    )
    p_orbit.add_argument(
        "--sigma-t", type=float, default=DEFAULT_SIGMA_T,
        help="temporal Gaussian weight std in seconds (default 20)",
    )
    p_orbit.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERS,
        help="RANSAC iterations (default 80)",
    )
    p_orbit.add_argument(
        "--tau", type=float, default=DEFAULT_TAU,
        help="RANSAC inlier threshold in metres (default 0.1)",
    )
    p_orbit.add_argument("--seed", type=int, default=None, help="random seed (for reproducibility)")
    p_orbit.add_argument("--output", default=None, help="output XML (default: rewrite in place)")
    p_orbit.add_argument(
        "--no-backup", action="store_true", help="do not create a .bak file when rewriting in place"
    )
    p_orbit.add_argument(
        "--dry-run", action="store_true", help="parse and fit only, write nothing"
    )
    p_orbit.set_defaults(func=cmd_correct_orbit)

    p_delta = subparsers.add_parser(
        "delta-lonlat",
        help="centre offsets of import/raw/correct parameter files",
        description="Compare raw/correct centre coordinates against import (metres); print statistics and tables.",
    )
    p_delta.add_argument(
        "--dir", default=".", help="directory containing *.slc_*.par (default: current directory)"
    )
    p_delta.add_argument(
        "--threshold", type=float, default=3.0, help="over-threshold test in metres (default 3.0)"
    )
    p_delta.add_argument("--outdir", default=None, help="table output directory (defaults to --dir)")
    p_delta.add_argument("--prefix", default="offset", help="output filename prefix (default offset)")
    p_delta.add_argument(
        "--format", choices=("auto", "xlsx", "csv"), default="auto",
        help="table format: auto=xlsx when pandas is present, else csv (default auto)",
    )
    p_delta.set_defaults(func=cmd_delta_lonlat)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

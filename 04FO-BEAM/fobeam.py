#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FO-BEAM: interferogram -> differential -> filter -> unwrap -> detrend -> atm -> geocode

9 steps
--------
1 prepare   symlink the workspace, mapping slc and dem files to common names
2 offset    initial registration offsets; read back the interferogram size
3 multilook multi-look plus amplitude quicklooks
4 diff      differential interferogram: orb (orbit-based simulation, default) or fft
5 adf       1-3 stage adaptive filtering (windows 128 / 64 / 32)
6 unwrap    coherence mask + minimum-cost-flow unwrapping + unwrapping model
7 detrend   quadratic trend surface fit and removal
8 atm       2D power-law atmosphere phase estimate and removal (--skip-atm)
9 geocode   dispmap → rashgt → geocode_back → GeoTIFF

Each step checks its outputs and is skipped when already complete (resumable).

Dependencies
----
python >= 3.8
External commands: create_offset, multi_look, raspwr, phase_sim_orb, SLC_diff_intf,
SLC_intf, rasmph_pwr, base_est_fft, phase_sim, create_diff_par, sub_phase,
ph_slope_base, adf, rascc, rascc_mask, mcf, rasrmg, unw_model, quad_fit,
quad_sub, atm_mod_2d, atm_sim_2d, dispmap, rashgt, geocode_back, rasdt_cmap,
data2geotiff

Examples
----
python fobeam.py --master 20230820 --slave 20230913 --slc-dir slc --dem-dir dem
python fobeam.py --master 20230820 --slave 20230913 --dry-run
python fobeam.py --master 20230820 --slave 20230913 --skip-atm
python fobeam.py --master 20230820 --slave 20230913 \
    --diff-engine fft --displacement vertical
"""


from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

VERSION = "1.0.0"

TOTAL_STEPS = 9

# External command list: used for dependency hints and missing checks
EXTERNAL_COMMANDS = (
    "create_offset", "multi_look", "raspwr", "phase_sim_orb", "SLC_diff_intf",
    "SLC_intf", "rasmph_pwr", "base_est_fft", "phase_sim", "create_diff_par",
    "sub_phase", "ph_slope_base", "adf", "rascc", "rascc_mask", "mcf",
    "rasrmg", "unw_model", "quad_fit", "quad_sub", "atm_mod_2d", "atm_sim_2d",
    "dispmap", "rashgt", "geocode_back", "rasdt_cmap", "data2geotiff",
)


def die(message: str, code: int = 2) -> "None":
    sys.stderr.write("Error: {}\n".format(message))
    raise SystemExit(code)


def read_par_value(par_path, key: str, default=None):
    """Read the first value of a key from a .par file."""
    path = Path(par_path)
    if not path.is_file():
        return default
    with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(key):
                remainder = stripped[len(key):].lstrip(": ").strip()
                fields = remainder.split()
                if fields:
                    return fields[0]
    return default


class InterferogramPipeline:
    """Interferogram pipeline driver."""

    def __init__(self, args):
        self.args = args
        self.workdir = Path(args.workdir).expanduser().resolve()
        self.slc_dir = Path(args.slc_dir).expanduser().resolve()
        self.dem_dir = Path(args.dem_dir).expanduser().resolve()
        self.master = str(args.master)
        self.slave = str(args.slave)
        self.pair = "{}_{}".format(self.master, self.slave)
        self.width = "WIDTH"
        self.lines = "LINES"

    # ---------------- basic helpers ----------------

    def path(self, name: str) -> Path:
        return self.workdir / name

    def name(self, name: str) -> str:
        """Return a name relative to the working directory (commands run there)."""
        return name

    def run(self, argv) -> None:
        """Run one external command."""
        command = [str(item) for item in argv]
        printable = " ".join(command)
        if self.args.dry_run:
            print("    [dry-run] {}".format(printable))
            return
        proc = subprocess.run(
            command,
            cwd=str(self.workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            die(
                "command failed (exit code {}): {}\n{}".format(
                    proc.returncode, printable, proc.stderr.strip()
                )
            )

    def link(self, source: Path, target_name: str) -> None:
        """Symlink source into the working directory (skipped when present)."""
        target = self.path(target_name)
        if target.exists() or target.is_symlink():
            return
        if self.args.dry_run:
            print("    [dry-run] ln -s {} {}".format(source, target_name))
            return
        try:
            target.symlink_to(source)
        except (OSError, NotImplementedError):
            shutil.copy2(str(source), str(target))

    def step(self, index: int, title: str, outputs=()) -> bool:
        """Print the step header; return False when its outputs already exist."""
        print("\n[{}/{}] {}".format(index, TOTAL_STEPS, title))
        if self.args.resume and not self.args.force and outputs:
            missing = [name for name in outputs if not self.path(name).exists()]
            if not missing:
                print("    outputs exist, skipping (use --force to rerun).")
                return False
        return True

    # ---------------- input checks ----------------

    def check_inputs(self) -> None:
        if self.args.dry_run:
            return
        required = [
            self.slc_dir / "{}.vv.slc".format(self.master),
            self.slc_dir / "{}.vv.slc.par".format(self.master),
            self.slc_dir / "{}.vv.slc".format(self.slave),
            self.slc_dir / "{}.vv.slc.par".format(self.slave),
        ]
        missing = [str(item) for item in required if not item.exists()]
        if missing:
            die("missing SLC input files:\n  " + "\n  ".join(missing))

        if not self.dem_dir.is_dir():
            die("DEM directory not found: {}".format(self.dem_dir))
        dem_patterns = ("*.hgt_sim", "*.lt_fine", "*.mli.utm", "*.utm.dem.par")
        for pattern in dem_patterns:
            if not list(self.dem_dir.glob(pattern)):
                die("DEM directory has no '{}': {}".format(pattern, self.dem_dir))

        available = [cmd for cmd in EXTERNAL_COMMANDS if shutil.which(cmd)]
        if not available:
            die(
                "no interferometry command found on PATH (e.g. create_offset).\n"
                "Add the tool bin directory to PATH first."
            )

    # ---------------- steps ----------------

    def step_prepare(self) -> None:
        if not self.step(1, "prepare: symlink the workspace"):
            return
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.link(self.slc_dir / "{}.vv.slc".format(self.master), "{}.slc".format(self.master))
        self.link(
            self.slc_dir / "{}.vv.slc.par".format(self.master),
            "{}.slc.par".format(self.master),
        )
        self.link(self.slc_dir / "{}.vv.slc".format(self.slave), "{}.slc".format(self.slave))
        self.link(
            self.slc_dir / "{}.vv.slc.par".format(self.slave),
            "{}.slc.par".format(self.slave),
        )
        self.link(self.first_dem("*.hgt_sim"), "ref.hgt_sim")
        self.link(self.first_dem("*.lt_fine"), "ref.lt_fine")
        self.link(self.first_dem("*.mli.utm"), "ref.mli.utm")
        self.link(self.first_dem("*.utm.dem.par"), "ref.utm.dem.par")
        print("    symlinks ready.")

    def first_dem(self, pattern: str) -> Path:
        """First matching DEM file; raises a readable error when absent."""
        matches = sorted(self.dem_dir.glob(pattern))
        if not matches:
            if self.args.dry_run:
                # dry-run only prints commands, no real file needed
                return self.dem_dir / pattern
            die("DEM directory has no '{}': {}".format(pattern, self.dem_dir))
        return matches[0]

    def step_offset(self) -> None:
        if not self.step(2, "offset: initial registration offsets", ["{}.off".format(self.pair)]):
            return
        self.run(
            [
                "create_offset",
                "{}.slc.par".format(self.master),
                "{}.slc.par".format(self.slave),
                "{}.off".format(self.pair),
                1, self.args.rlks, self.args.alks, 0,
            ]
        )
        width = read_par_value(self.path("{}.off".format(self.pair)), "interferogram_width")
        lines = read_par_value(
            self.path("{}.off".format(self.pair)), "interferogram_azimuth_lines"
        )
        if width:
            self.width = width
        if lines:
            self.lines = lines
        print("    interferogram size: width={} lines={}".format(self.width, self.lines))

    def step_multilook(self) -> None:
        if not self.step(
            3,
            "multilook: multi-look and amplitude quicklooks",
            ["{}.mli".format(self.master), "{}.mli".format(self.slave)],
        ):
            return
        for scene in (self.master, self.slave):
            self.run(
                [
                    "multi_look",
                    "{}.slc".format(scene), "{}.slc.par".format(scene),
                    "{}.mli".format(scene), "{}.mli.par".format(scene),
                    self.args.rlks, self.args.alks,
                ]
            )
            self.run(
                [
                    "raspwr", "{}.mli".format(scene), self.width,
                    "-", "-", "-", "-", "-", "-", "-",
                    "{}.mli.bmp".format(scene),
                ]
            )

    def step_diff(self) -> None:
        if not self.step(
            4,
            "diff: differential interferogram ({})".format(self.args.diff_engine),
            ["{}.diff".format(self.pair)],
        ):
            return

        if self.args.diff_engine == "orb":
            # simulate terrain phase from the orbit + DEM, then subtract from the interferogram
            self.run(
                [
                    "phase_sim_orb",
                    "{}.slc.par".format(self.master), "{}.slc.par".format(self.slave),
                    "{}.off".format(self.pair), "ref.hgt_sim",
                    "{}.sim_unw".format(self.pair), "{}.slc.par".format(self.master),
                    "-", "-", 1, 1,
                ]
            )
            self.run(
                [
                    "rasrmg", "{}.sim_unw".format(self.pair), "{}.mli".format(self.master),
                    self.width, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-",
                    "{}.sim_unw.bmp".format(self.pair),
                ]
            )
            self.run(
                [
                    "SLC_diff_intf",
                    "{}.slc".format(self.master), "{}.slc".format(self.slave),
                    "{}.slc.par".format(self.master), "{}.slc.par".format(self.slave),
                    "{}.off".format(self.pair), "{}.sim_unw".format(self.pair),
                    "{}.diff".format(self.pair),
                    self.args.rlks, self.args.alks, 0, 0, "-", "-", "-",
                ]
            )
            self.run(
                [
                    "rasmph_pwr", "{}.diff".format(self.pair), "{}.mli".format(self.master),
                    self.width, "-", "-", "-", "-", "-", "-", "-", "-",
                    "{}.diff.bmp".format(self.pair),
                ]
            )
            return

        # --- fft: spectral baseline estimation ---
        self.run(
            [
                "SLC_intf",
                "{}.slc".format(self.master), "{}.slc".format(self.slave),
                "{}.slc.par".format(self.master), "{}.slc.par".format(self.slave),
                "{}.off".format(self.pair), "{}.int".format(self.pair),
                self.args.rlks, self.args.alks, "-", "-", 0, 0,
            ]
        )
        self.run(
            [
                "rasmph_pwr", "{}.int".format(self.pair), "{}.mli".format(self.master),
                self.width, "-", "-", "-", "-", "-", "-", "-", "-",
                "{}.int_mli.bmp".format(self.pair),
            ]
        )
        self.run(
            [
                "base_est_fft", "{}.int".format(self.pair),
                "{}.slc.par".format(self.master), "{}.off".format(self.pair),
                "{}.base".format(self.pair),
            ]
        )
        self.run(
            [
                "phase_sim", "{}.slc.par".format(self.master), "{}.off".format(self.pair),
                "{}.base".format(self.pair), "ref.hgt_sim",
                "{}.sim_unw".format(self.pair),
            ]
        )
        self.run(
            [
                "rasrmg", "{}.sim_unw".format(self.pair), "{}.mli".format(self.master),
                self.width, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-",
                "{}.sim_unw.bmp".format(self.pair),
            ]
        )
        self.run(["create_diff_par", "{}.mli.par".format(self.master), "-", "diff_par", 1, 0])
        self.run(
            [
                "sub_phase", "{}.int".format(self.pair),
                "{}.sim_unw".format(self.pair), "diff_par",
                "{}.diff".format(self.pair), 1,
            ]
        )
        self.run(
            [
                "rasmph_pwr", "{}.diff".format(self.pair), "{}.mli".format(self.master),
                self.width, "-", "-", "-", "-", "-", "-", "-", "-",
                "{}.diff.bmp".format(self.pair),
            ]
        )
        self.run(
            [
                "ph_slope_base", "{}.int".format(self.pair),
                "{}.slc.par".format(self.master), "{}.off".format(self.pair),
                "{}.base".format(self.pair), "{}.flt".format(self.pair),
            ]
        )
        self.run(
            [
                "rasmph_pwr", "{}.flt".format(self.pair), "{}.mli".format(self.master),
                self.width, "-", "-", "-", "-", "-", "-", "-", "-",
                "{}.flt_mli.bmp".format(self.pair),
            ]
        )
        print("    note: the fft branch leaves the flat-earth interferogram in {}.flt.".format(self.pair))

    def step_adf(self) -> None:
        epoch = int(self.args.adf_epoch)
        if not self.step(
            5,
            "adf: adaptive filtering ({} stage)".format(epoch),
            ["{}.diff_sm3".format(self.pair), "{}.smcc3".format(self.pair)],
        ):
            return

        cc_out = self.args.adf_cc
        width = self.width

        def adf_once(source, out, cc, radius):
            self.run(
                [
                    "adf", source, out, cc, width,
                    ".2", radius, 7, 8, 0, 0, cc_out,
                ]
            )
            self.run(
                [
                    "rasmph_pwr", out, "{}.mli".format(self.master), width,
                    "-", "-", "-", "-", "-", "-", "-", "-",
                    "{}.bmp".format(out),
                ]
            )
            self.run(
                [
                    "rascc", cc, "{}.mli".format(self.master), width,
                    "-", "-", "-", "-", "-", "-", "-", "-", "-",
                    "{}.bmp".format(cc),
                ]
            )

        if epoch == 1:
            adf_once(
                "{}.diff".format(self.pair),
                "{}.diff_sm3".format(self.pair),
                "{}.smcc3".format(self.pair),
                128,
            )
            return

        adf_once(
            "{}.diff".format(self.pair),
            "{}.diff_sm1".format(self.pair),
            "{}.smcc1".format(self.pair),
            128,
        )
        if epoch == 2:
            adf_once(
                "{}.diff_sm1".format(self.pair),
                "{}.diff_sm3".format(self.pair),
                "{}.smcc3".format(self.pair),
                64,
            )
            return

        # epoch == 3: three-stage cascade
        adf_once(
            "{}.diff_sm1".format(self.pair),
            "{}.diff_sm2".format(self.pair),
            "{}.smcc2".format(self.pair),
            64,
        )
        adf_once(
            "{}.diff_sm2".format(self.pair),
            "{}.diff_sm3".format(self.pair),
            "{}.smcc3".format(self.pair),
            32,
        )

    def step_unwrap(self) -> None:
        if not self.step(
            6,
            "unwrap: phase unwrapping",
            ["{}.diff_unw".format(self.pair)],
        ):
            return
        print("    unwrapping seed (rpos, azpos) = ({}, {})".format(self.args.rpos, self.args.azpos))
        self.run(
            [
                "rascc_mask", "{}.smcc3".format(self.pair),
                "{}.mli".format(self.master), self.width,
                1, 1, 0, 1, 1,
                self.args.cc_thres, self.args.pwr_thres, 0.1, 0.9, 1, 0.35, 1,
                "{}.unw.mask.bmp".format(self.pair),
            ]
        )
        self.run(
            [
                "mcf", "{}.diff_sm3".format(self.pair), "{}.smcc3".format(self.pair),
                "{}.unw.mask.bmp".format(self.pair), "{}.diff_unw0".format(self.pair),
                self.width, 0, 0, 0, "-", "-", 1, 1, "-",
                self.args.rpos, self.args.azpos, 1,
            ]
        )
        self.run(
            [
                "rasrmg", "{}.diff_unw0".format(self.pair),
                "{}.mli".format(self.master), self.width,
                1, 1, 0, 1, 1, 0.5, 1.0, 0.35, "-", 1,
            ]
        )
        self.run(
            [
                "unw_model", "{}.diff_sm3".format(self.pair),
                "{}.diff_unw0".format(self.pair), "{}.diff_unw".format(self.pair),
                self.width, "-", "-", "-", self.width,
            ]
        )
        self.run(
            [
                "rasrmg", "{}.diff_unw".format(self.pair),
                "{}.mli".format(self.master), self.width,
                1, 1, 0, 1, 1, 0.5, 1.0, 0.35, "-", 1,
            ]
        )

    def step_detrend(self) -> None:
        if not self.step(
            7,
            "detrend: remove the quadratic trend surface",
            ["{}.diff_quad".format(self.pair)],
        ):
            return
        self.run(["create_diff_par", "{}.mli.par".format(self.master), "-", "diff_par", 1, 0])
        self.run(
            [
                "quad_fit", "{}.diff_unw".format(self.pair), "diff_par",
                4, 4, "-", "{}.plot_data".format(self.pair), 0,
            ]
        )
        self.run(
            [
                "quad_sub", "{}.diff_unw".format(self.pair), "diff_par",
                "{}.diff_quad".format(self.pair), 1, 0,
            ]
        )
        self.run(
            [
                "rasrmg", "{}.diff_quad".format(self.pair),
                "{}.mli".format(self.master), self.width,
                "-", "-", "-", "-", "-", 0.1, "-", "-", "-", "-",
                "{}.diff.quad.bmp".format(self.pair),
            ]
        )

    def step_atm(self) -> None:
        if self.args.skip_atm:
            print("\n[{}/{}] atm: skipped (--skip-atm)".format(8, TOTAL_STEPS))
            if not self.args.dry_run:
                source = self.path("{}.diff_quad".format(self.pair))
                target = self.path("{}.diff.final".format(self.pair))
                if source.exists() and not target.exists():
                    shutil.copy2(str(source), str(target))
                    print("    using diff_quad as the final differential interferogram.")
            return

        if not self.step(
            8,
            "atm: 2D power-law atmosphere phase estimate and removal",
            ["{}.diff.final".format(self.pair)],
        ):
            return
        self.run(
            [
                "atm_mod_2d", "{}.diff_quad".format(self.pair), "ref.hgt_sim",
                "{}.smcc3".format(self.pair), "diff_par",
                "{}.unw.mask.bmp".format(self.pair), 1,
                "{}.atm_a0".format(self.pair), "{}.atm_a1".format(self.pair),
                "{}.atm_sigma".format(self.pair), "{}.atm_sigma_h".format(self.pair),
                "{}.atm_s1".format(self.pair),
            ]
        )
        self.run(
            [
                "atm_sim_2d", "diff_par", "ref.hgt_sim",
                "{}.atm_a0".format(self.pair), "{}.atm_a1".format(self.pair),
                "{}.atm".format(self.pair), "{}.unw.mask.bmp".format(self.pair),
            ]
        )
        self.run(
            [
                "rasrmg", "{}.atm".format(self.pair),
                "{}.mli".format(self.master), self.width,
                "-", "-", "-", "-", "-", 0.1, "-", "-", "-", "-",
                "{}.atm.bmp".format(self.pair),
            ]
        )
        self.run(
            [
                "sub_phase", "{}.diff_quad".format(self.pair),
                "{}.atm".format(self.pair), "diff_par",
                "{}.diff.final".format(self.pair), 0, "-",
            ]
        )
        self.run(
            [
                "rasrmg", "{}.diff.final".format(self.pair),
                "{}.mli".format(self.master), self.width,
                "-", "-", "-", "-", "-", 1, "-", "-", "-", "-",
                "{}.diff.final.bmp".format(self.pair),
            ]
        )

    def step_geocode(self) -> None:
        if not self.step(
            9,
            "geocode: geocode the result to GeoTIFF",
            ["dispmap.ROI.tif"],
        ):
            return

        utm_width = read_par_value(self.path("ref.utm.dem.par"), "width")
        utm_lines = read_par_value(self.path("ref.utm.dem.par"), "nlines")
        if not utm_width or not utm_lines:
            if self.args.dry_run:
                utm_width, utm_lines = "UTM_WIDTH", "UTM_LINES"
            else:
                die("cannot read width / nlines from ref.utm.dem.par.")
        print("    UTM DEM size: width={} nlines={}".format(utm_width, utm_lines))

        # vertical displacement = 1, line of sight (LOS) = 0
        displacement_flag = 1 if self.args.displacement == "vertical" else 0

        self.run(
            [
                "dispmap", "{}.diff.final".format(self.pair), "ref.hgt_sim",
                "{}.mli.par".format(self.master), "{}.off".format(self.pair),
                "{}.disp".format(self.pair), displacement_flag,
            ]
        )
        self.run(
            [
                "rashgt", "{}.disp".format(self.pair),
                "{}.mli".format(self.master), self.width,
                1, 1, 0, 1, 1, 0.04, 10.0, 0.35, 1,
            ]
        )
        self.run(
            [
                "geocode_back", "{}.disp".format(self.pair), self.width,
                "ref.lt_fine", "dispmap.ROI", utm_width, utm_lines, 1, 0,
            ]
        )
        self.run(
            [
                "rasdt_cmap", "dispmap.ROI", "ref.mli.utm", utm_width,
                "-", "-", "-", "-", "-", "-0.08", "0.08", "-", "-", "-", "-",
                "dispmap.ROI.bmp",
            ]
        )
        self.run(["data2geotiff", "ref.utm.dem.par", "dispmap.ROI.bmp", 0, "dispmap.ROI.tif"])
        print("    output: {}".format(self.path("dispmap.ROI.tif")))

    # ---------------- driver ----------------

    def execute(self) -> int:
        print("=" * 72)
        print("FO-BEAM  differential interferometry pipeline")
        print("  master={}  slave={}  pair={}".format(self.master, self.slave, self.pair))
        print(
            "  rlks={} alks={} adf_epoch={} diff_engine={} displacement={}".format(
                self.args.rlks, self.args.alks, self.args.adf_epoch,
                self.args.diff_engine, self.args.displacement,
            )
        )
        print("  workdir={}".format(self.workdir))
        print("=" * 72)

        self.check_inputs()
        if not self.args.dry_run:
            self.workdir.mkdir(parents=True, exist_ok=True)

        self.step_prepare()
        self.step_offset()
        self.step_multilook()
        self.step_diff()
        self.step_adf()
        self.step_unwrap()
        self.step_detrend()
        self.step_atm()
        self.step_geocode()

        print("\n" + "=" * 72)
        if self.args.dry_run:
            print("dry-run finished: no command was executed.")
        else:
            print("pipeline finished. final product: {}".format(self.path("dispmap.ROI.tif")))
        print("=" * 72)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fobeam.py",
        description="FO-BEAM: differential interferometry / unwrapping / detrend / atmosphere / geocode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------")[-1],
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + VERSION)

    parser.add_argument("--master", required=True, help="master scene ID (e.g. 20230820)")
    parser.add_argument("--slave", required=True, help="slave scene ID (e.g. 20230913)")
    parser.add_argument("--slc-dir", default="slc", help="directory holding the SLC files (default slc)")
    parser.add_argument("--dem-dir", default="dem", help="directory holding the external DEM (default dem)")
    parser.add_argument("--workdir", default=".", help="working directory (default: current directory)")

    parser.add_argument("--rlks", type=int, default=2, help="range looks (default 2)")
    parser.add_argument("--alks", type=int, default=4, help="azimuth looks (default 4)")
    parser.add_argument("--rpos", type=int, default=3311, help="unwrapping seed, range coordinate")
    parser.add_argument("--azpos", type=int, default=3024, help="unwrapping seed, azimuth coordinate")
    parser.add_argument(
        "--cc-thres", type=float, default=0.3, help="coherence threshold (default 0.3)"
    )
    parser.add_argument(
        "--pwr-thres", type=float, default=0.25, help="intensity threshold (default 0.25)"
    )
    parser.add_argument(
        "--adf-cc", type=float, default=0.25,
        help="trailing coherence threshold for adf (default 0.25)",
    )
    parser.add_argument(
        "--adf-epoch", type=int, choices=(1, 2, 3), default=3,
        help="adaptive filtering stages, 1-3 (default 3)",
    )
    parser.add_argument(
        "--diff-engine", choices=("orb", "fft"), default="orb",
        help="differential mode: orb=orbit-based simulation (default), fft=spectral",
    )
    parser.add_argument(
        "--displacement", choices=("los", "vertical"), default="los",
        help="displacement direction: los=line of sight (default), vertical",
    )
    parser.add_argument("--skip-atm", action="store_true", help="skip the atmosphere phase estimate")
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="skip finished steps (default on)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false", help="disable resuming"
    )
    parser.add_argument("--force", action="store_true", help="rerun every step")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if os.name == "nt":
        print("note: FO-BEAM needs external interferometry commands and usually runs on Linux only.")
    return InterferogramPipeline(args).execute()


if __name__ == "__main__":
    sys.exit(main())

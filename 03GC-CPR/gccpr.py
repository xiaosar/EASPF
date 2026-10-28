#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GC-CPR: GPU cross-correlation registration and ROI quality statistics

Subcommands
------
register   two-pass iterative offset tracking registration
           Pass 1: NCC + spectral zero-padding oversampling + parabolic
                   sub-pixel fit; yields the offsets and correlation field
           Pass 2: integer-offset sampling from Pass 1, residual-only
                   search, then a Matrix-DFT zoom to sub-pixel
           Closing: bivariate least squares + 2.5-sigma iterative rejection
stat-tif   binned histogram of values/phase over a GeoTIFF ROI
stat-fft   power-spectrum metrics of a complex GeoTIFF ROI (band ratio, entropy)

Main classes
------
SarIO               .par parsing, big-endian binary IO, amplitude, multi-look, dump
SARSignalProcessor  Lanczos/spline interpolation, spectral shift, rectangular
                    low-pass, FFT oversampling, cosine window, Jacobi infill
MatchEngine         phase-correlation coarse match, NCC zoom sub-pixel, fitting
SARWorkflow         pipeline wrappers for Pass 1, Pass 2 and fit cleaning

Dependencies
----
python >= 3.8, numpy, scipy
register needs cupy and a usable GPU
stat-tif / stat-fft need rasterio + matplotlib

Examples
----
python gccpr.py register --m-par m.rslc.par --m-bin m.rslc \
    --s-par s.rslc.par --s-bin s.rslc --step-y 12 --step-x 6
python gccpr.py register --m-par m.rslc.par --m-bin m.rslc \
    --s-par s.rslc.par --s-bin s.rslc --skip-pass2
python gccpr.py stat-tif diff.tif --row-start 0 --row-end 512 \
    --col-start 0 --col-end 512 --min-val -3 --max-val 3 --step 0.25
python gccpr.py stat-fft diff.tif --row-start 0 --row-end 512 \
    --col-start 0 --col-end 512
"""


from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

VERSION = "1.0.0"

# cupy is imported lazily so --help and the stat-* subcommands work without a GPU.
cp = None


def die(message: str, code: int = 2) -> "None":
    sys.stderr.write("Error: {}\n".format(message))
    raise SystemExit(code)


def require_cupy():
    """Import cupy on first use and bind it to the module-level `cp`."""
    global cp
    if cp is None:
        try:
            import cupy as _cp
        except ImportError:
            die(
                "missing dependency cupy (register needs a working CUDA setup).\n"
                "Install it on a CUDA-capable machine: pip install cupy-cuda12x"
            )
        cp = _cp
    return cp


def scipy_ndimage():
    try:
        from scipy import ndimage
    except ImportError:
        die("missing dependency scipy. Install it first:  pip install scipy")
    return ndimage


# ===========================================================================
# 1. I/O and basic signal conversion
# ===========================================================================
class SarIO:
    """Production-grade I/O and basic signal conversion.

    Design goals: big-endian compatibility, safe memory mapping, OFF files.
    """

    @staticmethod
    def read_par(par_file, key: str = None, dtype: type = str):
        """Parse a .par file, stripping units and converting types."""
        params = {}
        with open(str(par_file), "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith(("title", "#", "sensor")):
                    continue
                if ":" not in line:
                    continue
                field_key, value = line.split(":", 1)
                fields = value.strip().split()
                params[field_key.strip()] = fields[0] if fields else ""

        if key:
            if key not in params:
                raise KeyError("key not found in the PAR file: {}".format(key))
            return dtype(params[key])
        return params

    @staticmethod
    def read_binary(filename, shape, dtype_str: str = "fcomplex"):
        """Read binary data with memmap; fcomplex / float / int / scomplex."""
        type_map = {
            "fcomplex": np.dtype(">c8"),   # 32bit Re + 32bit Im
            "float": np.dtype(">f4"),      # 32bit Float
            "int": np.dtype(">i2"),        # 16bit Short
        }

        if dtype_str == "scomplex":
            # SCOMPLEX: 16-bit integer real part + 16-bit integer imaginary part
            raw = np.memmap(filename, dtype=">i2", mode="r")
            real = raw[0::2].reshape(shape).astype(np.float32)
            imag = raw[1::2].reshape(shape).astype(np.float32)
            return real + 1j * imag

        if dtype_str not in type_map:
            raise ValueError("unsupported dtype: {}".format(dtype_str))

        return np.memmap(filename, dtype=type_map[dtype_str], mode="r").reshape(shape)

    @staticmethod
    def get_amplitude(data):
        """Amplitude: A = sqrt(Re^2 + Im^2)."""
        xp = cp.get_array_module(data)
        if xp.iscomplexobj(data):
            return xp.abs(data)
        return xp.sqrt(xp.maximum(data, 0))

    @staticmethod
    def to_db(data, fill_min: float = -30.0):
        """Visualisation aid: decibels, dB = 10 * log10(Intensity)."""
        xp = cp.get_array_module(data)
        pwr = data if not xp.iscomplexobj(data) else data.real ** 2 + data.imag ** 2
        db = 10.0 * xp.log10(xp.maximum(pwr, 1e-12))
        return xp.maximum(db, fill_min) if fill_min is not None else db

    @staticmethod
    def multilook_cpu(data, rlks: int = 1, azlks: int = 1):
        """CPU multi-look downsampling by block averaging with reshape."""
        if rlks == 1 and azlks == 1:
            return data
        height, width = data.shape
        out_h, out_w = height // azlks, width // rlks
        return data[: out_h * azlks, : out_w * rlks].reshape(
            out_h, azlks, out_w, rlks
        ).mean(axis=(1, 3))

    @staticmethod
    def save_with_par(data, filename):
        """Save as big-endian binary and write the matching .par.

        Mapping: Real -> range offset (dx), Imag -> azimuth offset (dy)
        """
        if hasattr(data, "get"):
            data = data.get()

        if np.iscomplexobj(data):
            dtype_save = np.dtype(">c8")
            img_fmt = "FCOMPLEX"
        else:
            dtype_save = np.dtype(">f4")
            img_fmt = "FLOAT"

        data.astype(dtype_save).tofile(filename)

        with open(str(filename) + ".par", "w", encoding="utf-8") as handle:
            handle.write("azimuth_lines: {}\n".format(data.shape[0]))
            handle.write("range_samples: {}\n".format(data.shape[1]))
            handle.write("image_format:  {}\n".format(img_fmt))
            handle.write("byte_order:    BIG_ENDIAN\n")
        print("  [OK] big-endian file and header written: {}".format(filename))


# ===========================================================================
# 2. SAR signal processing helpers
# ===========================================================================
class SARSignalProcessor:
    """High-order interpolation, spectral shift, rectangular low-pass, FFT oversampling."""

    @staticmethod
    def lanczos_interpolate(data, n_ovr: int, order: int = 5):
        """Lanczos interpolation (separable 1D); order 5-9 is recommended."""
        a = int(np.clip(order, 5, 9))

        def lanczos_kernel(x, a):
            x = cp.abs(x)
            res = cp.zeros_like(x)
            mask = (x < a) & (x > 1e-9)
            pi_x = cp.pi * x[mask]
            res[mask] = (cp.sin(pi_x) / pi_x) * (cp.sin(pi_x / a) / (pi_x / a))
            res[x <= 1e-9] = 1.0
            return res

        def interp_1d(arr, ovr, a):
            rows, cols = arr.shape
            new_cols = int(cols * ovr)
            x_out = cp.arange(new_cols) / ovr
            offsets = cp.arange(-a + 1, a + 1)
            x_in_idx = cp.floor(x_out).astype(cp.int32)
            sample_idx = x_in_idx[:, None] + offsets[None, :]

            weights = lanczos_kernel(x_out[:, None] - sample_idx, a)
            weights /= cp.sum(weights, axis=1, keepdims=True)

            arr_pad = cp.pad(arr, ((0, 0), (a, a)), mode="edge")
            return cp.sum(arr_pad[:, sample_idx + a] * weights[None, :, :], axis=2)

        tmp = interp_1d(data, n_ovr, a)
        return interp_1d(tmp.T, n_ovr, a).T

    @staticmethod
    def spline_interpolate(data, n_ovr: int, order: int = 5):
        """B-spline interpolation (GPU accelerated)."""
        import cupyx.scipy.ndimage as cnd

        height, width = data.shape
        gy, gx = cp.mgrid[
            0 : height - 1 : complex(0, height * n_ovr),
            0 : width - 1 : complex(0, width * n_ovr),
        ]
        return cnd.map_coordinates(data, [gy, gx], order=min(order, 5), mode="edge")

    @staticmethod
    def phase_ramp_shift(patch_fft, dy: float, dx: float):
        """Spectral phase-ramp shift.

        Shift theorem: f(t-d) <-> F(w) * exp(-2j*pi*f*d)
        """
        ny, nx = patch_fft.shape
        fy = cp.fft.fftfreq(ny)
        fx = cp.fft.fftfreq(nx)
        fy_grid, fx_grid = cp.meshgrid(fy, fx, indexing="ij")
        ramp = cp.exp(-2j * cp.pi * (fy_grid * dy + fx_grid * dx))
        return patch_fft * ramp

    @staticmethod
    def lowpass_filter(patch_fft, bw_frac: float = 0.8):
        """Rectangular low-pass (separable mask); bw_frac is the bandwidth fraction."""
        if bw_frac >= 1.0:
            return patch_fft

        ny, nx = patch_fft.shape
        fy = cp.abs(cp.fft.fftfreq(ny))
        fx = cp.abs(cp.fft.fftfreq(nx))

        mask_y = fy <= (0.5 * bw_frac)
        mask_x = fx <= (0.5 * bw_frac)
        mask_2d = cp.outer(mask_y, mask_x)
        return patch_fft * mask_2d

    @staticmethod
    def fft_oversample(patch, n_ovr: int):
        """Oversampling by spectral zero-padding (removes aliasing)."""
        if n_ovr <= 1:
            return patch

        ny, nx = patch.shape
        spectrum = cp.fft.fft2(patch)
        shifted = cp.fft.fftshift(spectrum)

        new_ny, new_nx = ny * n_ovr, nx * n_ovr
        pad_y = (new_ny - ny) // 2
        pad_x = (new_nx - nx) // 2

        padded = cp.pad(
            shifted, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant"
        )
        return cp.fft.ifft2(cp.fft.ifftshift(padded)) * (n_ovr ** 2)

    @staticmethod
    def apply_cosine_window(patch, taper_frac: float = 0.125):
        """Edge cosine window: removes the shift bias caused by patch truncation."""
        ny, nx = patch.shape

        def win1d(length, frac):
            window = cp.ones(length)
            margin = int(length * frac)
            if margin > 0:
                ramp = 0.5 * (1 - cp.cos(cp.pi * cp.arange(margin) / margin))
                window[:margin] = ramp
                window[-margin:] = ramp[::-1]
            return window

        return patch * cp.outer(win1d(ny, taper_frac), win1d(nx, taper_frac))

    @staticmethod
    def refine_offset_map(res_dict, k_std: float = 3.0, filter_radius: int = 2):
        """Offset refinement: statistical mask -> elastic infill -> uniform mean filter."""
        dy = res_dict["dy"].copy()
        dx = res_dict["dx"].copy()
        mask = res_dict["mask"].copy()

        for data in (dy, dx):
            valid_index = ~np.isnan(data) & mask
            if np.any(valid_index):
                mean_value = np.mean(data[valid_index])
                std_value = np.std(data[valid_index])
                outlier = np.abs(data - mean_value) > k_std * std_value
                data[outlier] = np.nan
                mask[outlier] = False

        print("  [Action] pyramid elastic infill (spring analogy) ...")
        dy_filled = SARSignalProcessor._spring_interp_solve(dy, mask, ds_factor=4)
        dx_filled = SARSignalProcessor._spring_interp_solve(dx, mask, ds_factor=4)

        print("  [Action] uniform smoothing (radius={}) ...".format(filter_radius))
        win_size = 2 * filter_radius + 1
        ndimage = scipy_ndimage()
        dy_refined = ndimage.uniform_filter(dy_filled, size=win_size, mode="reflect")
        dx_refined = ndimage.uniform_filter(dx_filled, size=win_size, mode="reflect")
        return dy_refined, dx_refined

    @staticmethod
    def _spring_interp_solve(data, mask, ds_factor: int = 4):
        """Multi-scale spring-analogy solver (coarse diffusion + fine refinement)."""
        ndimage = scipy_ndimage()
        ny, nx = data.shape
        hole_mask = ~mask | np.isnan(data)
        if not np.any(hole_mask):
            return data

        ny_ds, nx_ds = ny // ds_factor, nx // ds_factor
        if ny_ds > 10 and nx_ds > 10:
            data_ds = ndimage.zoom(
                np.nan_to_num(data, nan=np.nanmean(data)), 1 / ds_factor, order=1
            )
            mask_ds = ndimage.zoom(mask.astype(float), 1 / ds_factor, order=0) > 0.5

            data_ds_filled = SARSignalProcessor._jacobi_core(
                data_ds, mask_ds, max_iter=max(ny_ds, nx_ds)
            )
            upsampled = ndimage.zoom(
                data_ds_filled, (ny / ny_ds, nx / nx_ds), order=1
            )
            out = upsampled[:ny, :nx]
        else:
            out = data.copy()
            out[hole_mask] = np.nanmean(data)

        out[~hole_mask] = data[~hole_mask]
        return SARSignalProcessor._jacobi_core(out, mask, max_iter=50)

    @staticmethod
    def _jacobi_core(data, mask, max_iter: int):
        """Elastic infill core: Jacobi iteration, phi = (L+R+U+D)/4."""
        out = data.copy()
        hole_mask = ~mask
        for iteration in range(max_iter):
            prev = out.copy()
            out[1:-1, 1:-1] = 0.25 * (
                out[:-2, 1:-1] + out[2:, 1:-1] + out[1:-1, :-2] + out[1:-1, 2:]
            )
            # Neumann boundary condition
            out[0, :] = out[1, :]
            out[-1, :] = out[-2, :]
            out[:, 0] = out[:, 1]
            out[:, -1] = out[:, -2]
            # keep observed samples fixed
            out[~hole_mask] = data[~hole_mask]
            if iteration % 20 == 0:
                if np.max(np.abs(out - prev)) < 1e-4:
                    break
        return out


# ===========================================================================
# 3. Match engine
# ===========================================================================
class MatchEngine:
    """Phase-correlation coarse match, NCC zoom sub-pixel, polynomial fitting."""

    @staticmethod
    def calculate_coarse_offset(m_data, s_data, patch_size: int = 512,
                                bw_frac: float = 0.8):
        """Phase correlation with cosine windowing and low-pass (coarse match)."""
        m_mask = m_data > 1e-6
        rows_any = cp.any(m_mask, axis=1)
        cols_any = cp.any(m_mask, axis=0)
        if not cp.any(rows_any) or not cp.any(cols_any):
            return (0.0, 0.0), 0.0

        rmin = int(cp.where(rows_any)[0][0])
        rmax = int(cp.where(rows_any)[0][-1])
        cmin = int(cp.where(cols_any)[0][0])
        cmax = int(cp.where(cols_any)[0][-1])

        psize = min(patch_size, rmax - rmin, cmax - cmin)
        tm_y = rmin + (rmax - rmin) // 2 - psize // 2
        tm_x = cmin + (cmax - cmin) // 2 - psize // 2
        m_patch = m_data[tm_y : tm_y + psize, tm_x : tm_x + psize]

        m_patch_win = SARSignalProcessor.apply_cosine_window(m_patch, taper_frac=0.125)
        m_norm = (m_patch_win - cp.mean(m_patch_win)) / (cp.std(m_patch_win) + 1e-12)

        s_win = SARSignalProcessor.apply_cosine_window(s_data, taper_frac=0.05)
        s_norm = (s_win - cp.mean(s_win)) / (cp.std(s_win) + 1e-12)

        n_fft_y, n_fft_x = 2048, 2048
        f1 = cp.fft.fft2(m_norm, s=(n_fft_y, n_fft_x))
        f2 = cp.fft.fft2(s_norm, s=(n_fft_y, n_fft_x))
        f1 = SARSignalProcessor.lowpass_filter(f1, bw_frac=bw_frac)
        f2 = SARSignalProcessor.lowpass_filter(f2, bw_frac=bw_frac)

        cross_power = (f2 * f1.conj()) / (cp.abs(f2 * f1.conj()) + 1e-15)
        r_map = cp.abs(cp.fft.ifft2(cross_power))

        idx = cp.argmax(r_map)
        y_peak = int(idx // n_fft_x)
        x_peak = int(idx % n_fft_x)
        dy_rel = y_peak if y_peak < n_fft_y // 2 else y_peak - n_fft_y
        dx_rel = x_peak if x_peak < n_fft_x // 2 else x_peak - n_fft_x

        off_y, off_x = MatchEngine._refine_quadratic_periodic(
            r_map, y_peak, x_peak, n_fft_y, n_fft_x
        )
        dy_res = dy_rel + off_y - tm_y
        dx_res = dx_rel + off_x - tm_x

        return (float(dy_res), float(dx_res)), float(r_map.max())

    @staticmethod
    def _refine_quadratic_periodic(r_map, y_peak, x_peak, ny, nx):
        """Neighbourhood quadratic fit with periodic boundaries."""
        y_idx = [(y_peak + offset) % ny for offset in (-1, 0, 1)]
        x_idx = [(x_peak + offset) % nx for offset in (-1, 0, 1)]
        z = r_map[cp.ix_(y_idx, x_idx)].get()

        dy_sub, dx_sub = 0.0, 0.0
        denom_y = 2 * (z[0, 1] - 2 * z[1, 1] + z[2, 1])
        if abs(denom_y) > 1e-12:
            dy_sub = -(z[2, 1] - z[0, 1]) / denom_y

        denom_x = 2 * (z[1, 0] - 2 * z[1, 1] + z[1, 2])
        if abs(denom_x) > 1e-12:
            dx_sub = -(z[1, 2] - z[1, 0]) / denom_x

        return dy_sub, dx_sub

    @staticmethod
    def compute_fine_offset(m_patch, s_patch, bw_frac: float = 0.9):
        """Single-window fine match: cosine window + cross-spectrum low-pass + sub-pixel."""
        m_win = SARSignalProcessor.apply_cosine_window(m_patch, taper_frac=0.1)
        s_win = SARSignalProcessor.apply_cosine_window(s_patch, taper_frac=0.1)

        m_norm = (m_win - cp.mean(m_win)) / (cp.std(m_win) + 1e-12)
        s_norm = (s_win - cp.mean(s_win)) / (cp.std(s_win) + 1e-12)

        f_m = cp.fft.fft2(m_norm)
        f_s = cp.fft.fft2(s_norm)

        cross_power = (f_s * f_m.conj()) / (cp.abs(f_s * f_m.conj()) + 1e-15)
        cross_filtered = SARSignalProcessor.lowpass_filter(cross_power, bw_frac=bw_frac)
        r_map = cp.abs(cp.fft.ifft2(cross_filtered))

        ny, nx = r_map.shape
        idx = cp.argmax(r_map)
        y_peak = int(idx // nx)
        x_peak = int(idx % nx)

        dy_peak = y_peak if y_peak < ny // 2 else y_peak - ny
        dx_peak = x_peak if x_peak < nx // 2 else x_peak - nx

        off_y, off_x = MatchEngine._refine_quadratic_periodic(
            r_map, y_peak, x_peak, ny, nx
        )
        return (float(dy_peak + off_y), float(dx_peak + off_x)), float(r_map.max())

    @staticmethod
    def compute_fine_offset_batch(m_batch, s_batch, n_ovr: int = 2,
                                  bw_frac: float = 1.0):
        """[used by Pass 1] batched NCC fine match + spectral zero-padding oversampling.

        Logic:
        - normalised cross-correlation (NCC) for robust intensity matching;
        - spectral zero-padding for oversampling;
        - ccs is the standard correlation coefficient (0.0 - 1.0).
        """
        batch, height, width = m_batch.shape
        num_pix = height * width

        m_mean = cp.mean(m_batch, axis=(1, 2), keepdims=True)
        s_mean = cp.mean(s_batch, axis=(1, 2), keepdims=True)
        m_std = cp.std(m_batch, axis=(1, 2), keepdims=True) + 1e-12
        s_std = cp.std(s_batch, axis=(1, 2), keepdims=True) + 1e-12

        m_norm = (m_batch - m_mean) / m_std
        s_norm = (s_batch - s_mean) / s_std

        f1 = cp.fft.fft2(m_norm)
        f2 = cp.fft.fft2(s_norm)
        cross_spec = f2 * f1.conj()

        # spectral low-pass (suppress speckle noise, limit the effective bandwidth)
        cy, cx = height // 2, width // 2
        y_g, x_g = cp.ogrid[:height, :width]
        dist = cp.sqrt(
            ((y_g - cy) / (height / 2)) ** 2 + ((x_g - cx) / (width / 2)) ** 2
        )
        lowpass = cp.where(dist <= bw_frac, 1.0, 0.0)
        cross_spec = cross_spec * cp.fft.fftshift(lowpass)[None, :, :]

        if n_ovr > 1:
            new_h, new_w = height * n_ovr, width * n_ovr
            shifted = cp.fft.fftshift(cross_spec, axes=(1, 2))
            padded = cp.zeros((batch, new_h, new_w), dtype=cross_spec.dtype)
            start_y, start_x = (new_h - height) // 2, (new_w - width) // 2
            padded[:, start_y : start_y + height, start_x : start_x + width] = shifted
            r_map = cp.real(cp.fft.ifft2(cp.fft.ifftshift(padded, axes=(1, 2))))
            cur_h, cur_w = new_h, new_w
        else:
            r_map = cp.real(cp.fft.ifft2(cross_spec))
            cur_h, cur_w = height, width

        r_map /= num_pix

        r_flat = r_map.reshape(batch, -1)
        peak_idx = cp.argmax(r_flat, axis=1)
        ccs = cp.take_along_axis(r_flat, peak_idx[:, None], axis=1).squeeze()

        y_p, x_p = peak_idx // cur_w, peak_idx % cur_w
        dy_p = cp.where(y_p < cur_h // 2, y_p, y_p - cur_h) / n_ovr
        dx_p = cp.where(x_p < cur_w // 2, x_p, x_p - cur_w) / n_ovr

        batch_idx = cp.arange(batch)

        def get_val(offset_y, offset_x):
            return r_map[
                batch_idx, (y_p + offset_y) % cur_h, (x_p + offset_x) % cur_w
            ]

        z11 = r_map[batch_idx, y_p, x_p]
        z01, z21 = get_val(-1, 0), get_val(1, 0)
        z10, z12 = get_val(0, -1), get_val(0, 1)

        denom_y = z01 - 2 * z11 + z21
        denom_x = z10 - 2 * z11 + z12

        dy_sub = cp.where(
            cp.abs(denom_y) > 1e-12, -(z21 - z01) / (2 * denom_y), 0
        ) / n_ovr
        dx_sub = cp.where(
            cp.abs(denom_x) > 1e-12, -(z12 - z10) / (2 * denom_x), 0
        ) / n_ovr

        return (
            (dy_p + dy_sub).astype(cp.float32),
            (dx_p + dx_sub).astype(cp.float32),
            ccs.astype(cp.float32),
        )

    @staticmethod
    def compute_fine_offset_batch2(m_batch, s_batch, refine_factor: int = 16,
                                   refine_win: float = 1.5):
        """[alternative] zoom NCC: integer coarse search + Matrix-DFT sub-pixel.

        Unlike compute_fine_offset_batch this uses a local Matrix-DFT zoom instead
        of zero-padding; kept for comparisons (not used by register by default).
        """
        batch, height, width = m_batch.shape
        num_pix = height * width

        m_mean = cp.mean(m_batch, axis=(1, 2), keepdims=True)
        s_mean = cp.mean(s_batch, axis=(1, 2), keepdims=True)
        m_std = cp.std(m_batch, axis=(1, 2), keepdims=True) + 1e-12
        s_std = cp.std(s_batch, axis=(1, 2), keepdims=True) + 1e-12

        m_norm = (m_batch - m_mean) / m_std
        s_norm = (s_batch - s_mean) / s_std

        f1 = cp.fft.fft2(m_norm)
        f2 = cp.fft.fft2(s_norm)
        cross_spec = f2 * f1.conj()

        r_map_coarse = cp.real(cp.fft.ifft2(cross_spec)) / num_pix
        r_flat = r_map_coarse.reshape(batch, -1)
        peak_idx = cp.argmax(r_flat, axis=1)

        y_p_idx = peak_idx // width
        x_p_idx = peak_idx % width
        dy_coarse = cp.where(y_p_idx < height // 2, y_p_idx, y_p_idx - height).astype(
            cp.float32
        )
        dx_coarse = cp.where(x_p_idx < width // 2, x_p_idx, x_p_idx - width).astype(
            cp.float32
        )

        if refine_factor <= 1 or refine_win <= 0:
            ccs_coarse = cp.take_along_axis(r_flat, peak_idx[:, None], axis=1).squeeze()
            return dy_coarse, dx_coarse, cp.clip(ccs_coarse, 0.0, 1.0)

        refine_sz = int(refine_win * 2 * refine_factor)
        fy = cp.fft.fftfreq(height)
        fx = cp.fft.fftfreq(width)
        sub_offset = cp.linspace(-refine_win, refine_win, refine_sz)

        phase_y = cp.exp(
            2j * cp.pi * fy[None, None, :]
            * (dy_coarse[:, None, None] + sub_offset[None, :, None])
        )
        phase_x = cp.exp(
            2j * cp.pi * fx[None, :, None]
            * (dx_coarse[:, None, None] + sub_offset[None, None, :])
        )
        refined_map = cp.real(cp.matmul(cp.matmul(phase_y, cross_spec), phase_x))

        refined_flat = refined_map.reshape(batch, -1)
        sub_peak_idx = cp.argmax(refined_flat, axis=1)
        ccs_fine = cp.take_along_axis(
            refined_flat, sub_peak_idx[:, None], axis=1
        ).squeeze() / (num_pix ** 2)

        sub_y = sub_offset[sub_peak_idx // refine_sz]
        sub_x = sub_offset[sub_peak_idx % refine_sz]

        return (
            (dy_coarse + sub_y).astype(cp.float32),
            (dx_coarse + sub_x).astype(cp.float32),
            cp.clip(ccs_fine, 0.0, 1.0),
        )

    @staticmethod
    def compute_fine_offset2_batch(m_batch, s_batch, dy_base, dx_base,
                                   n_ovr: int = 2, bw_frac: float = 1.0):
        """[alternative] NCC residual search with an initial offset (zero-padding).

        dy_base/dx_base are the integer pixel shifts applied to the slave window.
        """
        batch, height, width = m_batch.shape
        num_pix = height * width

        dy_b = cp.asarray(dy_base).ravel()
        dx_b = cp.asarray(dx_base).ravel()

        m_norm = (m_batch - cp.mean(m_batch, axis=(1, 2), keepdims=True)) / (
            cp.std(m_batch, axis=(1, 2), keepdims=True) + 1e-12
        )
        s_norm = (s_batch - cp.mean(s_batch, axis=(1, 2), keepdims=True)) / (
            cp.std(s_batch, axis=(1, 2), keepdims=True) + 1e-12
        )

        cross_spec = cp.fft.fft2(s_norm) * cp.fft.fft2(m_norm).conj()

        cy, cx = height // 2, width // 2
        y_g, x_g = cp.ogrid[:height, :width]
        dist = cp.sqrt(
            ((y_g - cy) / (height / 2)) ** 2 + ((x_g - cx) / (width / 2)) ** 2
        )
        lowpass = cp.fft.fftshift(cp.where(dist <= bw_frac, 1.0, 0.0))[None, :, :]
        cross_spec = cross_spec * lowpass

        if n_ovr > 1:
            new_h, new_w = height * n_ovr, width * n_ovr
            shifted = cp.fft.fftshift(cross_spec, axes=(1, 2))
            padded = cp.zeros((batch, new_h, new_w), dtype=cross_spec.dtype)
            start_y, start_x = (new_h - height) // 2, (new_w - width) // 2
            padded[:, start_y : start_y + height, start_x : start_x + width] = shifted
            r_map = cp.real(cp.fft.ifft2(cp.fft.ifftshift(padded, axes=(1, 2))))
            cur_h, cur_w = new_h, new_w
        else:
            r_map = cp.real(cp.fft.ifft2(cross_spec))
            cur_h, cur_w = height, width

        r_map /= num_pix

        r_flat = r_map.reshape(batch, -1)
        peak_idx = cp.argmax(r_flat, axis=1)
        ccs = cp.take_along_axis(r_flat, peak_idx[:, None], axis=1).squeeze()

        y_p, x_p = peak_idx // cur_w, peak_idx % cur_w
        dy_delta = cp.where(y_p < cur_h // 2, y_p, y_p - cur_h).astype(
            cp.float32
        ) / n_ovr
        dx_delta = cp.where(x_p < cur_w // 2, x_p, x_p - cur_w).astype(
            cp.float32
        ) / n_ovr

        batch_idx = cp.arange(batch)
        z11 = r_map[batch_idx, y_p, x_p]
        z01 = r_map[batch_idx, (y_p - 1) % cur_h, x_p]
        z21 = r_map[batch_idx, (y_p + 1) % cur_h, x_p]
        z10 = r_map[batch_idx, y_p, (x_p - 1) % cur_w]
        z12 = r_map[batch_idx, y_p, (x_p + 1) % cur_w]

        dy_sub = cp.where(
            cp.abs(z01 - 2 * z11 + z21) > 1e-12,
            -(z21 - z01) / (2 * (z01 - 2 * z11 + z21)),
            0,
        ) / n_ovr
        dx_sub = cp.where(
            cp.abs(z10 - 2 * z11 + z12) > 1e-12,
            -(z12 - z10) / (2 * (z10 - 2 * z11 + z12)),
            0,
        ) / n_ovr

        return (
            (dy_b + dy_delta + dy_sub).astype(cp.float32),
            (dx_b + dx_delta + dx_sub).astype(cp.float32),
            cp.clip(ccs, 0.0, 1.0),
        )

    @staticmethod
    def compute_fine_offset2_batch2(m_batch, s_batch, dy_base, dx_base,
                                    refine_factor: int = 16,
                                    refine_win: float = 1.5,
                                    bw_frac: float = 1.0):
        """[used by Pass 2] zoom NCC iteration: integer offset + residual + sub-pixel.

        dy_base/dx_base are the integer pixel shifts already applied to the slave.
        Total shift = integer offset + correlation residual + Matrix-DFT sub-pixel.
        """
        batch, height, width = m_batch.shape
        num_pix = height * width

        dy_b = cp.asarray(dy_base).ravel()
        dx_b = cp.asarray(dx_base).ravel()

        m_norm = (m_batch - cp.mean(m_batch, axis=(1, 2), keepdims=True)) / (
            cp.std(m_batch, axis=(1, 2), keepdims=True) + 1e-12
        )
        s_norm = (s_batch - cp.mean(s_batch, axis=(1, 2), keepdims=True)) / (
            cp.std(s_batch, axis=(1, 2), keepdims=True) + 1e-12
        )

        f1 = cp.fft.fft2(m_norm)
        f2 = cp.fft.fft2(s_norm)
        cross_spec = f2 * f1.conj()

        if bw_frac < 1.0:
            cy, cx = height // 2, width // 2
            y_g, x_g = cp.ogrid[:height, :width]
            dist = cp.sqrt(
                ((y_g - cy) / (height / 2)) ** 2 + ((x_g - cx) / (width / 2)) ** 2
            )
            lowpass = cp.fft.fftshift(cp.where(dist <= bw_frac, 1.0, 0.0))[None, :, :]
            cross_spec = cross_spec * lowpass

        r_map_coarse = cp.real(cp.fft.ifft2(cross_spec)) / num_pix
        r_flat = r_map_coarse.reshape(batch, -1)
        peak_idx = cp.argmax(r_flat, axis=1)

        y_p = peak_idx // width
        x_p = peak_idx % width
        dy_res = cp.where(y_p < height // 2, y_p, y_p - height).astype(cp.float32)
        dx_res = cp.where(x_p < width // 2, x_p, x_p - width).astype(cp.float32)

        if refine_factor <= 1:
            ccs_coarse = cp.take_along_axis(r_flat, peak_idx[:, None], axis=1).squeeze()
            return dy_b + dy_res, dx_b + dx_res, cp.clip(ccs_coarse, 0.0, 1.0)

        refine_sz = int(refine_win * 2 * refine_factor)
        fy = cp.fft.fftfreq(height)
        fx = cp.fft.fftfreq(width)
        sub_offset = cp.linspace(-refine_win, refine_win, refine_sz)

        phase_y = cp.exp(
            2j * cp.pi * fy[None, None, :]
            * (dy_res[:, None, None] + sub_offset[None, :, None])
        )
        phase_x = cp.exp(
            2j * cp.pi * fx[None, :, None]
            * (dx_res[:, None, None] + sub_offset[None, None, :])
        )
        refined_map = cp.real(cp.matmul(cp.matmul(phase_y, cross_spec), phase_x))

        refined_flat = refined_map.reshape(batch, -1)
        sub_peak_idx = cp.argmax(refined_flat, axis=1)

        sub_y = sub_offset[sub_peak_idx // refine_sz]
        sub_x = sub_offset[sub_peak_idx % refine_sz]

        ccs_fine = cp.take_along_axis(
            refined_flat, sub_peak_idx[:, None], axis=1
        ).squeeze() / (num_pix ** 2)

        return (
            (dy_b + dy_res + sub_y).astype(cp.float32),
            (dx_b + dx_res + sub_x).astype(cp.float32),
            cp.clip(ccs_fine, 0.0, 1.0),
        )

    @staticmethod
    def calculate_grid_params(dim, step):
        """Grid planning: samples = ceil(dim/step) + 1."""
        n_samples = math.ceil(dim / step) + 1
        start = 0
        last_center = (n_samples - 1) * step
        end_limit = last_center + (step - 1)
        return n_samples, start, last_center, end_limit

    @staticmethod
    def get_safe_patch(img, y_center, x_center, rows, cols, win_h, win_w):
        """Centre-aligned sampling with automatic zero padding at the borders."""
        r_start, r_end = y_center - win_h // 2, y_center + win_h // 2
        c_start, c_end = x_center - win_w // 2, x_center + win_w // 2
        pad_top, pad_bottom = max(0, -r_start), max(0, r_end - rows)
        pad_left, pad_right = max(0, -c_start), max(0, c_end - cols)
        v_ys, v_ye = max(0, r_start), min(rows, r_end)
        v_xs, v_xe = max(0, c_start), min(cols, c_end)
        patch = img[v_ys:v_ye, v_xs:v_xe]
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            return np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)),
                          mode="constant")
        return patch

    @staticmethod
    def fit_offset_polynomial(y_coords, x_coords, dy_values, dx_values, n_poly: int = 6):
        """Bivariate polynomial fit.

        Parameter levels:
            1: A0
            3: A0 + A1*r + A2*az
            4: A0 + A1*r + A2*az + A3*r*az
            6: A0 + A1*r + A2*az + A3*r*az + A4*r^2 + A5*az^2
        """
        radius = x_coords
        azimuth = y_coords

        design_full = np.column_stack(
            [
                np.ones_like(radius),
                radius,
                azimuth,
                radius * azimuth,
                radius ** 2,
                azimuth ** 2,
            ]
        )

        if n_poly not in (1, 3, 4, 6):
            n_poly = 6 if n_poly > 6 else (4 if n_poly > 4 else (3 if n_poly > 3 else 1))

        design = design_full[:, :n_poly]

        coeffs_dy, _, _, _ = np.linalg.lstsq(design, dy_values, rcond=None)
        coeffs_dx, _, _, _ = np.linalg.lstsq(design, dx_values, rcond=None)

        res_dy = dy_values - (design @ coeffs_dy)
        res_dx = dx_values - (design @ coeffs_dx)

        return (
            (coeffs_dy, coeffs_dx),
            (res_dy, res_dx),
            (float(np.std(res_dy)), float(np.std(res_dx))),
        )


# ===========================================================================
# 4. Processing pipeline
# ===========================================================================
class SARWorkflow:
    """One-call wrappers from coarse to fine registration."""

    @staticmethod
    def run_coarse_registration(m_par_path, m_bin_path, s_par_path, s_bin_path,
                                target_size: int = 1024, bw_frac: float = 0.8):
        """Initial offset estimate: adaptive multi-look thumbnail + phase correlation.

        Returns (dy_fine, dx_fine, cc) with offsets scaled back to SLC pixels.
        """
        m_par = SarIO.read_par(m_par_path)
        s_par = SarIO.read_par(s_par_path)
        m_rows, m_cols = int(m_par["azimuth_lines"]), int(m_par["range_samples"])
        s_rows, s_cols = int(s_par["azimuth_lines"]), int(s_par["range_samples"])
        m_fmt = m_par.get("image_format", "fcomplex").lower()
        s_fmt = s_par.get("image_format", "fcomplex").lower()

        m_data = SarIO.read_binary(m_bin_path, (m_rows, m_cols), m_fmt)
        s_data = SarIO.read_binary(s_bin_path, (s_rows, s_cols), s_fmt)

        rlks = max(1, m_cols // target_size)
        azlks = max(1, m_rows // target_size)

        m_thumb = SarIO.multilook_cpu(SarIO.get_amplitude(m_data), rlks, azlks)
        s_thumb = SarIO.multilook_cpu(SarIO.get_amplitude(s_data), rlks, azlks)

        m_gpu = cp.asarray(m_thumb)
        s_gpu = cp.asarray(s_thumb)
        (dy_thumb, dx_thumb), cc = MatchEngine.calculate_coarse_offset(
            m_gpu, s_gpu, patch_size=512, bw_frac=bw_frac
        )

        return float(dy_thumb * azlks), float(dx_thumb * rlks), cc

    @staticmethod
    def run_offset_tracking(m_par, m_bin, s_par, s_bin,
                            step_y: int = 12, step_x: int = 6,
                            win_h: int = 128, win_w: int = 128,
                            cc_thresh: float = 0.05,
                            n_ovr: int = 2,
                            refine_factor: int = 16,
                            refine_win: float = 1.5,
                            batch_size: int = 1024,
                            out_prefix: str = "fine_offsets"):
        """Pass 1: batched GPU offset tracking (NCC with spectral zero-padding)."""
        start_time = time.time()
        cp.get_default_memory_pool().free_all_blocks()

        m_par_dict = SarIO.read_par(m_par)
        m_rows, m_cols = int(m_par_dict["azimuth_lines"]), int(
            m_par_dict["range_samples"]
        )
        ny, _, _, _ = MatchEngine.calculate_grid_params(m_rows, step_y)
        nx, _, _, _ = MatchEngine.calculate_grid_params(m_cols, step_x)

        print("=> [Step 1] loading images into device memory (grid {}x{}) ...".format(ny, nx))
        m_img_gpu = cp.asarray(
            SarIO.get_amplitude(
                SarIO.read_binary(m_bin, (m_rows, m_cols), "fcomplex")
            )
        )
        s_par_dict = SarIO.read_par(s_par)
        s_rows, s_cols = int(s_par_dict["azimuth_lines"]), int(
            s_par_dict["range_samples"]
        )
        s_img_gpu = cp.asarray(
            SarIO.get_amplitude(
                SarIO.read_binary(s_bin, (s_rows, s_cols), "fcomplex")
            )
        )

        pad_h, pad_w = win_h // 2, win_w // 2
        m_img_pad = cp.pad(m_img_gpu, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        s_img_pad = cp.pad(s_img_gpu, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        del m_img_gpu, s_img_gpu

        dy_map_gpu = cp.zeros((ny, nx), dtype=cp.float32)
        dx_map_gpu = cp.zeros((ny, nx), dtype=cp.float32)
        cc_map_gpu = cp.zeros((ny, nx), dtype=cp.float32)

        y_centers = cp.arange(ny) * step_y + pad_h
        x_centers = cp.arange(nx) * step_x + pad_w
        yy_c, xx_c = cp.meshgrid(y_centers, x_centers, indexing="ij")
        flat_yc, flat_xc = yy_c.ravel(), xx_c.ravel()
        win_y_idx, win_x_idx = cp.meshgrid(
            cp.arange(win_h) - pad_h, cp.arange(win_w) - pad_w, indexing="ij"
        )

        print("=> [Step 1] batched NCC search (n_ovr={}) ...".format(n_ovr))
        t_calc = time.time()
        total_pts = len(flat_yc)

        for b_start in range(0, total_pts, batch_size):
            b_end = min(b_start + batch_size, total_pts)

            idx_y = (flat_yc[b_start:b_end, None, None] + win_y_idx).astype(cp.int32)
            idx_x = (flat_xc[b_start:b_end, None, None] + win_x_idx).astype(cp.int32)

            m_batch = m_img_pad[idx_y, idx_x]
            s_batch = s_img_pad[idx_y, idx_x]

            dys, dxs, ccs = MatchEngine.compute_fine_offset_batch(
                m_batch, s_batch, n_ovr=n_ovr
            )

            dy_map_gpu.ravel()[b_start:b_end] = dys
            dx_map_gpu.ravel()[b_start:b_end] = dxs
            cc_map_gpu.ravel()[b_start:b_end] = ccs

            if b_end % (batch_size * 20) == 0 or b_end == total_pts:
                elapsed = time.time() - t_calc
                rate = b_end / elapsed if elapsed > 0 else 0
                print(
                    "   progress: {:5.1f}% | rate: {:7.0f} pts/s".format(
                        b_end / total_pts * 100, rate
                    )
                )

        mask_gpu = cc_map_gpu > cc_thresh
        dx_masked = cp.where(mask_gpu, dx_map_gpu, 0.0)
        dy_masked = cp.where(mask_gpu, dy_map_gpu, 0.0)
        off_masked = (dx_masked + 1j * dy_masked).astype(cp.complex64)

        SarIO.save_with_par(off_masked, "{}.offs".format(out_prefix))
        SarIO.save_with_par(dx_masked, "{}.offs.rg".format(out_prefix))
        SarIO.save_with_par(dy_masked, "{}.offs.az".format(out_prefix))
        SarIO.save_with_par(cc_map_gpu, "{}.cc".format(out_prefix))

        print(
            "=> [Step 1] done. valid points: {} | elapsed: {:.2f} s".format(
                int(cp.sum(mask_gpu)), time.time() - start_time
            )
        )

        return {
            "dy": dy_map_gpu.get(),
            "dx": dx_map_gpu.get(),
            "cc": cc_map_gpu.get(),
            "mask": mask_gpu.get(),
            "step_y": step_y,
            "step_x": step_x,
            "ny": ny,
            "nx": nx,
            "prefix": out_prefix,
        }

    @staticmethod
    def run_offset_tracking2(m_par, m_bin, s_par, s_bin,
                             step_y: int = 12, step_x: int = 6,
                             win_h: int = 128, win_w: int = 128,
                             cc_thresh: float = 0.05,
                             refine_factor: int = 16,
                             refine_win: float = 1.5,
                             batch_size: int = 1024,
                             out_prefix: str = "fine_offsets",
                             dy_init=None, dx_init=None):
        """Pass 2: iterative fine match (integer offset + residual search + zoom).

        Uses the real-valued Pass 1 offsets for integer-offset sampling, then searches the residual only.
        """
        start_time = time.time()
        cp.get_default_memory_pool().free_all_blocks()

        m_par_dict = SarIO.read_par(m_par)
        m_rows, m_cols = int(m_par_dict["azimuth_lines"]), int(
            m_par_dict["range_samples"]
        )
        ny, _, _, _ = MatchEngine.calculate_grid_params(m_rows, step_y)
        nx, _, _, _ = MatchEngine.calculate_grid_params(m_cols, step_x)

        if dy_init is None:
            flat_dy_init = cp.zeros(ny * nx, dtype=cp.float32)
        else:
            flat_dy_init = cp.asarray(dy_init).ravel()

        if dx_init is None:
            flat_dx_init = cp.zeros(ny * nx, dtype=cp.float32)
        else:
            flat_dx_init = cp.asarray(dx_init).ravel()

        print("=> [Action] loading images into device memory (grid {}x{}) ...".format(ny, nx))
        m_img_gpu = cp.asarray(
            SarIO.get_amplitude(
                SarIO.read_binary(m_bin, (m_rows, m_cols), "fcomplex")
            )
        )
        s_par_dict = SarIO.read_par(s_par)
        s_rows, s_cols = int(s_par_dict["azimuth_lines"]), int(
            s_par_dict["range_samples"]
        )
        s_img_gpu = cp.asarray(
            SarIO.get_amplitude(
                SarIO.read_binary(s_bin, (s_rows, s_cols), "fcomplex")
            )
        )

        pad_h, pad_w = win_h // 2, win_w // 2
        m_img_pad = cp.pad(m_img_gpu, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        s_img_pad = cp.pad(s_img_gpu, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        del m_img_gpu, s_img_gpu

        dy_map_gpu = cp.zeros((ny, nx), dtype=cp.float32)
        dx_map_gpu = cp.zeros((ny, nx), dtype=cp.float32)
        cc_map_gpu = cp.zeros((ny, nx), dtype=cp.float32)

        y_centers = cp.arange(ny) * step_y + pad_h
        x_centers = cp.arange(nx) * step_x + pad_w
        yy_c, xx_c = cp.meshgrid(y_centers, x_centers, indexing="ij")
        flat_yc, flat_xc = yy_c.ravel(), xx_c.ravel()

        win_y_idx, win_x_idx = cp.meshgrid(
            cp.arange(win_h) - pad_h, cp.arange(win_w) - pad_w, indexing="ij"
        )

        print(
            "=> [Action] Tracking2 iterative match (refine factor: {}) ...".format(
                refine_factor
            )
        )
        t_calc = time.time()
        total_pts = len(flat_yc)

        for b_start in range(0, total_pts, batch_size):
            b_end = min(b_start + batch_size, total_pts)

            b_dy_f = flat_dy_init[b_start:b_end]
            b_dx_f = flat_dx_init[b_start:b_end]
            b_dy_base = cp.round(b_dy_f).astype(cp.int32)
            b_dx_base = cp.round(b_dx_f).astype(cp.int32)

            idx_y_m = (flat_yc[b_start:b_end, None, None] + win_y_idx).astype(cp.int32)
            idx_x_m = (flat_xc[b_start:b_end, None, None] + win_x_idx).astype(cp.int32)
            idx_y_s = (
                flat_yc[b_start:b_end, None, None]
                + b_dy_base[:, None, None]
                + win_y_idx
            ).astype(cp.int32)
            idx_x_s = (
                flat_xc[b_start:b_end, None, None]
                + b_dx_base[:, None, None]
                + win_x_idx
            ).astype(cp.int32)

            m_batch = m_img_pad[idx_y_m, idx_x_m]
            s_batch = s_img_pad[idx_y_s, idx_x_s]

            dys, dxs, ccs = MatchEngine.compute_fine_offset2_batch2(
                m_batch, s_batch, b_dy_base, b_dx_base,
                refine_factor=refine_factor, refine_win=refine_win,
            )

            dy_map_gpu.ravel()[b_start:b_end] = dys
            dx_map_gpu.ravel()[b_start:b_end] = dxs
            cc_map_gpu.ravel()[b_start:b_end] = ccs

            if b_end % (batch_size * 20) == 0 or b_end == total_pts:
                elapsed = time.time() - t_calc
                print(
                    "   progress: {:5.1f}% | rate: {:7.0f} pts/s".format(
                        b_end / total_pts * 100, b_end / elapsed if elapsed else 0
                    )
                )

        mask_gpu = cc_map_gpu > cc_thresh
        dx_masked = cp.where(mask_gpu, dx_map_gpu, 0.0)
        dy_masked = cp.where(mask_gpu, dy_map_gpu, 0.0)
        off_masked = (dx_masked + 1j * dy_masked).astype(cp.complex64)

        SarIO.save_with_par(off_masked, "{}.offs".format(out_prefix))
        SarIO.save_with_par(dx_masked.get(), "{}.offs.rg".format(out_prefix))
        SarIO.save_with_par(dy_masked.get(), "{}.offs.az".format(out_prefix))
        SarIO.save_with_par(cc_map_gpu.get(), "{}.cc".format(out_prefix))

        print("=> [Done] finished. elapsed: {:.2f} s".format(time.time() - start_time))

        return {
            "dy": dy_map_gpu.get(),
            "dx": dx_map_gpu.get(),
            "cc": cc_map_gpu.get(),
            "mask": mask_gpu.get(),
            "step_y": step_y,
            "step_x": step_x,
            "ny": ny,
            "nx": nx,
        }

    @staticmethod
    def run_polynomial_refinement(results, n_poly: int = 3, max_iter: int = 3):
        """Polynomial fit with 2.5-sigma iterative outlier rejection.

        Returns (coeffs, sigma, accepted_samples, total_samples).
        """
        total_samples = results["ny"] * results["nx"]

        y_coords = np.arange(results["ny"]) * results["step_y"]
        x_coords = np.arange(results["nx"]) * results["step_x"]
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

        mask = results["mask"]
        v_y, v_x = yy[mask], xx[mask]
        v_dy, v_dx = results["dy"][mask], results["dx"][mask]

        if len(v_dy) <= n_poly:
            print("  [Error] too few valid points to fit.")
            return None, (0.0, 0.0), 0, total_samples

        final_coeffs = None
        final_sigma = (0.0, 0.0)

        for _ in range(max(1, max_iter)):
            (c_dy, c_dx), (res_dy, res_dx), (sig_dy, sig_dx) = (
                MatchEngine.fit_offset_polynomial(v_y, v_x, v_dy, v_dx, n_poly=n_poly)
            )

            clean_mask = (np.abs(res_dy) < 2.5 * sig_dy) & (
                np.abs(res_dx) < 2.5 * sig_dx
            )

            v_y, v_x = v_y[clean_mask], v_x[clean_mask]
            v_dy, v_dx = v_dy[clean_mask], v_dx[clean_mask]

            final_coeffs = (c_dy, c_dx)
            final_sigma = (sig_dy, sig_dx)

        return final_coeffs, final_sigma, len(v_dy), total_samples


# ===========================================================================
# 5. Subcommand: register
# ===========================================================================
def safe_mean(values) -> float:
    """Return nan for empty arrays to avoid runtime warnings."""
    return float(np.mean(values)) if getattr(values, "size", 0) else float("nan")


def print_report(title, results, coeffs, sigma, accepted, total):
    mask = results["mask"]
    print("\n" + "-" * 45)
    print("  [{}]".format(title))
    print(
        "  mean offset: Az={:8.4f}, Rg={:8.4f}".format(
            safe_mean(results["dy"][mask]), safe_mean(results["dx"][mask])
        )
    )
    print("  valid points: {} / {}".format(accepted, total))
    if coeffs:
        print(
            "  fit coefficient A0: Az={:8.5f}, Rg={:8.5f}".format(
                float(coeffs[0][0]), float(coeffs[1][0])
            )
        )
        print(
            "  std (sigma):  Az={:8.5f}, Rg={:8.5f}".format(
                float(sigma[0]), float(sigma[1])
            )
        )
    print("-" * 45)


def cmd_register(args) -> int:
    require_cupy()

    print("=" * 90)
    print("Two-pass SAR registration pipeline".center(78))
    print("=" * 90)

    print("\n>>> [STEP 1] Pass 1 coarse match (n_ovr={}) ...".format(args.pass1_novr))
    res_p1 = SARWorkflow.run_offset_tracking(
        m_par=args.m_par, m_bin=args.m_bin,
        s_par=args.s_par, s_bin=args.s_bin,
        step_y=args.step_y, step_x=args.step_x,
        win_h=args.win_h, win_w=args.win_w,
        cc_thresh=args.cc_thresh,
        n_ovr=args.pass1_novr,
        batch_size=args.batch_size,
        out_prefix=args.prefix_p1,
    )

    print_report(
        "PASS 1 report",
        res_p1,
        *SARWorkflow.run_polynomial_refinement(res_p1, n_poly=args.n_poly),
    )

    if args.skip_pass2:
        print("\n--skip-pass2 given, stopping here.")
        return 0

    print(
        "\n>>> [STEP 2] Pass 2 iterative fine match (factor: {}) ...".format(
            args.refine_factor
        )
    )

    res_p2 = SARWorkflow.run_offset_tracking2(
        m_par=args.m_par, m_bin=args.m_bin,
        s_par=args.s_par, s_bin=args.s_bin,
        dy_init=cp.asarray(res_p1["dy"]),
        dx_init=cp.asarray(res_p1["dx"]),
        step_y=args.step_y, step_x=args.step_x,
        win_h=args.win_h, win_w=args.win_w,
        cc_thresh=args.cc_thresh,
        refine_factor=args.refine_factor,
        refine_win=args.refine_win,
        batch_size=args.batch_size,
        out_prefix=args.prefix_p2,
    )

    print_report(
        "PASS 2 final report",
        res_p2,
        *SARWorkflow.run_polynomial_refinement(res_p2, n_poly=args.n_poly),
    )
    return 0


# ===========================================================================
# 6. Subcommands: stat-tif / stat-fft
# ===========================================================================
def read_tif(path):
    try:
        import rasterio
    except ImportError:
        die("missing dependency rasterio. Install it first:  pip install rasterio")
    with rasterio.open(str(path)) as src:
        return src.read()   # (bands, rows, cols)


def crop_array(data, row_start, row_end, col_start, col_end):
    return data[:, row_start:row_end, col_start:col_end]


def band_values(arr):
    """Flatten a ROI to 1-D values: two bands (re/im) -> phase; complex band -> phase."""
    if arr.shape[0] == 2:
        real = arr[0].flatten().astype(np.float64)
        imag = arr[1].flatten().astype(np.float64)
        values = np.arctan2(imag, real)
    else:
        band = arr[0]
        if np.iscomplexobj(band):
            values = np.angle(band.flatten())
        else:
            values = band.flatten().astype(np.float64)
    return values[~np.isnan(values)]


def command_string() -> str:
    return "python {}".format(" ".join([Path(sys.argv[0]).name] + sys.argv[1:]))


def cmd_stat_tif(args) -> int:
    values = band_values(
        crop_array(
            read_tif(args.tif),
            args.row_start, args.row_end, args.col_start, args.col_end,
        )
    )
    if values.size == 0:
        die("no valid pixels inside the ROI.")

    if args.bins:
        bins = np.array([float(item) for item in args.bins.split(",")], dtype=np.float64)
    elif args.min_val is not None and args.max_val is not None and args.step is not None:
        bins = np.arange(args.min_val, args.max_val + args.step, args.step)
    else:
        die("pass --bins, or pass --min-val/--max-val/--step together.")

    hist, bin_edges = np.histogram(values, bins=bins)
    percent = hist / len(values) * 100.0

    base = os.path.splitext(str(args.tif))[0]
    out_txt = base + "_stats.txt"
    with open(out_txt, "w", encoding="utf-8") as handle:
        handle.write("\nCommand:\n")
        handle.write(command_string() + "\n")
        handle.write("pixels: {}\n".format(len(values)))
        handle.write("mean: {}\n".format(float(np.mean(values))))
        handle.write("max: {}\n".format(float(np.max(values))))
        handle.write("min: {}\n".format(float(np.min(values))))
        handle.write("std: {}\n".format(float(np.std(values))))
        handle.write("\nBins:\n")
        for index in range(len(hist)):
            handle.write(
                "bin [{:.4f}, {:.4f}): count={} share={:.2f}%\n".format(
                    bin_edges[index], bin_edges[index + 1],
                    hist[index], percent[index],
                )
            )
    print("statistics written to:", out_txt)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        die("missing dependency matplotlib. Install it first:  pip install matplotlib")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        bin_edges[:-1], percent, width=np.diff(bin_edges),
        align="edge", edgecolor="black",
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Percent (%)")
    ax.set_title("Histogram (Percent)")
    plt.tight_layout()
    img_file = base + "_hist_percent.png"
    plt.savefig(img_file, dpi=300)
    plt.close(fig)
    print("percent histogram written to:", img_file)
    return 0


def cmd_stat_fft(args) -> int:
    arr = crop_array(
        read_tif(args.tif),
        args.row_start, args.row_end, args.col_start, args.col_end,
    )

    if arr.shape[0] == 2:
        complex_data = arr[0].astype(np.float64) + 1j * arr[1].astype(np.float64)
    else:
        band = arr[0]
        if not np.iscomplexobj(band):
            die("input is not complex; cannot compute a power spectrum.")
        complex_data = band.astype(np.complex128)

    complex_data = np.nan_to_num(complex_data)

    spectrum = np.fft.fftshift(np.fft.fft2(complex_data))
    power = np.abs(spectrum) ** 2
    total_energy = float(np.sum(power))
    if total_energy <= 0:
        die("total energy is 0; cannot compute spectral metrics.")

    rows, cols = power.shape
    cy, cx = rows // 2, cols // 2
    y_idx, x_idx = np.indices((rows, cols))
    radius = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    cutoff = min(rows, cols) / 4
    high_energy = float(np.sum(power[radius > cutoff]))
    low_energy = float(np.sum(power[radius <= cutoff]))
    high_ratio = high_energy / total_energy
    low_ratio = low_energy / total_energy

    probability = power.flatten() / total_energy
    probability = probability[probability > 0]
    spectral_entropy = float(-np.sum(probability * np.log2(probability)))

    base = os.path.splitext(str(args.tif))[0]
    out_txt = base + "_powerspec_stats.txt"
    with open(out_txt, "w", encoding="utf-8") as handle:
        handle.write("\nCommand:\n")
        handle.write(command_string() + "\n")
        handle.write("total energy: {}\n".format(total_energy))
        handle.write("high-band energy ratio: {:.4f}\n".format(high_ratio))
        handle.write("low-band energy ratio: {:.4f}\n".format(low_ratio))
        handle.write("spectral entropy: {:.4f}\n".format(spectral_entropy))
    print("power-spectrum metrics written to:", out_txt)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        die("missing dependency matplotlib. Install it first:  pip install matplotlib")

    metrics = ["Low Freq Ratio", "High Freq Ratio"]
    values = [low_ratio * 100, high_ratio * 100]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(metrics, values, color=["green", "red"], edgecolor="black")
    ax.set_ylabel("Percent (%)")
    ax.set_title("Low vs High Frequency Energy Ratio")
    plt.tight_layout()
    img_file = base + "_low_high_ratio.png"
    plt.savefig(img_file, dpi=300)
    plt.close(fig)
    print("low/high band ratio plot written to:", img_file)
    return 0


# ===========================================================================
# 7. CLI
# ===========================================================================
EXAMPLES = """\
Examples
--------
# 1) two-pass registration (needs a GPU and cupy)
python gccpr.py register --m-par 20240606.rslc.par --m-bin 20240606.rslc \\
    --s-par 20250731.rslc1.par --s-bin 20250731.rslc1 \\
    --step-y 12 --step-x 6 --win-h 128 --win-w 128 --cc-thresh 0.05

# 2) Pass 1 only
python gccpr.py register --m-par ... --m-bin ... --s-par ... --s-bin ... --skip-pass2

# 3) ROI value distribution
python gccpr.py stat-tif diff.tif --row-start 0 --row-end 512 \\
    --col-start 0 --col-end 512 --min-val -3 --max-val 3 --step 0.25

# 4) ROI power-spectrum metrics
python gccpr.py stat-fft diff.tif --row-start 0 --row-end 512 \\
    --col-start 0 --col-end 512
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gccpr.py",
        description="GC-CPR: GPU cross-correlation registration and quality statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + VERSION)
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # --- register ---
    p_reg = subparsers.add_parser(
        "register",
        help="two-pass iterative offset tracking (coarse Pass 1 + fine Pass 2)",
        description="Batched GPU NCC registration: Pass 1 zero-padded coarse, Pass 2 integer offset + zoom.",
    )
    p_reg.add_argument("--m-par", required=True, help="master .rslc.par")
    p_reg.add_argument("--m-bin", required=True, help="master .rslc binary")
    p_reg.add_argument("--s-par", required=True, help="slave .rslc.par")
    p_reg.add_argument("--s-bin", required=True, help="slave .rslc binary")
    p_reg.add_argument("--step-y", type=int, default=12, help="azimuth sampling step (default 12)")
    p_reg.add_argument("--step-x", type=int, default=6, help="range sampling step (default 6)")
    p_reg.add_argument("--win-h", type=int, default=128, help="window height (default 128)")
    p_reg.add_argument("--win-w", type=int, default=128, help="window width (default 128)")
    p_reg.add_argument(
        "--cc-thresh", type=float, default=0.05, help="correlation mask threshold (default 0.05)"
    )
    p_reg.add_argument(
        "--pass1-novr", type=int, default=1, help="Pass 1 zero-padding factor (default 1)"
    )
    p_reg.add_argument(
        "--refine-factor", type=int, default=16,
        help="Pass 2 sub-pixel zoom factor (default 16)",
    )
    p_reg.add_argument(
        "--refine-win", type=float, default=1.5,
        help="Pass 2 zoom half-width in pixels (default 1.5)",
    )
    p_reg.add_argument(
        "--n-poly", type=int, default=3, choices=(1, 3, 4, 6),
        help="number of polynomial coefficients (default 3)",
    )
    p_reg.add_argument(
        "--batch-size", type=int, default=512, help="windows per batch (default 512)"
    )
    p_reg.add_argument(
        "--prefix-p1", default="offset_p1_raw", help="Pass 1 output prefix"
    )
    p_reg.add_argument(
        "--prefix-p2", default="offset_p2_raw", help="Pass 2 output prefix"
    )
    p_reg.add_argument("--skip-pass2", action="store_true", help="run Pass 1 only")
    p_reg.set_defaults(func=cmd_register)

    # --- stat-tif ---
    p_stat = subparsers.add_parser(
        "stat-tif",
        help="value distribution over a GeoTIFF ROI",
        description="Bin the values/phase inside a ROI; write a text histogram and a PNG.",
    )
    p_stat.add_argument("tif", help="GeoTIFF path")
    p_stat.add_argument("--row-start", type=int, required=True, help="first row")
    p_stat.add_argument("--row-end", type=int, required=True, help="end row")
    p_stat.add_argument("--col-start", type=int, required=True, help="first column")
    p_stat.add_argument("--col-end", type=int, required=True, help="end column")
    p_stat.add_argument("--min-val", type=float, default=None, help="histogram minimum")
    p_stat.add_argument("--max-val", type=float, default=None, help="histogram maximum")
    p_stat.add_argument("--step", type=float, default=None, help="histogram step")
    p_stat.add_argument(
        "--bins", default=None, help="custom bin edges, comma separated, e.g. 0,0.5,1,2,3"
    )
    p_stat.set_defaults(func=cmd_stat_tif)

    # --- stat-fft ---
    p_fft = subparsers.add_parser(
        "stat-fft",
        help="power-spectrum metrics over a complex GeoTIFF ROI",
        description="Total energy, low/high band energy ratio and spectral entropy of a complex ROI.",
    )
    p_fft.add_argument("tif", help="complex GeoTIFF path")
    p_fft.add_argument("--row-start", type=int, required=True, help="first row")
    p_fft.add_argument("--row-end", type=int, required=True, help="end row")
    p_fft.add_argument("--col-start", type=int, required=True, help="first column")
    p_fft.add_argument("--col-end", type=int, required=True, help="end column")
    p_fft.set_defaults(func=cmd_stat_fft)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import math
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, asdict
from statistics import mean as _pymean
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import Dict, Any, Optional, Tuple, List

# ----------------------------- Core parsing & ARV -----------------------------

def _split_preserving_trailing(text: str, separator: str) -> List[str]:
    """Split ``text`` while retaining trailing empty fields."""
    if separator == " ":
        return text.split()
    if not separator:
        return [text]
    parts = text.split(separator)
    expected = text.count(separator) + 1
    if len(parts) < expected:
        parts.extend([""] * (expected - len(parts)))
    return parts


def parse_header(lines, separator="\t", data_start_override=None):
    """
    Returns:
      interval_s (float): sampling interval in seconds
      channel_titles (list[str]): ordered channel names from ChannelTitle=
      data_start_idx (int): index of first numeric row
    """
    interval_s = None
    channel_titles = None
    data_start_idx = None

    for i, ln in enumerate(lines):
        line = ln.rstrip("\r\n")
        stripped = line.lstrip()

        if stripped.startswith("Interval="):
            # Often "Interval=\t1 s"
            parts = [p for p in stripped.split("\t") if p]
            try:
                val = parts[1] if len(parts) > 1 else parts[0].split("=", 1)[-1]
                interval_s = float(val.split()[0])
            except Exception:
                pass

        if stripped.startswith("ChannelTitle="):
            titles = stripped.split("=", 1)[-1]
            channel_titles = [t.strip() for t in titles.split("\t")]

        if data_start_idx is None:
            row = ln.rstrip("\r\n")
            if separator == " ":
                parts = row.split()
            else:
                parts = _split_preserving_trailing(row, separator)
            if len(parts) >= 2 and parts[0].strip().isdigit():
                data_start_idx = i

    if data_start_override is not None:
        data_start_idx = max(0, min(len(lines) - 1, int(data_start_override)))

    if interval_s is None:
        raise ValueError("Could not parse sampling Interval= from file header.")
    if channel_titles is None:
        raise ValueError("Could not parse ChannelTitle= from file header.")
    if data_start_idx is None:
        raise ValueError("Could not locate start of numeric data.")
    return interval_s, channel_titles, data_start_idx


def safe_float(x: str, decimal: str = ".") -> float:
    try:
        if isinstance(x, str) and decimal != ".":
            x = x.replace(decimal, ".")
        return float(x)
    except Exception:
        return float("nan")


def load_series(filepath, layout_config=None):
    """
    Reads the file, returns (interval_s, {channel_name: [values...]}, channel_titles)
    Only channels listed in ChannelTitle= are returned.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    separator = "\t"
    decimal = "."
    data_start_override = None
    if layout_config is not None:
        separator = layout_config.separator or "\t"
        decimal = layout_config.decimal or "."
        if layout_config.first_data_row:
            data_start_override = layout_config.first_data_row - 1

    interval_s, raw_channel_titles, data_start = parse_header(
        lines,
        separator=separator,
        data_start_override=data_start_override,
    )
    channel_titles = []
    name_counts: Dict[str, int] = {}
    for idx, raw_name in enumerate(raw_channel_titles):
        if raw_name:
            base_name = raw_name
        else:
            base_name = f"__unnamed_column_{idx + 2}"

        count = name_counts.get(base_name, 0)
        if count:
            resolved_name = f"{base_name}__{count + 1}"
        else:
            resolved_name = base_name
        name_counts[base_name] = count + 1
        channel_titles.append(resolved_name)

    n_channels = len(channel_titles)
    series_map = {name: [] for name in channel_titles}

    comment_key = None
    if layout_config is not None:
        try:
            comment_column = getattr(layout_config, "comment_column", None)
        except AttributeError:
            comment_column = None
        if comment_column:
            try:
                mapped_index = int(comment_column) - 2
            except (TypeError, ValueError):
                mapped_index = None
            if mapped_index is not None and 0 <= mapped_index < len(channel_titles):
                comment_key = channel_titles[mapped_index]

    # IMPORTANT: No artificial limit here — reads the ENTIRE dataset
    for ln in lines[data_start:]:
        row_txt = ln.rstrip("\r\n")
        if separator == " ":
            row = row_txt.split()
        else:
            row = _split_preserving_trailing(row_txt, separator)
        if not row or not row[0].strip().isdigit():
            continue
        if len(row) < 1 + n_channels:
            row.extend([""] * (1 + n_channels - len(row)))
        elif len(row) > 1 + n_channels:
            row = row[: 1 + n_channels]

        vals = row[1:1 + n_channels]
        for name, val in zip(channel_titles, vals):
            if comment_key is not None and name == comment_key:
                series_map[name].append(val)
            else:
                series_map[name].append(safe_float(val, decimal=decimal))

    return interval_s, series_map, channel_titles


def arv(values):
    """Average Real Variability = mean absolute successive difference (unweighted)."""
    if len(values) < 2:
        return float("nan")
    num, denom = 0.0, 0
    prev = values[0]
    for v in values[1:]:
        if not (math.isnan(prev) or math.isnan(v)):
            num += abs(v - prev)
            denom += 1
        prev = v
    return num / denom if denom > 0 else float("nan")


# -------------------------- Outlier filtering (artifacts) ---------------------

FILTER_METHOD_ORDER = ["moving_average", "absolute_change", "std_from_mean"]
FILTER_METHODS = {
    "moving_average": {"label": "Moving average"},
    "absolute_change": {"label": "Absolute beat-beat change"},
    "std_from_mean": {"label": "STD from mean"},
}
_FILTER_LABEL_TO_CODE = {info["label"]: code for code, info in FILTER_METHODS.items()}
_LEGACY_FILTER_ALIASES = {
    "rolling": "moving_average",
    "global": "std_from_mean",
}


def normalize_filter_method(value: Optional[str]) -> str:
    if not value:
        return "moving_average"
    if value in FILTER_METHODS:
        return value
    if value in _FILTER_LABEL_TO_CODE:
        return _FILTER_LABEL_TO_CODE[value]
    lowered = value.lower()
    if lowered in _LEGACY_FILTER_ALIASES:
        return _LEGACY_FILTER_ALIASES[lowered]
    for label, code in _FILTER_LABEL_TO_CODE.items():
        if lowered == label.lower():
            return code
    normalized = lowered.replace(" ", "_")
    if normalized in FILTER_METHODS:
        return normalized
    return "moving_average"


def filter_label_from_value(value: Optional[str]) -> str:
    code = normalize_filter_method(value)
    return FILTER_METHODS[code]["label"]

def _nanmean(xs):
    vals = [x for x in xs if not math.isnan(x)]
    return float("nan") if not vals else sum(vals)/len(vals)

def _nansd(xs, ddof=1):
    vals = [x for x in xs if not math.isnan(x)]
    n = len(vals)
    if n <= ddof:
        return float("nan")
    m = sum(vals)/n
    var = sum((x - m)**2 for x in vals) / (n - ddof)
    return math.sqrt(var)

def _mad(xs):
    """Median Absolute Deviation (MAD), ignoring NaN."""
    vals = [x for x in xs if not math.isnan(x)]
    if not vals:
        return float("nan")
    vals_sorted = sorted(vals)
    mid = len(vals_sorted)//2
    med = (vals_sorted[mid] if len(vals_sorted)%2==1
           else 0.5*(vals_sorted[mid-1]+vals_sorted[mid]))
    abs_dev = [abs(x - med) for x in vals_sorted]
    abs_dev.sort()
    mid2 = len(abs_dev)//2
    mad = (abs_dev[mid2] if len(abs_dev)%2==1
           else 0.5*(abs_dev[mid2-1]+abs_dev[mid2]))
    return mad

def filter_outliers_global(values, k_sd=3.0):
    """
    Simple global filter: drop points > k_sd * SD from the global mean.
    Returns (kept_values, excluded_values).
    """
    m = _nanmean(values)
    s = _nansd(values, ddof=1)
    clean = [v for v in values if not math.isnan(v)]
    if math.isnan(m) or math.isnan(s) or s == 0:
        return clean, []
    lo, hi = m - k_sd * s, m + k_sd * s
    kept, excluded = [], []
    for v in clean:
        if lo <= v <= hi:
            kept.append(v)
        else:
            excluded.append(v)
    return kept, excluded

def filter_outliers_rolling_robust(values, interval_s, window_s=15.0, k_mad=4.0):
    """
    Rolling robust filter for exercise tests:
    - Compute rolling median and rolling MAD over ~window_s.
    - Convert MAD to robust sigma: sigma ≈ 1.4826 * MAD.
    - Drop points with |x - median| > k_mad * sigma_robust.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0 or interval_s <= 0:
        return [v for v in vals if not math.isnan(v)], []

    w = max(3, int(round(window_s / interval_s)))  # window length in samples
    half = w // 2

    kept, excluded = [], []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = [v for v in vals[lo:hi] if not math.isnan(v)]
        if not window:
            continue
        # median
        ws = sorted(window)
        mid = len(ws)//2
        med = ws[mid] if len(ws)%2==1 else 0.5*(ws[mid-1]+ws[mid])

        # MAD
        abs_dev = [abs(v - med) for v in window]
        abs_dev.sort()
        mid2 = len(abs_dev)//2
        mad = abs_dev[mid2] if len(abs_dev)%2==1 else 0.5*(abs_dev[mid2-1]+abs_dev[mid2])
        sigma = 1.4826 * mad  # ~equivalent to SD if normal
        x = vals[i]
        if math.isnan(x):
            continue
        if sigma == 0:
            # if flat window, keep point if exactly med
            if abs(x - med) <= 1e-12:
                kept.append(x)
            else:
                excluded.append(x)
        else:
            if abs(x - med) <= k_mad * sigma:
                kept.append(x)
            else:
                excluded.append(x)
    return kept, excluded


def filter_outliers_absolute_change(values, max_delta):
    """Exclude beats where the absolute change from the previous kept beat exceeds ``max_delta``."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return [], []
    try:
        threshold = float(max_delta)
    except (TypeError, ValueError):
        threshold = 0.0
    if threshold <= 0:
        return clean, []
    kept, excluded = [], []
    prev = None
    for v in clean:
        if prev is None:
            kept.append(v)
            prev = v
            continue
        if abs(v - prev) <= threshold:
            kept.append(v)
            prev = v
        else:
            excluded.append(v)
    return kept, excluded


def compute_metrics(series, use_filter=False, filter_method="moving_average", interval_s=1.0,
                    k_sd=3.0, window_s=15.0, k_mad=4.0, delta_cutoff=0.0):
    """
    Returns dict with:
      n_raw, n_kept, mean, sd, cv, arv
    """
    raw = [v for v in series if not math.isnan(v)]
    n_raw = len(raw)

    if not use_filter:
        kept = list(raw)
        excluded = []
    else:
        method_code = normalize_filter_method(filter_method)
        if method_code == "std_from_mean":
            kept, excluded = filter_outliers_global(raw, k_sd=float(k_sd))
        elif method_code == "absolute_change":
            kept, excluded = filter_outliers_absolute_change(raw, max_delta=float(delta_cutoff))
        else:
            kept, excluded = filter_outliers_rolling_robust(
                raw,
                interval_s=interval_s,
                window_s=float(window_s),
                k_mad=float(k_mad),
            )
    n_kept = len(kept)
    m = _pymean(kept) if n_kept else float("nan")

    # sample SD
    sd = _nansd(kept, ddof=1)
    cv = (sd / m * 100.0) if (n_kept > 1 and (m not in (0.0, 0) and not math.isnan(m))) else float("nan")
    arv_val = arv(kept)
    return {
        "n_raw": n_raw,
        "n_kept": n_kept,
        "mean": m,
        "sd": sd,
        "cv": cv,
        "arv": arv_val,
        "excluded_values": excluded,
    }


def compute_all_for_file(path, window_spec, use_filter, filter_method,
                         k_sd, window_s, k_mad, delta_cutoff, layout_config=None):
    interval_s, series_map, channel_titles = load_series(path, layout_config=layout_config)

    layout_cfg = layout_config or LayoutConfig.default()

    def resolve_column(column_number, label):
        try:
            idx = int(column_number)
        except (TypeError, ValueError):
            raise KeyError(f"{label} column is not a valid number: {column_number!r}")
        if idx < 2:
            raise KeyError(f"{label} column must be 2 or greater (column 1 is typically time).")
        mapped_index = idx - 2
        if mapped_index < 0 or mapped_index >= len(channel_titles):
            raise KeyError(
                f"{label} column {idx} is out of range for available data columns 2-{len(channel_titles) + 1}."
            )
        return channel_titles[mapped_index]

    sbp_key = resolve_column(getattr(layout_cfg, "sbp_column", 2), "SBP")
    dbp_key = resolve_column(getattr(layout_cfg, "dbp_column", 3), "DBP")
    map_key = resolve_column(getattr(layout_cfg, "map_column", 4), "MAP")

    comment_series = None
    comment_column = getattr(layout_cfg, "comment_column", None)
    if comment_column:
        try:
            comment_key = resolve_column(comment_column, "Comment")
        except KeyError:
            comment_key = None
        if comment_key is not None and comment_key in series_map:
            comment_series = series_map[comment_key]

    n_samples = len(series_map[sbp_key])
    start_idx, end_idx, start_time, end_time = resolve_window_spec(
        window_spec,
        interval_s,
        n_samples,
        comment_series,
    )

    sbp = series_map[sbp_key][start_idx:end_idx]
    dbp = series_map[dbp_key][start_idx:end_idx]
    mapp = series_map[map_key][start_idx:end_idx]

    ms_sbp = compute_metrics(
        sbp, use_filter, filter_method, interval_s, k_sd, window_s, k_mad, delta_cutoff
    )
    ms_dbp = compute_metrics(
        dbp, use_filter, filter_method, interval_s, k_sd, window_s, k_mad, delta_cutoff
    )
    ms_map = compute_metrics(
        mapp, use_filter, filter_method, interval_s, k_sd, window_s, k_mad, delta_cutoff
    )

    return {
        "file": os.path.basename(path),
        "interval_s": interval_s,
        "start_s": float(start_time),
        "end_s": None if end_time is None else float(end_time),

        # SBP
        "n_SBP_raw": ms_sbp["n_raw"],
        "n_SBP_kept": ms_sbp["n_kept"],
        "Mean_SBP": ms_sbp["mean"],
        "SD_SBP": ms_sbp["sd"],
        "CV_SBP_pct": ms_sbp["cv"],
        "ARV_SBP": ms_sbp["arv"],

        # DBP
        "n_DBP_raw": ms_dbp["n_raw"],
        "n_DBP_kept": ms_dbp["n_kept"],
        "Mean_DBP": ms_dbp["mean"],
        "SD_DBP": ms_dbp["sd"],
        "CV_DBP_pct": ms_dbp["cv"],
        "ARV_DBP": ms_dbp["arv"],

        # MAP
        "n_MAP_raw": ms_map["n_raw"],
        "n_MAP_kept": ms_map["n_kept"],
        "Mean_MAP": ms_map["mean"],
        "SD_MAP": ms_map["sd"],
        "CV_MAP_pct": ms_map["cv"],
        "ARV_MAP": ms_map["arv"],
        "excluded_values": {
            "SBP": list(ms_sbp.get("excluded_values", [])),
            "DBP": list(ms_dbp.get("excluded_values", [])),
            "MAP": list(ms_map.get("excluded_values", [])),
        },
    }

# ----------------------------- Helpers for GUI -----------------------------

def parse_time_any(s):
    """
    Accepts:
      - "" -> None
      - "600" (seconds)
      - "mm:ss" (e.g., "10:00")
      - "hh:mm:ss"
    Returns seconds (float) or None.
    """
    s = (s or "").strip()
    if not s:
        return None
    if ":" not in s:
        # plain seconds
        return float(s)
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    raise ValueError("Time format must be seconds, mm:ss, or hh:mm:ss")


def _time_to_index(time_s: Optional[float], interval_s: float, n_samples: int, is_start: bool) -> int:
    if time_s is None:
        return 0 if is_start else n_samples
    if interval_s <= 0:
        return 0 if is_start else n_samples
    if is_start:
        idx = int(math.floor(time_s / interval_s))
    else:
        idx = int(math.ceil(time_s / interval_s))
    if idx < 0:
        return 0
    if idx > n_samples:
        return n_samples
    return idx


def _find_comment_index(comments, text: str, find_last: bool = False) -> Optional[int]:
    if not comments:
        return None
    needle = text.lower()
    indices = range(len(comments) - 1, -1, -1) if find_last else range(len(comments))
    for idx in indices:
        comment = comments[idx]
        if comment is None:
            continue
        if needle in str(comment).lower():
            return idx
    return None


def resolve_window_spec(
    window_spec: Dict[str, Dict[str, Any]],
    interval_s: float,
    series_length: int,
    comments,
) -> Tuple[int, int, float, Optional[float]]:
    total_time = max(0.0, interval_s * series_length)

    def resolve_endpoint(spec: Dict[str, Any], is_start: bool, label: str) -> Tuple[int, float]:
        mode = (spec.get("mode") or "none").lower()
        default_idx = 0 if is_start else series_length
        default_time = 0.0 if is_start else total_time

        if mode in ("none",):
            return default_idx, (0.0 if is_start else total_time)

        if mode == "time":
            time_val = spec.get("value")
            if time_val is None:
                return default_idx, (0.0 if is_start else total_time)
            time_val = float(time_val)
            if time_val < 0:
                time_val = 0.0
            if time_val > total_time:
                time_val = total_time
            idx = _time_to_index(time_val, interval_s, series_length, is_start)
            return idx, time_val

        if mode == "epoch":
            epoch_val = spec.get("value")
            if epoch_val is None:
                return default_idx, (0.0 if is_start else total_time)
            epoch_val = int(epoch_val)
            if is_start:
                idx = max(0, min(series_length, epoch_val - 1))
            else:
                idx = max(0, min(series_length, epoch_val))
            time_val = idx * interval_s
            if time_val > total_time:
                time_val = total_time
            return idx, time_val

        if mode == "comment":
            if comments is None:
                raise ValueError(f"{label} comment anchor requires a comment column to be configured.")
            text = (spec.get("text") or "").strip()
            if not text:
                raise ValueError(f"{label} comment text is required to anchor the window.")
            offset = float(spec.get("offset") or 0.0)
            idx = _find_comment_index(comments, text, find_last=not is_start)
            if idx is None:
                raise ValueError(f"{label} comment containing '{text}' was not found.")
            time_val = idx * interval_s + offset
            if time_val < -1e-9:
                raise ValueError(f"{label} comment offset places the window before the start of the data.")
            if time_val > total_time + 1e-9:
                raise ValueError(f"{label} comment offset places the window beyond the end of the data.")
            idx = _time_to_index(time_val, interval_s, series_length, is_start)
            return idx, time_val

        raise ValueError(f"Unsupported window mode '{mode}'.")

    start_spec = window_spec.get("start", {}) if window_spec else {}
    end_spec = window_spec.get("end", {}) if window_spec else {}
    start_idx, start_time = resolve_endpoint(start_spec, True, "Start")
    end_idx, end_time = resolve_endpoint(end_spec, False, "End")

    start_mode = (start_spec.get("mode") or "none").lower()
    end_mode = (end_spec.get("mode") or "none").lower()
    start_explicit = start_mode == "comment" or (
        start_mode in ("time", "epoch") and start_spec.get("value") is not None
    )
    end_explicit = end_mode == "comment" or (
        end_mode in ("time", "epoch") and end_spec.get("value") is not None
    )

    if end_idx < start_idx or (end_idx == start_idx and start_explicit and end_explicit):
        raise ValueError("End boundary occurs before start boundary after resolving the window specification.")

    if end_idx >= series_length and (end_spec.get("mode") in (None, "", "none") or end_spec.get("value") is None):
        end_time = None
    else:
        if end_time is not None and end_time > total_time:
            end_time = total_time

    if start_time is None:
        start_time = 0.0
    if end_time is not None and end_time < 0:
        end_time = 0.0

    return start_idx, end_idx, start_time, end_time

def fmt_float(x, digits=6):
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        return f"{x:.{digits}g}"
    return str(x)

# --------------------------- Layout configuration ---------------------------


@dataclass
class LayoutConfig:
    name: str
    header_lines: int = 9
    first_data_row: int = 10
    separator: str = "\t"
    decimal: str = "."
    time_column: int = 1
    sbp_column: int = 2
    dbp_column: int = 3
    map_column: int = 4
    comment_column: Optional[int] = None

    @classmethod
    def default(cls):
        return cls(
            name="Default",
            header_lines=9,
            first_data_row=10,
            separator="\t",
            decimal=".",
            time_column=1,
            sbp_column=2,
            dbp_column=3,
            map_column=4,
            comment_column=None,
        )

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return cls.default()
        def parse_int(value, default):
            if value in (None, ""):
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        # Allow legacy configs that stored channel names instead of column numbers.
        time_column = parse_int(data.get("time_column"), 1)
        sbp_column = parse_int(data.get("sbp_column"), 2)
        dbp_column = parse_int(data.get("dbp_column"), 3)
        map_column = parse_int(data.get("map_column"), 4)
        comment_column = parse_int(data.get("comment_column"), None)
        if comment_column is not None and comment_column < 2:
            comment_column = None

        return cls(
            name=data.get("name", "Unnamed"),
            header_lines=int(data.get("header_lines", 0) or 0),
            first_data_row=int(data.get("first_data_row", 1) or 1),
            separator=data.get("separator", "\t") or "\t",
            decimal=data.get("decimal", ".") or ".",
            time_column=time_column,
            sbp_column=sbp_column,
            dbp_column=dbp_column,
            map_column=map_column,
            comment_column=comment_column,
        )

    def to_dict(self):
        return asdict(self)

# ---------------------------------- GUI ------------------------------------

class ARVApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Blood Pressure Variability Tool")
        self.geometry("1180x640")
        self.minsize(1024, 600)

        self.filepaths = []
        self.results = []

        self.layout_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layout_configs.json")
        self.layout_configs = {}
        self.layout_config_order = []
        self.active_layout_name = None
        self._load_layout_configs()
        self.layout_config_var = tk.StringVar(value=self.active_layout_name)

        # Preference state variables
        self.start_var = tk.StringVar(value="")
        self.end_var = tk.StringVar(value="")
        self.start_mode_var = tk.StringVar(value="Time")
        self.end_mode_var = tk.StringVar(value="Time")
        self.start_epoch_var = tk.StringVar(value="")
        self.end_epoch_var = tk.StringVar(value="")
        self.start_comment_var = tk.StringVar(value="")
        self.end_comment_var = tk.StringVar(value="")
        self.start_offset_var = tk.StringVar(value="0")
        self.end_offset_var = tk.StringVar(value="0")
        self.filter_on = tk.BooleanVar(value=False)
        self.filter_method = tk.StringVar(value=FILTER_METHODS["moving_average"]["label"])
        self.k_sd_var = tk.StringVar(value="3.0")
        self.window_s_var = tk.StringVar(value="15")
        self.k_mad_var = tk.StringVar(value="4.0")
        self.delta_cutoff_var = tk.StringVar(value="20")
        self.output_std = tk.BooleanVar(value=True)
        self.output_cov = tk.BooleanVar(value=True)
        self.output_arv = tk.BooleanVar(value=True)

        self._preferences_window = None
        self._preferences_snapshot = None
        self._start_mode_frames = {}
        self._end_mode_frames = {}
        self._filter_method_frames = {}
        self._filter_method_combo = None
        self._filter_method_inputs = []
        self._layout_window = None
        self._layout_preview_tree = None
        self._layout_preview_row_tree = None
        self._preview_y_scrollbar = None
        self._preview_scroll_syncing = False
        self._layout_file_label_var = None
        self._layout_field_vars = {}
        self._layout_field_traces = []
        self._layout_config_combo = None

        self._exclusions_window = None
        self._exclusion_tree = None
        self._exclusion_records = []

        # Styling (optional nicer look)
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.configure(menu=self._build_menubar())

        # Layout: left file list, right results
        root_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        root_pane.pack(fill=tk.BOTH, expand=True)
        self._root_pane = root_pane
        self._initial_sash_positioned = False
        self._sash_bind_id = None

        self.left_frame = ttk.Frame(root_pane, padding=8)
        root_pane.add(self.left_frame, weight=1)

        self.right_frame = ttk.Frame(root_pane, padding=(4, 8, 8, 8))
        root_pane.add(self.right_frame, weight=4)

        # Ensure the left pane starts at a reasonable default width.
        self._sash_bind_id = root_pane.bind("<Configure>", self._ensure_initial_sash)

        # Left panel
        ttk.Label(self.left_frame, text="Loaded files", anchor="w").pack(fill=tk.X, pady=(0, 4))
        list_frame = ttk.Frame(self.left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.files_list = tk.Listbox(list_frame, height=20, exportselection=False)
        self.files_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.files_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_list.configure(yscrollcommand=scrollbar.set)
        self.files_list.bind("<<ListboxSelect>>", self._update_layout_preview)

        # Right panel top controls
        controls = ttk.Frame(self.right_frame)
        controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(controls, text="Compute", command=self.compute).pack(side=tk.RIGHT)
        ttk.Button(controls, text="Save CSV…", command=self.save_csv).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(controls, text="View exclusions", command=self.view_exclusions).pack(side=tk.RIGHT, padx=(0, 8))

        # Results table
        self._tree_columns = (
            "file", "interval_s", "start_s", "end_s",
            "n_SBP_raw", "n_SBP_kept", "Mean_SBP", "SD_SBP", "CV_SBP_pct", "ARV_SBP",
            "n_DBP_raw", "n_DBP_kept", "Mean_DBP", "SD_DBP", "CV_DBP_pct", "ARV_DBP",
            "n_MAP_raw", "n_MAP_kept", "Mean_MAP", "SD_MAP", "CV_MAP_pct", "ARV_MAP",
        )
        tree_container = ttk.Frame(self.right_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)
        tree_container.rowconfigure(0, weight=1)
        tree_container.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(tree_container, columns=self._tree_columns, show="headings", height=18)
        for c in self._tree_columns:
            self.tree.heading(c, text=c)
            base_w = 100
            if c == "file":
                w = 240
            elif "Mean" in c or "SD_" in c or "CV_" in c or "ARV_" in c:
                w = 110
            else:
                w = base_w
            self.tree.column(c, width=w, anchor=tk.CENTER, stretch=False)
        self.tree.grid(row=0, column=0, sticky="nsew")

        tree_y_scroll = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        tree_y_scroll.grid(row=0, column=1, sticky="ns")
        tree_x_scroll = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        tree_x_scroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=tree_y_scroll.set, xscrollcommand=tree_x_scroll.set)

        self._apply_result_column_preferences()

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))

    # --------------------------- Menu construction ------------------------
    def _build_menubar(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Add files", command=self.add_files)
        file_menu.add_command(label="Clear file list", command=self.clear_files)
        file_menu.add_separator()
        file_menu.add_command(label="Format file layout", command=self.open_layout_dialog)
        menubar.add_cascade(label="Files", menu=file_menu)

        menubar.add_command(label="Preferences", command=self.open_preferences)
        menubar.add_command(label="About", command=self.open_about)

        return menubar

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        added = 0
        if not paths:
            folder = filedialog.askdirectory(title="Select folder with files")
            if folder:
                for entry in sorted(os.listdir(folder)):
                    full = os.path.join(folder, entry)
                    if os.path.isfile(full):
                        if full not in self.filepaths:
                            self.filepaths.append(full)
                            added += 1
            else:
                return
        else:
            for p in paths:
                if p not in self.filepaths:
                    self.filepaths.append(p)
                    added += 1

        if added:
            self.update_file_list()
        self.status.set(f"Added {added} file(s). Total: {len(self.filepaths)}")

    def clear_files(self):
        self.filepaths = []
        self.results = []
        self.update_file_list()
        for i in self.tree.get_children():
            self.tree.delete(i)
        self._refresh_exclusions_window()
        self.status.set("Cleared files and results.")

    def update_file_list(self):
        self.files_list.delete(0, tk.END)
        for path in self.filepaths:
            self.files_list.insert(tk.END, os.path.basename(path))
        self._update_layout_preview()

    def _ensure_initial_sash(self, event=None):
        if self._initial_sash_positioned:
            return
        # Avoid trying to position before the widget has a meaningful size.
        if event is not None and getattr(event, "width", 0) <= 1:
            return
        try:
            self._root_pane.sashpos(0, 250)
        except tk.TclError:
            return
        self._initial_sash_positioned = True
        # Stop listening once the sash has been positioned.
        if self._sash_bind_id is not None:
            self._root_pane.unbind("<Configure>", self._sash_bind_id)
            self._sash_bind_id = None

    def _autofit_tree_columns(self, tree, min_width=60, padding=24, max_width=600):
        """Resize treeview columns to fit their visible content.

        Args:
            tree: The ``ttk.Treeview`` widget to resize.
            min_width: Smallest width (pixels) allowed for any column.
            padding: Extra pixels to add beyond the measured content width.
            max_width: Optional hard cap for column width (``None`` for no cap).
        """
        if not tree:
            return
        columns = tree["columns"]
        if isinstance(columns, str):
            columns = (columns,)
        if not columns:
            return
        display_columns = tree.cget("displaycolumns")
        if display_columns not in (None, "", "#all"):
            if isinstance(display_columns, str):
                display_columns = (display_columns,)
            columns = display_columns

        try:
            body_font = tkfont.nametofont(tree.cget("font"))
        except tk.TclError:
            body_font = tkfont.nametofont("TkDefaultFont")

        try:
            heading_font = tkfont.nametofont("TkHeadingFont")
        except tk.TclError:
            heading_font = body_font

        for col in columns:
            heading_text = tree.heading(col).get("text", "")
            content_width = heading_font.measure(heading_text)
            for item in tree.get_children():
                cell_text = tree.set(item, col)
                content_width = max(content_width, body_font.measure(str(cell_text)))

            target_width = max(min_width, content_width + padding)
            if max_width is not None:
                target_width = min(target_width, max_width)
            tree.column(col, width=int(target_width), stretch=False)

    def _scroll_preview_y(self, *args):
        if self._layout_preview_tree is not None:
            self._layout_preview_tree.yview(*args)

    def _on_preview_y_scroll(self, source, first, last):
        if self._preview_y_scrollbar is not None:
            self._preview_y_scrollbar.set(first, last)

        if self._preview_scroll_syncing:
            return

        other = self._layout_preview_row_tree if source == "data" else self._layout_preview_tree
        if other is None:
            return

        try:
            fraction = float(first)
        except (TypeError, ValueError):
            return

        try:
            self._preview_scroll_syncing = True
            other.yview_moveto(fraction)
        finally:
            self._preview_scroll_syncing = False

    # --------------------------- Layout config logic ---------------------

    def _iter_layout_configs(self):
        for name in self.layout_config_order:
            cfg = self.layout_configs.get(name)
            if cfg is not None:
                yield cfg

    def _load_layout_configs(self):
        data = None
        if os.path.exists(self.layout_config_path):
            try:
                with open(self.layout_config_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                data = None

        self.layout_configs = {}
        self.layout_config_order = []

        if isinstance(data, dict):
            for item in data.get("configs", []):
                cfg = LayoutConfig.from_dict(item)
                self.layout_configs[cfg.name] = cfg
                self.layout_config_order.append(cfg.name)
            active = data.get("active")
        else:
            active = None

        if not self.layout_configs:
            default_cfg = LayoutConfig.default()
            self.layout_configs[default_cfg.name] = default_cfg
            self.layout_config_order = [default_cfg.name]

        if active not in self.layout_configs:
            active = self.layout_config_order[0]

        self.active_layout_name = active

    def _persist_layout_configs(self):
        data = {
            "active": self.active_layout_name,
            "configs": [cfg.to_dict() for cfg in self._iter_layout_configs()],
        }
        try:
            with open(self.layout_config_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            if hasattr(self, "status"):
                self.status.set(f"Warning: could not save layout configs: {exc}")

    def _get_active_layout(self):
        if self.active_layout_name in self.layout_configs:
            return self.layout_configs[self.active_layout_name]
        return next(self._iter_layout_configs(), LayoutConfig.default())

    def _set_active_layout(self, name, persist=True):
        if name not in self.layout_configs:
            return
        self.active_layout_name = name
        try:
            self.layout_config_var.set(name)
        except Exception:
            pass
        if persist:
            self._persist_layout_configs()
        if hasattr(self, "status"):
            self.status.set(f"Active layout set to '{name}'.")

    def _layout_names(self):
        return [cfg.name for cfg in self._iter_layout_configs()]

    def _separator_display(self, separator):
        mapping = {
            "\t": "Tab",
            ",": "Comma",
            ";": "Semicolon",
            " ": "Space",
        }
        return mapping.get(separator, separator or "Tab")

    def _decimal_display(self, decimal):
        mapping = {
            ".": "Dot",
            ",": "Comma",
        }
        return mapping.get(decimal, decimal or "Dot")

    def _collect_layout_from_fields(self, target_name=None, quiet=False):
        if not self._layout_field_vars:
            return None

        try:
            first_data_row = int((self._layout_field_vars["first_data_row"].get() or "").strip())
        except ValueError:
            if quiet:
                return None
            messagebox.showerror("Layout", "First data row must be an integer.")
            return None

        if first_data_row < 1:
            if quiet:
                return None
            messagebox.showerror("Layout", "First data row must be 1 or greater.")
            return None

        header_lines = max(0, first_data_row - 1)

        separator_label = self._layout_field_vars["separator"].get()
        decimal_label = self._layout_field_vars["decimal"].get()
        separator_map = {"Tab": "\t", "Comma": ",", "Semicolon": ";", "Space": " "}
        decimal_map = {"Dot": ".", "Comma": ","}
        separator = separator_map.get(separator_label, "\t")
        decimal = decimal_map.get(decimal_label, ".")

        name = target_name or self.layout_config_var.get() or self.active_layout_name or "Custom"

        def parse_column(field_key, label, minimum):
            raw = self._layout_field_vars[field_key].get().strip()
            if not raw:
                if quiet:
                    return None
                messagebox.showerror("Layout", f"{label} column is required and must be a number.")
                return None
            try:
                value = int(raw)
            except ValueError:
                if quiet:
                    return None
                messagebox.showerror("Layout", f"{label} column must be an integer.")
                return None
            if value < minimum:
                if quiet:
                    return None
                messagebox.showerror(
                    "Layout",
                    f"{label} column must be {minimum} or greater.",
                )
                return None
            return value

        time_column = parse_column("time_column", "Time", 1)
        sbp_column = parse_column("sbp_column", "SBP", 2)
        dbp_column = parse_column("dbp_column", "DBP", 2)
        map_column = parse_column("map_column", "MAP", 2)

        comment_raw = self._layout_field_vars["comment_column"].get().strip()
        comment_column = None
        if comment_raw:
            try:
                comment_column = int(comment_raw)
            except ValueError:
                if quiet:
                    return None
                messagebox.showerror("Layout", "Comment column must be an integer or left blank.")
                return None
            if comment_column < 2:
                if quiet:
                    return None
                messagebox.showerror("Layout", "Comment column must be 2 or greater.")
                return None

        if None in (time_column, sbp_column, dbp_column, map_column):
            return None

        return LayoutConfig(
            name=name,
            header_lines=header_lines,
            first_data_row=first_data_row,
            separator=separator,
            decimal=decimal,
            time_column=time_column,
            sbp_column=sbp_column,
            dbp_column=dbp_column,
            map_column=map_column,
            comment_column=comment_column,
        )

    def _apply_layout_config_to_fields(self, layout):
        if layout is None:
            return
        if not self._layout_field_vars:
            return

        self._layout_field_vars["first_data_row"].set(str(layout.first_data_row))
        self._layout_field_vars["separator"].set(self._separator_display(layout.separator))
        self._layout_field_vars["decimal"].set(self._decimal_display(layout.decimal))
        self._layout_field_vars["time_column"].set(str(layout.time_column))
        self._layout_field_vars["sbp_column"].set(str(layout.sbp_column))
        self._layout_field_vars["dbp_column"].set(str(layout.dbp_column))
        self._layout_field_vars["map_column"].set(str(layout.map_column))
        if getattr(layout, "comment_column", None) is None:
            self._layout_field_vars["comment_column"].set("")
        else:
            self._layout_field_vars["comment_column"].set(str(layout.comment_column))

    def _on_layout_config_selected(self, *_event):
        selected = self.layout_config_var.get()
        if selected not in self.layout_configs:
            return
        self._set_active_layout(selected)
        self._apply_layout_config_to_fields(self.layout_configs[selected])
        self._update_layout_preview()

    def _save_layout_from_fields(self):
        target = self.layout_config_var.get() or self.active_layout_name
        if not target:
            messagebox.showerror("Layout", "Select or name a layout before saving.")
            return
        cfg = self._collect_layout_from_fields(target_name=target)
        if cfg is None:
            return
        self.layout_configs[cfg.name] = cfg
        if cfg.name not in self.layout_config_order:
            self.layout_config_order.append(cfg.name)
        self._set_active_layout(cfg.name, persist=False)
        self._persist_layout_configs()
        if hasattr(self, "status"):
            self.status.set(f"Saved layout '{cfg.name}'.")
        self._update_layout_combo_values()

    def _save_layout_as(self):
        cfg = self._collect_layout_from_fields(quiet=False)
        if cfg is None:
            return
        name = simpledialog.askstring("Save layout", "Enter a name for this layout:", parent=self._layout_window)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self.layout_configs and not messagebox.askyesno(
            "Overwrite layout",
            f"A layout named '{name}' already exists. Overwrite it?",
            parent=self._layout_window,
        ):
            return
        cfg.name = name
        self.layout_configs[name] = cfg
        if name not in self.layout_config_order:
            self.layout_config_order.append(name)
        self._set_active_layout(name, persist=False)
        self._persist_layout_configs()
        if hasattr(self, "status"):
            self.status.set(f"Saved new layout '{name}'.")
        self._update_layout_combo_values()

    def _delete_layout(self):
        target = self.layout_config_var.get()
        if not target or target not in self.layout_configs:
            return
        if len(self.layout_config_order) <= 1:
            messagebox.showinfo("Layout", "At least one layout must remain.")
            return
        if not messagebox.askyesno("Delete layout", f"Delete layout '{target}'?", parent=self._layout_window):
            return
        self.layout_configs.pop(target, None)
        if target in self.layout_config_order:
            self.layout_config_order.remove(target)
        fallback = self.layout_config_order[0]
        self._set_active_layout(fallback, persist=False)
        self._persist_layout_configs()
        self._update_layout_combo_values()
        if hasattr(self, "status"):
            self.status.set(f"Deleted layout '{target}'.")

    def _update_layout_combo_values(self):
        if not hasattr(self, "_layout_config_combo") or self._layout_config_combo is None:
            return
        names = self._layout_names()
        self._layout_config_combo["values"] = names
        if self.active_layout_name in names:
            self._layout_config_combo.set(self.active_layout_name)
        self._update_layout_preview()

    # --------------------------- Dialog windows --------------------------
    def open_preferences(self):
        if self._preferences_window is not None and self._preferences_window.winfo_exists():
            self._preferences_window.lift()
            return

        win = tk.Toplevel(self)
        win.title("Preferences")
        win.transient(self)
        win.resizable(False, False)
        self._preferences_window = win
        self._preferences_snapshot = self._export_preferences()
        win.protocol("WM_DELETE_WINDOW", lambda: self._cancel_preferences(win))

        content = ttk.Frame(win, padding=16)
        content.pack(fill=tk.BOTH, expand=True)
        self._filter_method_frames = {}
        self._filter_method_inputs = []
        self._filter_method_combo = None

        # Time window
        time_frame = ttk.LabelFrame(content, text="Time window")
        time_frame.pack(fill=tk.X, expand=True, pady=(0, 12))
        time_frame.columnconfigure(1, weight=1)

        ttk.Label(
            time_frame,
            text="Choose how the analysis window boundaries should be resolved.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(time_frame, text="Start anchor").grid(row=1, column=0, sticky="w")
        start_mode_combo = ttk.Combobox(
            time_frame,
            textvariable=self.start_mode_var,
            values=["Time", "Epoch", "Comment", "None"],
            state="readonly",
            width=14,
        )
        start_mode_combo.grid(row=1, column=1, sticky="ew")

        start_fields = ttk.Frame(time_frame)
        start_fields.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        start_fields.columnconfigure(0, weight=1)

        start_time_frame = ttk.Frame(start_fields)
        ttk.Label(start_time_frame, text="Time (s or mm:ss)").grid(row=0, column=0, sticky="w")
        ttk.Entry(start_time_frame, textvariable=self.start_var, width=14).grid(row=0, column=1, padx=(8, 0))

        start_epoch_frame = ttk.Frame(start_fields)
        ttk.Label(start_epoch_frame, text="Epoch # (1-based)").grid(row=0, column=0, sticky="w")
        ttk.Entry(start_epoch_frame, textvariable=self.start_epoch_var, width=14).grid(row=0, column=1, padx=(8, 0))

        start_comment_frame = ttk.Frame(start_fields)
        start_comment_frame.columnconfigure(1, weight=1)
        ttk.Label(start_comment_frame, text="Comment contains").grid(row=0, column=0, sticky="w")
        ttk.Entry(start_comment_frame, textvariable=self.start_comment_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(start_comment_frame, text="Offset (s or mm:ss)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(start_comment_frame, textvariable=self.start_offset_var, width=14).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self._start_mode_frames = {
            "Time": start_time_frame,
            "Epoch": start_epoch_frame,
            "Comment": start_comment_frame,
            "None": ttk.Frame(start_fields),
        }
        # Placeholder for "None" mode
        self._start_mode_frames["None"].grid_columnconfigure(0, weight=1)

        ttk.Label(time_frame, text="End anchor").grid(row=3, column=0, sticky="w")
        end_mode_combo = ttk.Combobox(
            time_frame,
            textvariable=self.end_mode_var,
            values=["Time", "Epoch", "Comment", "None"],
            state="readonly",
            width=14,
        )
        end_mode_combo.grid(row=3, column=1, sticky="ew")

        end_fields = ttk.Frame(time_frame)
        end_fields.grid(row=4, column=0, columnspan=2, sticky="ew")
        end_fields.columnconfigure(0, weight=1)

        end_time_frame = ttk.Frame(end_fields)
        ttk.Label(end_time_frame, text="Time (s or mm:ss)").grid(row=0, column=0, sticky="w")
        ttk.Entry(end_time_frame, textvariable=self.end_var, width=14).grid(row=0, column=1, padx=(8, 0))

        end_epoch_frame = ttk.Frame(end_fields)
        ttk.Label(end_epoch_frame, text="Epoch # (1-based)").grid(row=0, column=0, sticky="w")
        ttk.Entry(end_epoch_frame, textvariable=self.end_epoch_var, width=14).grid(row=0, column=1, padx=(8, 0))

        end_comment_frame = ttk.Frame(end_fields)
        end_comment_frame.columnconfigure(1, weight=1)
        ttk.Label(end_comment_frame, text="Comment contains").grid(row=0, column=0, sticky="w")
        ttk.Entry(end_comment_frame, textvariable=self.end_comment_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(end_comment_frame, text="Offset (s or mm:ss)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(end_comment_frame, textvariable=self.end_offset_var, width=14).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        self._end_mode_frames = {
            "Time": end_time_frame,
            "Epoch": end_epoch_frame,
            "Comment": end_comment_frame,
            "None": ttk.Frame(end_fields),
        }
        self._end_mode_frames["None"].grid_columnconfigure(0, weight=1)

        ttk.Label(
            time_frame,
            text="Comment offsets accept positive seconds (after) and negative seconds (before) the comment, in s or mm:ss format.",
            foreground="gray",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))

        def _on_start_mode_change(*_args):
            self._update_window_mode_frames()

        def _on_end_mode_change(*_args):
            self._update_window_mode_frames()

        start_mode_combo.bind("<<ComboboxSelected>>", _on_start_mode_change)
        end_mode_combo.bind("<<ComboboxSelected>>", _on_end_mode_change)

        self._update_window_mode_frames()

        # Artifact filtering
        filter_frame = ttk.LabelFrame(content, text="Artifact filtering")
        filter_frame.pack(fill=tk.X, expand=True, pady=(0, 12))
        filter_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            filter_frame,
            text="Apply artifact filtering",
            variable=self.filter_on,
            command=self._update_filter_controls_state,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(filter_frame, text="Filter method").grid(row=1, column=0, sticky="w", pady=(8, 0))
        method_labels = [FILTER_METHODS[code]["label"] for code in FILTER_METHOD_ORDER]
        method_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_method,
            values=method_labels,
            state="readonly",
            width=26,
        )
        method_combo.grid(row=1, column=1, padx=(12, 0), pady=(8, 0), sticky="w")
        method_combo.bind("<<ComboboxSelected>>", self._update_filter_setting_frames)
        self._filter_method_combo = method_combo

        moving_frame = ttk.Frame(filter_frame)
        moving_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(moving_frame, text="Window seconds").grid(row=0, column=0, sticky="w")
        window_entry = ttk.Entry(moving_frame, textvariable=self.window_s_var, width=8)
        window_entry.grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(moving_frame, text="k_MAD").grid(row=1, column=0, sticky="w", pady=(6, 0))
        k_mad_entry = ttk.Entry(moving_frame, textvariable=self.k_mad_var, width=8)
        k_mad_entry.grid(row=1, column=1, sticky="w", padx=(12, 0), pady=(6, 0))

        delta_frame = ttk.Frame(filter_frame)
        delta_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(delta_frame, text="Max delta").grid(row=0, column=0, sticky="w")
        delta_entry = ttk.Entry(delta_frame, textvariable=self.delta_cutoff_var, width=8)
        delta_entry.grid(row=0, column=1, sticky="w", padx=(12, 0))

        std_frame = ttk.Frame(filter_frame)
        std_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(std_frame, text="k_SD").grid(row=0, column=0, sticky="w")
        k_sd_entry = ttk.Entry(std_frame, textvariable=self.k_sd_var, width=8)
        k_sd_entry.grid(row=0, column=1, sticky="w", padx=(12, 0))

        self._filter_method_frames = {
            "moving_average": moving_frame,
            "absolute_change": delta_frame,
            "std_from_mean": std_frame,
        }
        self._filter_method_inputs = [window_entry, k_mad_entry, delta_entry, k_sd_entry]
        for frame in self._filter_method_frames.values():
            frame.grid_remove()

        self._update_filter_setting_frames()
        self._update_filter_controls_state()

        # Output metrics
        metrics_frame = ttk.LabelFrame(content, text="Output metrics")
        metrics_frame.pack(fill=tk.X, expand=True, pady=(0, 12))
        for col_idx in range(3):
            metrics_frame.columnconfigure(col_idx, weight=1)
        ttk.Label(
            metrics_frame,
            text="Choose which variability metrics should be included in tables and exports.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(
            metrics_frame,
            text="Standard deviation",
            variable=self.output_std,
            command=self._on_metric_preference_changed,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            metrics_frame,
            text="Coefficient of Variation",
            variable=self.output_cov,
            command=self._on_metric_preference_changed,
        ).grid(row=1, column=1, sticky="w", pady=(8, 0), padx=(16, 0))
        ttk.Checkbutton(
            metrics_frame,
            text="Average real variability",
            variable=self.output_arv,
            command=self._on_metric_preference_changed,
        ).grid(row=1, column=2, sticky="w", pady=(8, 0), padx=(16, 0))

        buttons = ttk.Frame(content)
        buttons.pack(fill=tk.X, expand=True)
        ttk.Button(buttons, text="Cancel", command=lambda: self._cancel_preferences(win)).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(buttons, text="Save", command=lambda: self._save_preferences(win)).pack(side=tk.RIGHT)

        win.grab_set()

    def _update_window_mode_frames(self):
        start_mode = (self.start_mode_var.get() or "Time").title()
        if self._start_mode_frames:
            for frame in self._start_mode_frames.values():
                frame.grid_forget()
            active = self._start_mode_frames.get(start_mode) or self._start_mode_frames.get("Time")
            if active is not None:
                active.grid(row=0, column=0, sticky="ew")

        end_mode = (self.end_mode_var.get() or "Time").title()
        if self._end_mode_frames:
            for frame in self._end_mode_frames.values():
                frame.grid_forget()
            active_end = self._end_mode_frames.get(end_mode) or self._end_mode_frames.get("Time")
            if active_end is not None:
                active_end.grid(row=0, column=0, sticky="ew")

    def _result_display_columns(self):
        columns = [
            "file", "interval_s", "start_s", "end_s",
            "n_SBP_raw", "n_SBP_kept", "Mean_SBP",
        ]
        if self.output_std.get():
            columns.append("SD_SBP")
        if self.output_cov.get():
            columns.append("CV_SBP_pct")
        if self.output_arv.get():
            columns.append("ARV_SBP")

        columns.extend(["n_DBP_raw", "n_DBP_kept", "Mean_DBP"])
        if self.output_std.get():
            columns.append("SD_DBP")
        if self.output_cov.get():
            columns.append("CV_DBP_pct")
        if self.output_arv.get():
            columns.append("ARV_DBP")

        columns.extend(["n_MAP_raw", "n_MAP_kept", "Mean_MAP"])
        if self.output_std.get():
            columns.append("SD_MAP")
        if self.output_cov.get():
            columns.append("CV_MAP_pct")
        if self.output_arv.get():
            columns.append("ARV_MAP")

        return columns

    def _csv_headers(self):
        headers = [
            "file", "interval_s", "start_s", "end_s",
            "n_SBP_raw", "n_SBP_kept", "Mean_SBP",
        ]
        if self.output_std.get():
            headers.append("SD_SBP")
        if self.output_cov.get():
            headers.append("CV_SBP_pct")
        if self.output_arv.get():
            headers.append("ARV_SBP")

        headers.extend(["n_DBP_raw", "n_DBP_kept", "Mean_DBP"])
        if self.output_std.get():
            headers.append("SD_DBP")
        if self.output_cov.get():
            headers.append("CV_DBP_pct")
        if self.output_arv.get():
            headers.append("ARV_DBP")

        headers.extend(["n_MAP_raw", "n_MAP_kept", "Mean_MAP"])
        if self.output_std.get():
            headers.append("SD_MAP")
        if self.output_cov.get():
            headers.append("CV_MAP_pct")
        if self.output_arv.get():
            headers.append("ARV_MAP")

        headers.append("error")
        return headers

    def _apply_result_column_preferences(self):
        if getattr(self, "tree", None) is None:
            return
        display_columns = self._result_display_columns()
        try:
            self.tree.configure(displaycolumns=display_columns)
        except tk.TclError:
            return

    def _on_metric_preference_changed(self, *_args):
        self._apply_result_column_preferences()
        self._autofit_tree_columns(self.tree, min_width=90, padding=28, max_width=None)

    def _update_filter_setting_frames(self, *_args):
        if not self._filter_method_frames:
            return
        method = normalize_filter_method(self.filter_method.get())
        for frame in self._filter_method_frames.values():
            frame.grid_remove()
        selected = self._filter_method_frames.get(method)
        if selected is not None:
            selected.grid()

    def _update_filter_controls_state(self, *_args):
        enabled = bool(self.filter_on.get())
        if self._filter_method_combo is not None:
            try:
                self._filter_method_combo.configure(state="readonly" if enabled else "disabled")
            except tk.TclError:
                pass
        state = "normal" if enabled else "disabled"
        for widget in getattr(self, "_filter_method_inputs", []):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def _format_result_row(self, result: Dict[str, Any]):
        values = []
        error_text = result.get("error")
        for col in self._tree_columns:
            if col == "file":
                values.append(result.get("file", ""))
            elif error_text:
                values.append("ERROR")
            else:
                values.append(fmt_float(result.get(col), 6))
        return values

    def _gather_exclusion_records(self):
        records = []
        for res in self.results:
            if not isinstance(res, dict):
                continue
            if res.get("error"):
                continue
            excluded_map = res.get("excluded_values") or {}
            if not isinstance(excluded_map, dict):
                continue
            file_name = res.get("file", "")
            ordered_measures = ("SBP", "DBP", "MAP")
            for measure in ordered_measures:
                values = excluded_map.get(measure)
                if values:
                    records.append({
                        "file": file_name,
                        "measure": measure,
                        "values": list(values),
                    })
            for measure, values in excluded_map.items():
                if measure in ordered_measures:
                    continue
                if values:
                    records.append({
                        "file": file_name,
                        "measure": measure,
                        "values": list(values),
                    })
        return records

    def _refresh_exclusions_window(self):
        if self._exclusions_window is None or not self._exclusions_window.winfo_exists():
            self._exclusion_tree = None
            self._exclusion_records = []
            return
        tree = self._exclusion_tree
        if tree is None:
            return
        tree.delete(*tree.get_children())
        records = self._gather_exclusion_records()
        self._exclusion_records = records
        if not records:
            tree.insert("", tk.END, values=("No excluded values recorded.", "", ""))
        else:
            for rec in records:
                formatted = ", ".join(fmt_float(val, 6) for val in rec["values"])
                tree.insert("", tk.END, values=(rec["file"], rec["measure"], formatted))
        self._autofit_tree_columns(tree, min_width=120, padding=32, max_width=None)

    def _close_exclusions_window(self):
        win = self._exclusions_window
        if win is not None and win.winfo_exists():
            try:
                win.destroy()
            except tk.TclError:
                pass
        self._exclusions_window = None
        self._exclusion_tree = None
        self._exclusion_records = []

    def _export_exclusions_csv(self):
        records = self._gather_exclusion_records()
        value_rows = []
        for rec in records:
            for value in rec["values"]:
                value_rows.append((rec["file"], rec["measure"], fmt_float(value, 10)))
        if not value_rows:
            messagebox.showinfo("No exclusions", "No excluded values to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Save exclusions CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["file", "measurement", "excluded_value"])
                for file_name, measure, value in value_rows:
                    writer.writerow([file_name, measure, value])
            self.status.set(f"Saved exclusion CSV: {path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def view_exclusions(self):
        if self._exclusions_window is not None and self._exclusions_window.winfo_exists():
            self._exclusions_window.lift()
            self._refresh_exclusions_window()
            return

        win = tk.Toplevel(self)
        win.title("Excluded blood pressure values")
        win.transient(self)
        win.resizable(True, True)
        self._exclusions_window = win

        content = ttk.Frame(win, padding=12)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        columns = ("file", "measure", "values")
        tree = ttk.Treeview(content, columns=columns, show="headings", height=14)
        tree.heading("file", text="Participant")
        tree.heading("measure", text="Measurement")
        tree.heading("values", text="Excluded values")
        tree.column("file", anchor=tk.W, width=200, stretch=False)
        tree.column("measure", anchor=tk.W, width=140, stretch=False)
        tree.column("values", anchor=tk.W, stretch=True)
        tree.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(content, orient=tk.VERTICAL, command=tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=y_scroll.set)

        x_scroll = ttk.Scrollbar(content, orient=tk.HORIZONTAL, command=tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        tree.configure(xscrollcommand=x_scroll.set)

        btns = ttk.Frame(content)
        btns.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Close", command=self._close_exclusions_window).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Export CSV…", command=self._export_exclusions_csv).pack(side=tk.RIGHT, padx=(0, 8))

        self._exclusion_tree = tree
        win.protocol("WM_DELETE_WINDOW", self._close_exclusions_window)

        self._refresh_exclusions_window()

    def _export_preferences(self) -> Dict[str, Any]:
        return {
            "start_mode": self.start_mode_var.get(),
            "start_time": self.start_var.get(),
            "start_epoch": self.start_epoch_var.get(),
            "start_comment": self.start_comment_var.get(),
            "start_offset": self.start_offset_var.get(),
            "end_mode": self.end_mode_var.get(),
            "end_time": self.end_var.get(),
            "end_epoch": self.end_epoch_var.get(),
            "end_comment": self.end_comment_var.get(),
            "end_offset": self.end_offset_var.get(),
            "filter_on": bool(self.filter_on.get()),
            "filter_method": self.filter_method.get(),
            "k_sd": self.k_sd_var.get(),
            "window_s": self.window_s_var.get(),
            "k_mad": self.k_mad_var.get(),
            "delta_cutoff": self.delta_cutoff_var.get(),
            "show_std": bool(self.output_std.get()),
            "show_cov": bool(self.output_cov.get()),
            "show_arv": bool(self.output_arv.get()),
        }

    def _restore_preferences(self, snapshot: Optional[Dict[str, Any]]):
        if not snapshot:
            return
        self.start_mode_var.set(snapshot.get("start_mode", "Time"))
        self.start_var.set(snapshot.get("start_time", ""))
        self.start_epoch_var.set(snapshot.get("start_epoch", ""))
        self.start_comment_var.set(snapshot.get("start_comment", ""))
        self.start_offset_var.set(snapshot.get("start_offset", "0"))
        self.end_mode_var.set(snapshot.get("end_mode", "Time"))
        self.end_var.set(snapshot.get("end_time", ""))
        self.end_epoch_var.set(snapshot.get("end_epoch", ""))
        self.end_comment_var.set(snapshot.get("end_comment", ""))
        self.end_offset_var.set(snapshot.get("end_offset", "0"))
        self.filter_on.set(bool(snapshot.get("filter_on", False)))
        saved_method = snapshot.get("filter_method", FILTER_METHODS["moving_average"]["label"])
        self.filter_method.set(filter_label_from_value(saved_method))
        self.k_sd_var.set(snapshot.get("k_sd", "3.0"))
        self.window_s_var.set(snapshot.get("window_s", "15"))
        self.k_mad_var.set(snapshot.get("k_mad", "4.0"))
        delta_value = snapshot.get("delta_cutoff", "20")
        if delta_value is None:
            delta_text = "20"
        elif delta_value == "":
            delta_text = ""
        else:
            delta_text = str(delta_value)
        self.delta_cutoff_var.set(delta_text)
        self.output_std.set(bool(snapshot.get("show_std", True)))
        self.output_cov.set(bool(snapshot.get("show_cov", True)))
        self.output_arv.set(bool(snapshot.get("show_arv", True)))
        self._update_window_mode_frames()
        self._update_filter_setting_frames()
        self._update_filter_controls_state()
        self._apply_result_column_preferences()

    def _parse_offset_seconds(self, value: str) -> float:
        text = (value or "").strip()
        if not text:
            return 0.0
        sign = 1.0
        if text[0] in "+-":
            if text[0] == "-":
                sign = -1.0
            text = text[1:].strip()
        if not text:
            return 0.0
        if text.lower().endswith("s"):
            text = text[:-1].strip()
        if not text:
            return 0.0
        if ":" in text:
            seconds = parse_time_any(text)
        else:
            seconds = float(text)
        return sign * float(seconds)

    def _parse_window_endpoint(
        self,
        label: str,
        mode_var: tk.StringVar,
        time_var: tk.StringVar,
        epoch_var: tk.StringVar,
        comment_var: tk.StringVar,
        offset_var: tk.StringVar,
    ) -> Dict[str, Any]:
        mode = (mode_var.get() or "None").strip().lower()
        if mode == "time":
            raw = (time_var.get() or "").strip()
            if not raw:
                return {"mode": "time", "value": None}
            try:
                value = parse_time_any(raw)
            except Exception as exc:
                raise ValueError(f"{label} time: {exc}") from exc
            return {"mode": "time", "value": value}
        if mode == "epoch":
            raw_epoch = (epoch_var.get() or "").strip()
            if not raw_epoch:
                return {"mode": "epoch", "value": None}
            try:
                epoch_val = int(raw_epoch)
            except ValueError as exc:
                raise ValueError(f"{label} epoch must be an integer.") from exc
            if epoch_val < 1:
                raise ValueError(f"{label} epoch must be 1 or greater.")
            return {"mode": "epoch", "value": epoch_val}
        if mode == "comment":
            comment_text = (comment_var.get() or "").strip()
            if not comment_text:
                raise ValueError(f"{label} comment text is required to anchor the window.")
            offset_raw = (offset_var.get() or "").strip()
            try:
                offset_val = self._parse_offset_seconds(offset_raw)
            except Exception as exc:
                raise ValueError(f"{label} comment offset is invalid: {exc}") from exc
            return {"mode": "comment", "text": comment_text, "offset": offset_val}
        return {"mode": "none"}

    def _build_window_spec(self) -> Dict[str, Dict[str, Any]]:
        start_spec = self._parse_window_endpoint(
            "Start",
            self.start_mode_var,
            self.start_var,
            self.start_epoch_var,
            self.start_comment_var,
            self.start_offset_var,
        )
        end_spec = self._parse_window_endpoint(
            "End",
            self.end_mode_var,
            self.end_var,
            self.end_epoch_var,
            self.end_comment_var,
            self.end_offset_var,
        )
        return {"start": start_spec, "end": end_spec}

    def _save_preferences(self, window):
        self._preferences_snapshot = None
        self._close_preferences(window)

    def _cancel_preferences(self, window):
        if self._preferences_snapshot is not None:
            self._restore_preferences(self._preferences_snapshot)
        self._close_preferences(window)
    def open_layout_dialog(self):
        if self._layout_window is not None and self._layout_window.winfo_exists():
            self._layout_window.lift()
            return

        for var, trace in self._layout_field_traces:
            try:
                var.trace_remove("write", trace)
            except Exception:
                pass
        self._layout_field_traces = []
        self._layout_field_vars = {}

        win = tk.Toplevel(self)
        win.title("Format file layout")
        win.geometry("960x720")
        win.minsize(820, 620)
        win.transient(self)
        self._layout_window = win
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_layout(win))

        container = ttk.Frame(win, padding=16)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=0)
        container.rowconfigure(2, weight=1)
        container.rowconfigure(3, weight=0)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        self._layout_file_label_var = tk.StringVar()
        ttk.Label(header, textvariable=self._layout_file_label_var, font=("TkDefaultFont", 11, "bold"))\
            .grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 6))

        ttk.Label(header, text="Config:").grid(row=1, column=0, sticky="w")
        self.layout_config_var.set(self.active_layout_name)
        config_combo = ttk.Combobox(
            header,
            textvariable=self.layout_config_var,
            values=self._layout_names(),
            state="readonly",
            width=24,
        )
        config_combo.grid(row=1, column=1, sticky="w")
        config_combo.bind("<<ComboboxSelected>>", self._on_layout_config_selected)
        self._layout_config_combo = config_combo

        ttk.Button(header, text="Save", command=self._save_layout_from_fields).grid(row=1, column=2, padx=(8, 0))
        ttk.Button(header, text="Save as…", command=self._save_layout_as).grid(row=1, column=3, padx=(8, 0))
        ttk.Button(header, text="Delete", command=self._delete_layout).grid(row=1, column=4, padx=(8, 0))

        ttk.Label(header, text="Saved layouts let you switch quickly between file formats.", foreground="gray")\
            .grid(row=2, column=0, columnspan=5, sticky="w", pady=(6, 0))

        top_section = ttk.Frame(container)
        top_section.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        top_section.columnconfigure(0, weight=0)
        top_section.columnconfigure(1, weight=1)
        top_section.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(top_section)
        left_panel.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        left_panel.columnconfigure(0, weight=1)

        # Prepare field variables
        field_names = [
            "first_data_row",
            "separator",
            "decimal",
            "time_column",
            "sbp_column",
            "dbp_column",
            "map_column",
            "comment_column",
        ]
        for name in field_names:
            self._layout_field_vars[name] = tk.StringVar()

        def _on_field_change(*_args):
            self._update_layout_preview()

        for name in field_names:
            var = self._layout_field_vars[name]
            trace_id = var.trace_add("write", _on_field_change)
            self._layout_field_traces.append((var, trace_id))

        data_frame = ttk.LabelFrame(left_panel, text="Data format")
        data_frame.grid(row=0, column=0, sticky="nsew")
        data_frame.columnconfigure(1, weight=1)

        spinbox_cls = ttk.Spinbox if hasattr(ttk, "Spinbox") else tk.Spinbox

        ttk.Label(data_frame, text="First data row").grid(row=0, column=0, sticky="w", pady=(0, 6))
        first_row_spin = spinbox_cls(
            data_frame, from_=1, to=99999, width=8, textvariable=self._layout_field_vars["first_data_row"], increment=1
        )
        first_row_spin.grid(row=0, column=1, sticky="w", pady=(0, 6))

        ttk.Label(data_frame, text="Separator").grid(row=1, column=0, sticky="w", pady=(0, 6))
        ttk.Combobox(
            data_frame,
            textvariable=self._layout_field_vars["separator"],
            values=["Tab", "Comma", "Semicolon", "Space"],
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w", pady=(0, 6))

        ttk.Label(data_frame, text="Decimal symbol").grid(row=2, column=0, sticky="w", pady=(0, 6))
        ttk.Combobox(
            data_frame,
            textvariable=self._layout_field_vars["decimal"],
            values=["Dot", "Comma"],
            state="readonly",
            width=12,
        ).grid(row=2, column=1, sticky="w", pady=(0, 6))

        ttk.Label(left_panel, text="Configure only the fields that affect import for your files.", foreground="gray")\
            .grid(row=1, column=0, sticky="w", pady=(12, 0))

        right_panel = ttk.Frame(top_section)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)

        mapping_frame = ttk.LabelFrame(right_panel, text="Column mapping")
        mapping_frame.grid(row=0, column=0, sticky="ew")
        mapping_frame.columnconfigure(1, weight=1)

        ttk.Label(mapping_frame, text="Time column").grid(row=0, column=0, sticky="w", pady=(0, 6))
        time_spin = spinbox_cls(
            mapping_frame,
            from_=1,
            to=9999,
            width=10,
            textvariable=self._layout_field_vars["time_column"],
            increment=1,
        )
        time_spin.grid(row=0, column=1, sticky="w", pady=(0, 6))

        ttk.Label(mapping_frame, text="SBP column").grid(row=1, column=0, sticky="w", pady=(0, 6))
        sbp_spin = spinbox_cls(
            mapping_frame,
            from_=2,
            to=9999,
            width=10,
            textvariable=self._layout_field_vars["sbp_column"],
            increment=1,
        )
        sbp_spin.grid(row=1, column=1, sticky="w", pady=(0, 6))

        ttk.Label(mapping_frame, text="DBP column").grid(row=2, column=0, sticky="w", pady=(0, 6))
        dbp_spin = spinbox_cls(
            mapping_frame,
            from_=2,
            to=9999,
            width=10,
            textvariable=self._layout_field_vars["dbp_column"],
            increment=1,
        )
        dbp_spin.grid(row=2, column=1, sticky="w", pady=(0, 6))

        ttk.Label(mapping_frame, text="MAP column").grid(row=3, column=0, sticky="w", pady=(0, 6))
        map_spin = spinbox_cls(
            mapping_frame,
            from_=2,
            to=9999,
            width=10,
            textvariable=self._layout_field_vars["map_column"],
            increment=1,
        )
        map_spin.grid(row=3, column=1, sticky="w", pady=(0, 6))

        ttk.Label(mapping_frame, text="Comment column").grid(row=4, column=0, sticky="w", pady=(0, 6))
        comment_spin = spinbox_cls(
            mapping_frame,
            from_=2,
            to=9999,
            width=10,
            textvariable=self._layout_field_vars["comment_column"],
            increment=1,
        )
        comment_spin.grid(row=4, column=1, sticky="w", pady=(0, 6))

        ttk.Label(
            mapping_frame,
            text="Columns are 1-based. Use the preview below to verify how delimiters split your file.",
            foreground="gray",
            wraplength=520,
            justify=tk.LEFT,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        preview_frame = ttk.LabelFrame(container, text="Preview of data file")
        preview_frame.grid(row=2, column=0, sticky="nsew", pady=(16, 0))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        preview_container = ttk.Frame(preview_frame)
        preview_container.grid(row=0, column=0, sticky="nsew")
        preview_container.rowconfigure(0, weight=1)
        preview_container.columnconfigure(1, weight=1)

        style = self.style
        style.configure("Preview.RowNumbers.Treeview", background="#f3f4f6", fieldbackground="#f3f4f6")
        style.configure("Preview.RowNumbers.Treeview.Heading", background="#e5e7eb", foreground="#111827")

        row_tree = ttk.Treeview(
            preview_container,
            columns=("row",),
            show="headings",
            selectmode="none",
            style="Preview.RowNumbers.Treeview",
            height=18,
        )
        row_tree.heading("row", text="#")
        row_tree.column("row", width=60, anchor=tk.E, stretch=False)
        row_tree.grid(row=0, column=0, sticky="ns")
        row_tree.tag_configure("header", background="#eef2ff")
        row_tree.tag_configure("data_start", background="#e6ffef")

        preview_tree = ttk.Treeview(preview_container, show="headings", height=18)
        preview_tree.grid(row=0, column=1, sticky="nsew")

        preview_y_scroll = ttk.Scrollbar(preview_container, orient=tk.VERTICAL, command=self._scroll_preview_y)
        preview_y_scroll.grid(row=0, column=2, sticky="ns")
        preview_x_scroll = ttk.Scrollbar(preview_container, orient=tk.HORIZONTAL, command=preview_tree.xview)
        preview_x_scroll.grid(row=1, column=1, sticky="ew")
        preview_tree.configure(
            yscrollcommand=lambda f, l: self._on_preview_y_scroll("data", f, l),
            xscrollcommand=preview_x_scroll.set,
        )
        row_tree.configure(yscrollcommand=lambda f, l: self._on_preview_y_scroll("row", f, l))
        preview_tree.tag_configure("header", background="#eef2ff")
        preview_tree.tag_configure("data_start", background="#e6ffef")
        self._layout_preview_tree = preview_tree
        self._layout_preview_row_tree = row_tree
        self._preview_y_scrollbar = preview_y_scroll

        buttons = ttk.Frame(container)
        buttons.grid(row=3, column=0, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="Close", command=lambda: self._close_layout(win)).pack(side=tk.RIGHT)

        self._apply_layout_config_to_fields(self._get_active_layout())
        self._update_layout_combo_values()
        self._update_layout_preview()

    def _close_preferences(self, window):
        try:
            exists = window.winfo_exists()
        except Exception:
            exists = False
        if exists:
            window.destroy()
        self._preferences_window = None
        self._preferences_snapshot = None
        self._start_mode_frames = {}
        self._end_mode_frames = {}

    def _close_layout(self, window):
        if window.winfo_exists():
            window.destroy()
        self._layout_window = None
        self._layout_preview_tree = None
        self._layout_preview_row_tree = None
        self._preview_y_scrollbar = None
        self._preview_scroll_syncing = False
        for var, trace in self._layout_field_traces:
            try:
                var.trace_remove("write", trace)
            except Exception:
                pass
        self._layout_field_traces = []
        self._layout_field_vars = {}
        self._layout_file_label_var = None
        self._layout_config_combo = None

    def _update_layout_preview(self, *_event):
        if not self._layout_preview_tree:
            return
        if not (self._layout_window and self._layout_window.winfo_exists()):
            return

        tree = self._layout_preview_tree
        row_tree = self._layout_preview_row_tree
        tree.configure(columns=(), displaycolumns=())
        tree.delete(*tree.get_children())
        if row_tree is not None:
            row_tree.delete(*row_tree.get_children())

        if not self.filepaths:
            columns = ("message",)
            tree.configure(columns=columns, displaycolumns=columns)
            tree.heading("message", text="Preview")
            tree.column("message", anchor=tk.W, stretch=True)
            tree.insert("", tk.END, values=("No files loaded. Add files to preview their layout.",))
            if self._layout_file_label_var is not None:
                self._layout_file_label_var.set("No file selected.")
            self._autofit_tree_columns(tree, min_width=200, padding=24, max_width=None)
            if row_tree is not None:
                self._autofit_tree_columns(row_tree, min_width=40, padding=18, max_width=80)
            return

        selection = self.files_list.curselection()
        index = selection[0] if selection else 0
        index = max(0, min(len(self.filepaths) - 1, index))
        path = self.filepaths[index]

        if self._layout_file_label_var is not None:
            self._layout_file_label_var.set(f"Filename: {os.path.basename(path)}")

        layout_cfg = self._collect_layout_from_fields(quiet=True) or self._get_active_layout()
        header_lines = getattr(layout_cfg, "header_lines", 0) or 0
        first_data_row = getattr(layout_cfg, "first_data_row", 0) or 0
        separator = getattr(layout_cfg, "separator", "\t") or "\t"

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                snippet = fh.readlines()[:200]
        except Exception as exc:
            columns = ("message",)
            tree.configure(columns=columns, displaycolumns=columns)
            tree.heading("message", text="Preview")
            tree.column("message", anchor=tk.W, stretch=True)
            tree.insert(
                "",
                tk.END,
                values=(f"Could not load preview for {os.path.basename(path)}: {exc}",),
            )
            self._autofit_tree_columns(tree, min_width=200, padding=24, max_width=None)
            if row_tree is not None:
                self._autofit_tree_columns(row_tree, min_width=40, padding=18, max_width=80)
            return

        if not snippet:
            columns = ("message",)
            tree.configure(columns=columns, displaycolumns=columns)
            tree.heading("message", text="Preview")
            tree.column("message", anchor=tk.W, stretch=True)
            tree.insert("", tk.END, values=(f"{os.path.basename(path)} is empty.",))
            self._autofit_tree_columns(tree, min_width=200, padding=24, max_width=None)
            if row_tree is not None:
                self._autofit_tree_columns(row_tree, min_width=40, padding=18, max_width=80)
            return

        try:
            parsed_rows = []
            max_cols = 0
            for lineno, raw in enumerate(snippet, start=1):
                text = raw.rstrip("\r\n")
                if separator == " ":
                    cells = text.split()
                else:
                    cells = _split_preserving_trailing(text, separator)
                parsed_rows.append((lineno, cells))
                if len(cells) > max_cols:
                    max_cols = len(cells)
        except Exception as exc:
            columns = ("message",)
            tree.configure(columns=columns, displaycolumns=columns)
            tree.heading("message", text="Preview")
            tree.column("message", anchor=tk.W, stretch=True)
            tree.insert(
                "",
                tk.END,
                values=(f"Could not parse preview for {os.path.basename(path)}: {exc}",),
            )
            self._autofit_tree_columns(tree, min_width=200, padding=24, max_width=None)
            if row_tree is not None:
                self._autofit_tree_columns(row_tree, min_width=40, padding=18, max_width=80)
            return

        if max_cols == 0:
            max_cols = 1

        columns = [f"col_{i}" for i in range(1, max_cols + 1)]
        tree.configure(columns=columns, displaycolumns=columns)

        for idx in range(1, max_cols + 1):
            col_id = f"col_{idx}"
            tree.heading(col_id, text=str(idx))
            tree.column(col_id, anchor=tk.W, stretch=False)

        for lineno, cells in parsed_rows:
            row_tags = []
            if header_lines and lineno <= header_lines:
                row_tags.append("header")
            if first_data_row and lineno == first_data_row:
                row_tags.append("data_start")

            padded_cells = [cell for cell in cells] + [""] * (max_cols - len(cells))
            if row_tree is not None:
                row_tree.insert("", tk.END, values=(str(lineno),), tags=row_tags)
            tree.insert("", tk.END, values=padded_cells, tags=row_tags)

        self._autofit_tree_columns(tree, min_width=60, padding=24, max_width=None)
        if row_tree is not None:
            self._autofit_tree_columns(row_tree, min_width=40, padding=18, max_width=80)

    def open_about(self):
        win = tk.Toplevel(self)
        win.title("About")
        win.transient(self)
        win.resizable(False, False)
        ttk.Label(win, text="Blood Pressure Variability Tool\n(About information coming soon)", padding=20).pack()
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 12))
        win.grab_set()

    def compute(self):
        if not self.filepaths:
            messagebox.showwarning("No files", "Please add one or more TXT files first.")
            return

        try:
            window_spec = self._build_window_spec()
        except ValueError as exc:
            messagebox.showerror("Time window", str(exc))
            return

        # parse filter params
        use_filter = bool(self.filter_on.get())
        method = normalize_filter_method(self.filter_method.get())
        try:
            k_sd = float(self.k_sd_var.get())
        except Exception:
            k_sd = 3.0
        try:
            window_s = float(self.window_s_var.get())
        except Exception:
            window_s = 15.0
        try:
            k_mad = float(self.k_mad_var.get())
        except Exception:
            k_mad = 4.0
        try:
            delta_cutoff = float(self.delta_cutoff_var.get())
        except Exception:
            delta_cutoff = 0.0

        # compute
        self.results = []
        # clear table
        for i in self.tree.get_children():
            self.tree.delete(i)

        self._apply_result_column_preferences()

        errors = 0
        for p in self.filepaths:
            try:
                res = compute_all_for_file(
                    p, window_spec,
                    use_filter=use_filter,
                    filter_method=method,
                    k_sd=k_sd,
                    window_s=window_s,
                    k_mad=k_mad,
                    delta_cutoff=delta_cutoff,
                    layout_config=self._get_active_layout(),
                )
                res["error"] = ""
                self.results.append(res)
                self.tree.insert("", tk.END, values=self._format_result_row(res))
            except Exception as e:
                errors += 1
                error_entry = {
                    "file": os.path.basename(p),
                    "error": str(e),
                }
                self.results.append(error_entry)
                self.tree.insert("", tk.END, values=self._format_result_row(error_entry))

        self._autofit_tree_columns(self.tree, min_width=90, padding=28, max_width=None)

        if errors:
            self.status.set(f"Done with {len(self.results)} result(s). {errors} file(s) had errors (see table).")
        else:
            self.status.set(f"Done. Computed metrics for {len(self.results)} file(s).")
        self._refresh_exclusions_window()

    def save_csv(self):
        if not self.results:
            messagebox.showinfo("No results", "Nothing to save. Click 'Compute' first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save results CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        headers = self._csv_headers()

        rows = []
        for r in self.results:
            rr = {k: r.get(k, "") for k in headers}
            rows.append(rr)

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            self.status.set(f"Saved CSV: {path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


if __name__ == "__main__":
    app = ARVApp()
    app.mainloop()

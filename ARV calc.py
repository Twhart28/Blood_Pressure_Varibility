#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import math
import tkinter as tk
from dataclasses import dataclass, asdict
from tkinter import ttk, filedialog, messagebox, simpledialog
from statistics import mean as _pymean

# ----------------------------- Core parsing & ARV -----------------------------

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
        l = ln.strip()

        if l.startswith("Interval="):
            # Often "Interval=\t1 s"
            parts = [p for p in l.split("\t") if p]
            try:
                val = parts[1] if len(parts) > 1 else parts[0].split("=", 1)[-1]
                interval_s = float(val.split()[0])
            except Exception:
                pass

        if l.startswith("ChannelTitle="):
            titles = l.split("=", 1)[-1].strip()
            channel_titles = [t.strip() for t in titles.split("\t") if t.strip()]

        if data_start_idx is None:
            row = ln.rstrip("\n")
            if separator == " ":
                parts = row.split()
            else:
                parts = row.split(separator)
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

    interval_s, channel_titles, data_start = parse_header(
        lines,
        separator=separator,
        data_start_override=data_start_override,
    )
    n_channels = len(channel_titles)
    series_map = {name: [] for name in channel_titles}

    # IMPORTANT: No artificial limit here — reads the ENTIRE dataset
    for ln in lines[data_start:]:
        row_txt = ln.rstrip("\n")
        if separator == " ":
            row = row_txt.split()
        else:
            row = row_txt.split(separator)
        if not row or not row[0].strip().isdigit():
            continue
        if len(row) < 1 + n_channels:
            continue

        vals = row[1:1 + n_channels]
        for name, val in zip(channel_titles, vals):
            series_map[name].append(safe_float(val, decimal=decimal))

    return interval_s, series_map, channel_titles


def select_window(series, interval_s, start_s, end_s):
    n = len(series)
    if n == 0:
        return []
    start_idx = 0 if start_s is None else max(0, int(start_s // interval_s))
    end_idx = n if end_s is None else min(n, int(math.ceil(end_s / interval_s)))
    if end_idx - start_idx <= 1:
        return []
    return series[start_idx:end_idx]


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
    """
    m = _nanmean(values)
    s = _nansd(values, ddof=1)
    if math.isnan(m) or math.isnan(s) or s == 0:
        return [v for v in values if not math.isnan(v)]
    lo, hi = m - k_sd*s, m + k_sd*s
    return [v for v in values if (not math.isnan(v)) and (lo <= v <= hi)]

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
        return [v for v in vals if not math.isnan(v)]

    w = max(3, int(round(window_s / interval_s)))  # window length in samples
    half = w // 2

    kept = []
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
            if abs(x - med) <= k_mad * sigma:
                kept.append(x)
    return kept


def compute_metrics(series, use_filter=False, filter_method="rolling", interval_s=1.0,
                    k_sd=3.0, window_s=15.0, k_mad=4.0):
    """
    Returns dict with:
      n_raw, n_kept, mean, sd, cv, arv
    """
    raw = [v for v in series if not math.isnan(v)]
    n_raw = len(raw)

    if not use_filter:
        kept = raw
    else:
        if filter_method == "global":
            kept = filter_outliers_global(raw, k_sd=float(k_sd))
        else:
            kept = filter_outliers_rolling_robust(raw, interval_s=interval_s,
                                                  window_s=float(window_s),
                                                  k_mad=float(k_mad))
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
        "arv": arv_val
    }


def compute_all_for_file(path, start_s, end_s, use_filter, filter_method,
                         k_sd, window_s, k_mad, layout_config=None):
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

    sbp = select_window(series_map[sbp_key], interval_s, start_s, end_s)
    dbp = select_window(series_map[dbp_key], interval_s, start_s, end_s)
    mapp = select_window(series_map[map_key], interval_s, start_s, end_s)

    ms_sbp = compute_metrics(
        sbp, use_filter, filter_method, interval_s, k_sd, window_s, k_mad
    )
    ms_dbp = compute_metrics(
        dbp, use_filter, filter_method, interval_s, k_sd, window_s, k_mad
    )
    ms_map = compute_metrics(
        mapp, use_filter, filter_method, interval_s, k_sd, window_s, k_mad
    )

    return {
        "file": os.path.basename(path),
        "interval_s": interval_s,
        "start_s": 0.0 if start_s is None else float(start_s),
        "end_s": None if end_s is None else float(end_s),

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
        self.filter_on = tk.BooleanVar(value=False)
        self.filter_method = tk.StringVar(value="rolling")
        self.k_sd_var = tk.StringVar(value="3.0")
        self.window_s_var = tk.StringVar(value="15")
        self.k_mad_var = tk.StringVar(value="4.0")

        self._preferences_window = None
        self._layout_window = None
        self._layout_preview_tree = None
        self._layout_file_label_var = None
        self._layout_field_vars = {}
        self._layout_field_traces = []
        self._layout_config_combo = None

        # Styling (optional nicer look)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
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

        # Results table
        cols = (
            "file", "interval_s", "start_s", "end_s",
            "n_SBP_raw", "n_SBP_kept", "Mean_SBP", "SD_SBP", "CV_SBP_pct", "ARV_SBP",
            "n_DBP_raw", "n_DBP_kept", "Mean_DBP", "SD_DBP", "CV_DBP_pct", "ARV_DBP",
            "n_MAP_raw", "n_MAP_kept", "Mean_MAP", "SD_MAP", "CV_MAP_pct", "ARV_MAP",
        )
        tree_container = ttk.Frame(self.right_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)
        tree_container.rowconfigure(0, weight=1)
        tree_container.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(tree_container, columns=cols, show="headings", height=18)
        for c in cols:
            self.tree.heading(c, text=c)
            base_w = 100
            if c == "file":
                w = 240
            elif "Mean" in c or "SD_" in c or "CV_" in c or "ARV_" in c:
                w = 110
            else:
                w = base_w
            self.tree.column(c, width=w, anchor=tk.CENTER)
        self.tree.grid(row=0, column=0, sticky="nsew")

        tree_y_scroll = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        tree_y_scroll.grid(row=0, column=1, sticky="ns")
        tree_x_scroll = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        tree_x_scroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=tree_y_scroll.set, xscrollcommand=tree_x_scroll.set)

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

        def parse_int(value, fallback):
            value = (value or "").strip()
            if not value:
                return fallback
            return int(value)

        try:
            header_lines = parse_int(self._layout_field_vars["header_lines"].get(), 0)
            first_data_row = parse_int(
                self._layout_field_vars["first_data_row"].get(),
                header_lines + 1 if header_lines else 1,
            )
        except ValueError:
            if quiet:
                return None
            messagebox.showerror("Layout", "Header lines and first data row must be integers.")
            return None

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
        )

    def _apply_layout_config_to_fields(self, layout):
        if layout is None:
            return
        if not self._layout_field_vars:
            return

        self._layout_field_vars["header_lines"].set(str(layout.header_lines))
        self._layout_field_vars["first_data_row"].set(str(layout.first_data_row))
        self._layout_field_vars["separator"].set(self._separator_display(layout.separator))
        self._layout_field_vars["decimal"].set(self._decimal_display(layout.decimal))
        self._layout_field_vars["time_column"].set(str(layout.time_column))
        self._layout_field_vars["sbp_column"].set(str(layout.sbp_column))
        self._layout_field_vars["dbp_column"].set(str(layout.dbp_column))
        self._layout_field_vars["map_column"].set(str(layout.map_column))

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
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_preferences(win))

        content = ttk.Frame(win, padding=16)
        content.pack(fill=tk.BOTH, expand=True)

        # Time window
        time_frame = ttk.LabelFrame(content, text="Time window")
        time_frame.pack(fill=tk.X, expand=True, pady=(0, 12))

        ttk.Label(time_frame, text="Start (s or mm:ss)").grid(row=0, column=0, sticky="w")
        ttk.Entry(time_frame, textvariable=self.start_var, width=12).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(time_frame, text="End (s or mm:ss)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(time_frame, textvariable=self.end_var, width=12).grid(row=1, column=1, padx=(8, 0), pady=(8, 0))

        # Calculations
        calc_frame = ttk.LabelFrame(content, text="Calculations")
        calc_frame.pack(fill=tk.X, expand=True, pady=(0, 12))

        ttk.Checkbutton(calc_frame, text="Apply artifact filtering", variable=self.filter_on).grid(row=0, column=0, sticky="w")

        ttk.Label(calc_frame, text="Filter method").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(calc_frame, textvariable=self.filter_method, values=["rolling", "global"], state="readonly", width=12).grid(row=1, column=1, padx=(12, 0), pady=(8, 0), sticky="w")

        ttk.Label(calc_frame, text="k_SD (global)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(calc_frame, textvariable=self.k_sd_var, width=8).grid(row=2, column=1, padx=(12, 0), pady=(8, 0), sticky="w")

        ttk.Label(calc_frame, text="Window seconds (rolling)").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(calc_frame, textvariable=self.window_s_var, width=8).grid(row=3, column=1, padx=(12, 0), pady=(8, 0), sticky="w")

        ttk.Label(calc_frame, text="k_MAD (rolling)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(calc_frame, textvariable=self.k_mad_var, width=8).grid(row=4, column=1, padx=(12, 0), pady=(8, 12), sticky="w")

        for child in calc_frame.winfo_children():
            child.grid_configure(padx=(0, 8))

        ttk.Button(content, text="Close", command=lambda: self._close_preferences(win)).pack(anchor="e")

        win.grab_set()

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
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
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

        left_panel = ttk.Frame(container)
        left_panel.grid(row=1, column=0, sticky="ns", padx=(0, 16))
        left_panel.columnconfigure(0, weight=1)

        # Prepare field variables
        field_names = [
            "header_lines",
            "first_data_row",
            "separator",
            "decimal",
            "time_column",
            "sbp_column",
            "dbp_column",
            "map_column",
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

        ttk.Label(data_frame, text="Header lines").grid(row=0, column=0, sticky="w", pady=(0, 6))
        header_spin = spinbox_cls(
            data_frame, from_=0, to=999, width=8, textvariable=self._layout_field_vars["header_lines"], increment=1
        )
        header_spin.grid(row=0, column=1, sticky="w", pady=(0, 6))

        ttk.Label(data_frame, text="First data row").grid(row=1, column=0, sticky="w", pady=(0, 6))
        first_row_spin = spinbox_cls(
            data_frame, from_=1, to=99999, width=8, textvariable=self._layout_field_vars["first_data_row"], increment=1
        )
        first_row_spin.grid(row=1, column=1, sticky="w", pady=(0, 6))

        ttk.Label(data_frame, text="Separator").grid(row=2, column=0, sticky="w", pady=(0, 6))
        ttk.Combobox(
            data_frame,
            textvariable=self._layout_field_vars["separator"],
            values=["Tab", "Comma", "Semicolon", "Space"],
            state="readonly",
            width=12,
        ).grid(row=2, column=1, sticky="w", pady=(0, 6))

        ttk.Label(data_frame, text="Decimal symbol").grid(row=3, column=0, sticky="w", pady=(0, 6))
        ttk.Combobox(
            data_frame,
            textvariable=self._layout_field_vars["decimal"],
            values=["Dot", "Comma"],
            state="readonly",
            width=12,
        ).grid(row=3, column=1, sticky="w", pady=(0, 6))

        ttk.Label(left_panel, text="Configure only the fields that affect import for your files.", foreground="gray")\
            .grid(row=1, column=0, sticky="w", pady=(12, 0))

        right_panel = ttk.Frame(container)
        right_panel.grid(row=1, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)

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

        ttk.Label(
            mapping_frame,
            text="Columns are 1-based. Use the preview below to verify how delimiters split your file.",
            foreground="gray",
            wraplength=360,
            justify=tk.LEFT,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        preview_frame = ttk.LabelFrame(right_panel, text="Preview of data file")
        preview_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        preview_container = ttk.Frame(preview_frame)
        preview_container.grid(row=0, column=0, sticky="nsew")
        preview_container.rowconfigure(0, weight=1)
        preview_container.columnconfigure(0, weight=1)

        preview_tree = ttk.Treeview(preview_container, show="headings")
        preview_tree.grid(row=0, column=0, sticky="nsew")

        preview_y_scroll = ttk.Scrollbar(preview_container, orient=tk.VERTICAL, command=preview_tree.yview)
        preview_y_scroll.grid(row=0, column=1, sticky="ns")
        preview_x_scroll = ttk.Scrollbar(preview_container, orient=tk.HORIZONTAL, command=preview_tree.xview)
        preview_x_scroll.grid(row=1, column=0, sticky="ew")
        preview_tree.configure(yscrollcommand=preview_y_scroll.set, xscrollcommand=preview_x_scroll.set)
        preview_tree.tag_configure("header", background="#eef2ff")
        preview_tree.tag_configure("data_start", background="#e6ffef")
        self._layout_preview_tree = preview_tree

        signal_frame = ttk.LabelFrame(right_panel, text="Signal preview")
        signal_frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        ttk.Label(
            signal_frame,
            text="Signal preview coming soon. Current settings affect numerical calculations only.",
            foreground="gray",
            wraplength=360,
            justify=tk.LEFT,
        ).pack(fill=tk.BOTH, expand=True, padx=8, pady=12)

        buttons = ttk.Frame(container)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="Close", command=lambda: self._close_layout(win)).pack(side=tk.RIGHT)

        self._apply_layout_config_to_fields(self._get_active_layout())
        self._update_layout_combo_values()
        self._update_layout_preview()

    def _close_preferences(self, window):
        if window.winfo_exists():
            window.destroy()
        self._preferences_window = None

    def _close_layout(self, window):
        if window.winfo_exists():
            window.destroy()
        self._layout_window = None
        self._layout_preview_tree = None
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
        for col in tree["columns"]:
            tree.heading(col, text="")
        tree.delete(*tree.get_children())

        if not self.filepaths:
            tree["columns"] = ("message",)
            tree.heading("message", text="Preview")
            tree.column("message", anchor=tk.W, width=400, stretch=True)
            tree.insert("", tk.END, values=("No files loaded. Add files to preview their layout.",))
            if self._layout_file_label_var is not None:
                self._layout_file_label_var.set("No file selected.")
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
            tree["columns"] = ("message",)
            tree.heading("message", text="Preview")
            tree.column("message", anchor=tk.W, width=400, stretch=True)
            tree.insert(
                "",
                tk.END,
                values=(f"Could not load preview for {os.path.basename(path)}: {exc}",),
            )
            return

        if not snippet:
            tree["columns"] = ("message",)
            tree.heading("message", text="Preview")
            tree.column("message", anchor=tk.W, width=400, stretch=True)
            tree.insert("", tk.END, values=(f"{os.path.basename(path)} is empty.",))
            return

        parsed_rows = []
        max_cols = 0
        for lineno, raw in enumerate(snippet, start=1):
            text = raw.rstrip("\n")
            if separator == " ":
                cells = text.split()
            else:
                cells = text.split(separator)
            parsed_rows.append((lineno, cells))
            if len(cells) > max_cols:
                max_cols = len(cells)

        columns = ["row"] + [f"col_{i}" for i in range(1, max_cols + 1)]
        tree["columns"] = columns

        tree.heading("row", text="Row")
        tree.column("row", width=60, anchor=tk.E, stretch=False)
        for idx in range(1, max_cols + 1):
            col_id = f"col_{idx}"
            tree.heading(col_id, text=f"Col {idx}")
            tree.column(col_id, width=120, anchor=tk.W, stretch=True)

        for lineno, cells in parsed_rows:
            values = [str(lineno)] + [cell for cell in cells] + [""] * (max_cols - len(cells))
            tags = []
            if header_lines and lineno <= header_lines:
                tags.append("header")
            if first_data_row and lineno == first_data_row:
                tags.append("data_start")
            tree.insert("", tk.END, values=values, tags=tags)

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
            start_s = parse_time_any(self.start_var.get())
            end_s = parse_time_any(self.end_var.get())
            if start_s is not None and end_s is not None and end_s <= start_s:
                messagebox.showwarning("Time window", "End time must be greater than start time.")
                return
        except Exception as e:
            messagebox.showerror("Invalid time", f"Could not parse time: {e}")
            return

        # parse filter params
        use_filter = bool(self.filter_on.get())
        method = self.filter_method.get()
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

        # compute
        self.results = []
        # clear table
        for i in self.tree.get_children():
            self.tree.delete(i)

        errors = 0
        for p in self.filepaths:
            try:
                res = compute_all_for_file(
                    p, start_s, end_s,
                    use_filter=use_filter,
                    filter_method=method,
                    k_sd=k_sd,
                    window_s=window_s,
                    k_mad=k_mad,
                    layout_config=self._get_active_layout(),
                )
                self.results.append(res)
                self.tree.insert("", tk.END, values=(
                    res["file"],
                    fmt_float(res["interval_s"]),
                    fmt_float(res["start_s"]),
                    fmt_float(res["end_s"]),

                    fmt_float(res["n_SBP_raw"], 6), fmt_float(res["n_SBP_kept"], 6),
                    fmt_float(res["Mean_SBP"], 6), fmt_float(res["SD_SBP"], 6),
                    fmt_float(res["CV_SBP_pct"], 6), fmt_float(res["ARV_SBP"], 6),

                    fmt_float(res["n_DBP_raw"], 6), fmt_float(res["n_DBP_kept"], 6),
                    fmt_float(res["Mean_DBP"], 6), fmt_float(res["SD_DBP"], 6),
                    fmt_float(res["CV_DBP_pct"], 6), fmt_float(res["ARV_DBP"], 6),

                    fmt_float(res["n_MAP_raw"], 6), fmt_float(res["n_MAP_kept"], 6),
                    fmt_float(res["Mean_MAP"], 6), fmt_float(res["SD_MAP"], 6),
                    fmt_float(res["CV_MAP_pct"], 6), fmt_float(res["ARV_MAP"], 6),
                ))
            except Exception as e:
                errors += 1
                self.tree.insert("", tk.END, values=(
                    os.path.basename(p), "", "", "",
                    "", "", "", "", "", "ERROR",
                    "", "", "", "", "", "ERROR",
                    "", "", "", "", "", "ERROR",
                ))
                self.results.append({
                    "file": os.path.basename(p),
                    "error": str(e),
                })

        if errors:
            self.status.set(f"Done with {len(self.results)} result(s). {errors} file(s) had errors (see table).")
        else:
            self.status.set(f"Done. Computed metrics for {len(self.results)} file(s).")

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

        headers = [
            "file", "interval_s", "start_s", "end_s",
            "n_SBP_raw", "n_SBP_kept", "Mean_SBP", "SD_SBP", "CV_SBP_pct", "ARV_SBP",
            "n_DBP_raw", "n_DBP_kept", "Mean_DBP", "SD_DBP", "CV_DBP_pct", "ARV_DBP",
            "n_MAP_raw", "n_MAP_kept", "Mean_MAP", "SD_MAP", "CV_MAP_pct", "ARV_MAP",
            "error"
        ]

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

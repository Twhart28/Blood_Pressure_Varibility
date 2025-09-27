#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from statistics import mean as _pymean

# ----------------------------- Core parsing & ARV -----------------------------

def parse_header(lines):
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
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip().isdigit():
                data_start_idx = i

    if interval_s is None:
        raise ValueError("Could not parse sampling Interval= from file header.")
    if channel_titles is None:
        raise ValueError("Could not parse ChannelTitle= from file header.")
    if data_start_idx is None:
        raise ValueError("Could not locate start of numeric data.")
    return interval_s, channel_titles, data_start_idx


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_series(filepath):
    """
    Reads the file, returns (interval_s, {channel_name: [values...]})
    Only channels listed in ChannelTitle= are returned.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    interval_s, channel_titles, data_start = parse_header(lines)
    n_channels = len(channel_titles)
    series_map = {name: [] for name in channel_titles}

    # IMPORTANT: No artificial limit here — reads the ENTIRE dataset
    for ln in lines[data_start:]:
        row = ln.rstrip("\n").split("\t")
        if not row or not row[0].strip().isdigit():
            continue
        if len(row) < 1 + n_channels:
            continue

        vals = row[1:1 + n_channels]
        for name, val in zip(channel_titles, vals):
            series_map[name].append(safe_float(val))

    return interval_s, series_map


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
                         k_sd, window_s, k_mad):
    interval_s, series_map = load_series(path)

    def resolve(name):
        # exact, case-insensitive, or contains
        if name in series_map:
            return name
        for k in series_map.keys():
            if k.lower() == name.lower():
                return k
        for k in series_map.keys():
            if name.lower() in k.lower():
                return k
        raise KeyError(f"Channel '{name}' not found. Available: {list(series_map.keys())}")

    sbp_key = resolve("reSYS")
    dbp_key = resolve("reDIA")
    map_key = resolve("reMAP")

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

# ---------------------------------- GUI ------------------------------------

class ARVApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Blood Pressure Variability Tool")
        self.geometry("1180x640")
        self.minsize(1024, 600)

        self.filepaths = []
        self.results = []

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
        self._layout_preview_box = None

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

        win = tk.Toplevel(self)
        win.title("Format file layout")
        win.geometry("600x560")
        win.transient(self)
        self._layout_window = win
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_layout(win))

        container = ttk.Frame(win, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        spec_frame = ttk.LabelFrame(container, text="Data specifications")
        spec_frame.pack(fill=tk.X, pady=(0, 12))

        labels = [
            ("Header lines", "9"),
            ("Data type", "RR"),
            ("Time units", "s"),
            ("Data units", "mmHg"),
            ("Column separator", "Tab"),
            ("Time index column", "None"),
        ]
        for i, (label, default) in enumerate(labels):
            ttk.Label(spec_frame, text=label).grid(row=i, column=0, sticky="w", pady=4, padx=4)
            entry = ttk.Entry(spec_frame, width=24)
            entry.grid(row=i, column=1, sticky="w", pady=4, padx=4)
            entry.insert(0, default)

        update_frame = ttk.LabelFrame(container, text="Update/add additional columns")
        update_frame.pack(fill=tk.X, pady=(0, 12))

        fields = ["Date", "Time", "Age/Gender", "Weight", "Height"]
        for i, name in enumerate(fields):
            ttk.Label(update_frame, text=f"{name}:").grid(row=i, column=0, sticky="w", pady=4, padx=4)
            ttk.Entry(update_frame, width=24).grid(row=i, column=1, sticky="w", pady=4, padx=4)

        preview_frame = ttk.LabelFrame(container, text="Preview of data file")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        preview_container = ttk.Frame(preview_frame)
        preview_container.pack(fill=tk.BOTH, expand=True)
        preview_container.rowconfigure(0, weight=1)
        preview_container.columnconfigure(0, weight=1)

        preview_box = tk.Text(preview_container, height=10, wrap="none")
        preview_box.grid(row=0, column=0, sticky="nsew")

        preview_y_scroll = ttk.Scrollbar(preview_container, orient=tk.VERTICAL, command=preview_box.yview)
        preview_y_scroll.grid(row=0, column=1, sticky="ns")
        preview_x_scroll = ttk.Scrollbar(preview_container, orient=tk.HORIZONTAL, command=preview_box.xview)
        preview_x_scroll.grid(row=1, column=0, sticky="ew")
        preview_box.configure(yscrollcommand=preview_y_scroll.set, xscrollcommand=preview_x_scroll.set)
        self._layout_preview_box = preview_box

        signal_frame = ttk.LabelFrame(container, text="Signal preview")
        signal_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        ttk.Label(signal_frame, text="Data can not be imported with selected settings", foreground="gray")\
            .pack(fill=tk.BOTH, expand=True, pady=12)

        ttk.Button(container, text="Close", command=lambda: self._close_layout(win)).pack(anchor="e", pady=(12, 0))

        self._update_layout_preview()

    def _close_preferences(self, window):
        if window.winfo_exists():
            window.destroy()
        self._preferences_window = None

    def _close_layout(self, window):
        if window.winfo_exists():
            window.destroy()
        self._layout_window = None
        self._layout_preview_box = None

    def _update_layout_preview(self, *_event):
        if not self._layout_preview_box:
            return
        if not (self._layout_window and self._layout_window.winfo_exists()):
            return

        preview_box = self._layout_preview_box
        preview_box.configure(state="normal")
        preview_box.delete("1.0", tk.END)

        if not self.filepaths:
            preview_box.insert("1.0", "No files loaded. Add files to preview their layout.")
        else:
            selection = self.files_list.curselection()
            index = selection[0] if selection else 0
            path = self.filepaths[index]
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    snippet = fh.readlines()[:100]
                if not snippet:
                    preview_box.insert("1.0", f"{os.path.basename(path)} is empty.")
                else:
                    preview_box.insert("1.0", "".join(snippet))
            except Exception as exc:
                preview_box.insert("1.0", f"Could not load preview for {os.path.basename(path)}:\n{exc}")

        preview_box.configure(state="disabled")

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
                    k_mad=k_mad
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

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
        self.title("BP Variability (ARV/SD/CV) with Optional Artifact Filtering")
        self.geometry("1180x640")
        self.minsize(980, 560)

        self.filepaths = []
        self.results = []

        # Top: file controls
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Files:").pack(side=tk.LEFT)
        self.files_var = tk.StringVar(value="")
        self.files_entry = ttk.Entry(top, textvariable=self.files_var)
        self.files_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        ttk.Button(top, text="Add Files…", command=self.add_files).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(top, text="Clear", command=self.clear_files).pack(side=tk.LEFT)

        # Time window + filtering controls row
        tw = ttk.Frame(self, padding=(8,0,8,8))
        tw.pack(side=tk.TOP, fill=tk.X, pady=(4,4))

        ttk.Label(tw, text="Start (s or mm:ss):").pack(side=tk.LEFT)
        self.start_var = tk.StringVar(value="")
        ttk.Entry(tw, width=10, textvariable=self.start_var).pack(side=tk.LEFT, padx=(6,16))

        ttk.Label(tw, text="End (s or mm:ss):").pack(side=tk.LEFT)
        self.end_var = tk.StringVar(value="")
        ttk.Entry(tw, width=10, textvariable=self.end_var).pack(side=tk.LEFT, padx=(6,20))

        # Filter enable
        self.filter_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(tw, text="Filter BP artifacts", variable=self.filter_on).pack(side=tk.LEFT, padx=(0,14))

        # Method dropdown
        ttk.Label(tw, text="Method:").pack(side=tk.LEFT)
        self.filter_method = tk.StringVar(value="rolling")  # 'rolling' or 'global'
        ttk.Combobox(tw, width=10, textvariable=self.filter_method,
                     values=["rolling", "global"], state="readonly").pack(side=tk.LEFT, padx=(6,14))

        # Params
        # Global: k_sd
        ttk.Label(tw, text="k_SD (global):").pack(side=tk.LEFT)
        self.k_sd_var = tk.StringVar(value="3.0")
        ttk.Entry(tw, width=6, textvariable=self.k_sd_var).pack(side=tk.LEFT, padx=(6,16))

        # Rolling: window_s, k_mad
        ttk.Label(tw, text="Window s (rolling):").pack(side=tk.LEFT)
        self.window_s_var = tk.StringVar(value="15")
        ttk.Entry(tw, width=6, textvariable=self.window_s_var).pack(side=tk.LEFT, padx=(6,16))

        ttk.Label(tw, text="k_MAD (rolling):").pack(side=tk.LEFT)
        self.k_mad_var = tk.StringVar(value="4.0")
        ttk.Entry(tw, width=6, textvariable=self.k_mad_var).pack(side=tk.LEFT, padx=(6,16))

        ttk.Button(tw, text="Compute", command=self.compute).pack(side=tk.LEFT, padx=(10,0))
        ttk.Button(tw, text="Save CSV…", command=self.save_csv).pack(side=tk.LEFT, padx=(8,0))

        # Results table
        cols = (
            "file", "interval_s", "start_s", "end_s",

            "n_SBP_raw", "n_SBP_kept", "Mean_SBP", "SD_SBP", "CV_SBP_pct", "ARV_SBP",
            "n_DBP_raw", "n_DBP_kept", "Mean_DBP", "SD_DBP", "CV_DBP_pct", "ARV_DBP",
            "n_MAP_raw", "n_MAP_kept", "Mean_MAP", "SD_MAP", "CV_MAP_pct", "ARV_MAP",
        )
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=14)
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
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(4,8))

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0,6))

        # Styling (optional nicer look)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select TXT files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not paths:
            return
        for p in paths:
            if p not in self.filepaths:
                self.filepaths.append(p)
        self.files_var.set("; ".join(self.filepaths))
        self.status.set(f"Added {len(paths)} file(s). Total: {len(self.filepaths)}")

    def clear_files(self):
        self.filepaths = []
        self.files_var.set("")
        self.results = []
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.status.set("Cleared files and results.")

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

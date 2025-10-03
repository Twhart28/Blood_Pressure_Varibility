#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

# ---- deps -------------------------------------------------------
try:
    import pyarrow as pa
    import pyarrow.csv as pv
except Exception:
    sys.stderr.write("This script requires PyArrow. Try: pip install pyarrow\n")
    raise

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---- Header scan ------------------------------------------------
def find_data_start_and_titles(path: Path):
    """
    Find the first data row (after 'Range=') and build canonical column names:
    ['Index'] + channel_titles + ['Comment']
    """
    data_start = 0
    channel_titles = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("ChannelTitle="):
                parts = line.rstrip("\n").split("\t")
                channel_titles = parts[1:] if len(parts) > 1 else []
            if line.startswith("Range="):
                data_start = i + 1
    names = ["Index"] + channel_titles + ["Comment"]
    return data_start, names


# ---- Fast pre-sanitizer -----------------------------------------
def sanitize_to_temp(path: Path, data_start: int, expected_cols: int, delimiter="\t"):
    """
    Create a sanitized temp file with exactly `expected_cols` per data row:
    - Lines before data_start are copied as-is (header)
    - For data rows:
        * If len(parts) > expected_cols: overflow is joined into the last column
        * If len(parts) < expected_cols: pad with empty strings
    Returns the temp file Path.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    tmp_path = Path(tmp.name)
    tmp.close()

    with path.open("r", encoding="utf-8", errors="ignore") as src, \
         tmp_path.open("w", encoding="utf-8", errors="ignore") as dst:
        for i, line in enumerate(src):
            # Keep header untouched
            if i < data_start:
                dst.write(line)
                continue

            # Normalize line endings
            line = line.rstrip("\n").rstrip("\r")

            # Skip true empty lines in data section
            if not line:
                continue

            parts = line.split(delimiter)

            if len(parts) > expected_cols:
                # join overflow into the last column (Comment)
                head = parts[:expected_cols-1]
                tail = delimiter.join(parts[expected_cols-1:])
                parts = head + [tail]
            elif len(parts) < expected_cols:
                parts = parts + [""] * (expected_cols - len(parts))

            dst.write(delimiter.join(parts) + "\n")

    return tmp_path


# ---- PyArrow read -----------------------------------------------
def read_with_pyarrow(path, data_start, names, selected_cols, include_comment,
                      row_start=None, row_end=None):
    read_options = pv.ReadOptions(
        skip_rows=data_start,
        autogenerate_column_names=True,
        use_threads=True
    )
    parse_options = pv.ParseOptions(delimiter="\t")
    convert_options = pv.ConvertOptions(
        strings_can_be_null=True,
        null_values=["NaN"]
    )

    table = pv.read_csv(
        str(path),
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options
    )

    # Rename columns to our canonical list (trim/pad names safely)
    current_n = table.num_columns
    wanted_n = len(names)
    if current_n >= wanted_n:
        table = table.rename_columns(names + [f"EXTRA_{i}" for i in range(current_n - wanted_n)])
    else:
        padded = names + [f"EXTRA_{i}" for i in range(wanted_n - current_n)]
        table = table.rename_columns(padded[:current_n])

    # Select desired columns
    usecols = list(selected_cols)
    if "Comment" in usecols and not include_comment:
        usecols.remove("Comment")
    table = table.select(usecols)

    # Optional row slicing
    if (row_start is not None) or (row_end is not None):
        start = int(row_start or 0)
        if (row_end is not None) and (row_end > start):
            length = int(row_end - start)
            table = table.slice(start, length)
        else:
            table = table.slice(start)

    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    return df


# ---- UI ---------------------------------------------------------
class LoaderUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fast TSV Loader (PyArrow + Comment-safe)")
        self.geometry("760x560")

        self.path_var = tk.StringVar()
        self.include_comment_var = tk.BooleanVar(value=False)
        self.row_start_var = tk.StringVar(value="")
        self.row_end_var = tk.StringVar(value="")

        self.names = None
        self.data_start = None

        # File picker
        frm_file = ttk.Frame(self)
        frm_file.pack(fill="x", padx=10, pady=10)
        ttk.Label(frm_file, text="File:").pack(side="left")
        ttk.Entry(frm_file, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(frm_file, text="Browse...", command=self.pick_file).pack(side="left")

        # Columns list
        frm_cols = ttk.LabelFrame(self, text="Select columns to load (Ctrl/Cmd + click for multi-select)")
        frm_cols.pack(fill="both", expand=True, padx=10, pady=10)

        self.lst_cols = tk.Listbox(frm_cols, selectmode="extended", height=14)
        self.lst_cols.pack(side="left", fill="both", expand=True, padx=(8,4), pady=8)
        self.scr_cols = ttk.Scrollbar(frm_cols, orient="vertical", command=self.lst_cols.yview)
        self.scr_cols.pack(side="left", fill="y", pady=8)
        self.lst_cols.config(yscrollcommand=self.scr_cols.set)

        # Options
        frm_opts = ttk.Frame(self)
        frm_opts.pack(fill="x", padx=10, pady=4)
        ttk.Checkbutton(frm_opts, text="Include trailing Comment column",
                        variable=self.include_comment_var).pack(side="left", padx=(0,20))
        ttk.Label(frm_opts, text="Row start (optional):").pack(side="left")
        ttk.Entry(frm_opts, width=10, textvariable=self.row_start_var).pack(side="left", padx=(4,20))
        ttk.Label(frm_opts, text="Row end (optional):").pack(side="left")
        ttk.Entry(frm_opts, width=10, textvariable=self.row_end_var).pack(side="left", padx=4)

        # Buttons
        frm_btns = ttk.Frame(self)
        frm_btns.pack(fill="x", padx=10, pady=10)
        ttk.Button(frm_btns, text="Load & Plot", command=self.load_and_plot).pack(side="right")
        ttk.Button(frm_btns, text="Quit", command=self.destroy).pack(side="right", padx=6)

        # Status
        self.status_var = tk.StringVar(value="Pick a file to begin.")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=(0,10))

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Select TSV file",
            filetypes=[("Text/TSV files", "*.txt *.tsv"), ("All files", "*.*")]
        )
        if not path:
            return
        self.path_var.set(path)
        try:
            self.data_start, self.names = find_data_start_and_titles(Path(path))
            self.lst_cols.delete(0, "end")
            for name in self.names:
                self.lst_cols.insert("end", name)
            preselect = ["Index"]
            for candidate in ["Heart Rate", "HR", "SBP", "DBP", "MAP"]:
                if candidate in self.names:
                    preselect.append(candidate)
            for i, name in enumerate(self.names):
                if name in preselect:
                    self.lst_cols.selection_set(i)
            self.status_var.set(f"Parsed header. Data starts at line {self.data_start}.")
        except Exception as e:
            messagebox.showerror("Header parse error", str(e))
            self.status_var.set("Failed to parse header.")

    def load_and_plot(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please choose a file first.")
            return
        if not self.names or self.data_start is None:
            try:
                self.data_start, self.names = find_data_start_and_titles(Path(path))
            except Exception as e:
                messagebox.showerror("Header parse error", str(e))
                return

        sel_indices = self.lst_cols.curselection()
        if not sel_indices:
            messagebox.showwarning("No columns", "Select at least one column (e.g., Index + a signal).")
            return
        selected_cols = [self.names[i] for i in sel_indices]

        # Row range
        row_start = self.row_start_var.get().strip()
        row_end = self.row_end_var.get().strip()
        row_start = int(row_start) if row_start.isdigit() else None
        row_end = int(row_end) if row_end.isdigit() else None

        include_comment = bool(self.include_comment_var.get())

        # If including Comment, sanitize first so column counts are consistent
        src_path = Path(path)
        read_path = src_path
        tmp_for_cleanup = None

        try:
            if include_comment:
                expected_cols = len(self.names)  # Index + channels + Comment
                self.status_var.set("Sanitizing comment column...")
                self.update_idletasks()
                tmp_for_cleanup = sanitize_to_temp(src_path, self.data_start, expected_cols, delimiter="\t")
                read_path = tmp_for_cleanup

            self.status_var.set("Reading with PyArrow...")
            self.update_idletasks()

            df = read_with_pyarrow(
                read_path,
                self.data_start,
                self.names,
                selected_cols,
                include_comment,
                row_start=row_start,
                row_end=row_end
            )

            # Clean up temp file
            if tmp_for_cleanup is not None and tmp_for_cleanup.exists():
                try:
                    tmp_for_cleanup.unlink()
                except Exception:
                    pass

            # Plot
            numeric_cols = [c for c in df.columns if c != "Comment"]
            if not numeric_cols:
                messagebox.showinfo("Loaded", f"Loaded {len(df)} rows, but no numeric columns selected to plot.")
                return

            if "Index" in df.columns:
                x = pd.to_numeric(df["Index"], errors="coerce").to_numpy()
            else:
                x = np.arange(len(df), dtype=np.int32)

            step = max(1, len(x) // 100_000)
            x_ds = x[::step]

            plt.figure()
            plotted_any = False
            for col in df.columns:
                if col in ("Index", "Comment"):
                    continue
                y = pd.to_numeric(df[col], errors="coerce").to_numpy()
                plt.plot(x_ds, y[::step], label=col)
                plotted_any = True

            if not plotted_any:
                messagebox.showinfo("Loaded", f"Loaded {len(df)} rows, but nothing numeric to plot.")
                return

            plt.title(Path(path).name)
            plt.xlabel("Index (samples)")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.show()

            self.status_var.set(f"Loaded {len(df)} rows. Plotted {len(x_ds)} points per selected signal.")

        except Exception as e:
            messagebox.showerror("Read error", str(e))
            self.status_var.set("Read failed.")
        finally:
            # Ensure temp file is removed on any error path
            if 'tmp_for_cleanup' in locals() and tmp_for_cleanup is not None and tmp_for_cleanup.exists():
                try:
                    tmp_for_cleanup.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    app = LoaderUI()
    app.mainloop()

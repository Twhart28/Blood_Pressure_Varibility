#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TSV/CSV Preview + Flexible Loader (PyArrow)
- Preview ~150 lines with user-chosen delimiter
- User assigns header row (or None), data start row, and columns to load
- Optional sanitizer: folds extra fields on a line into the last selected column
- Loads full file via PyArrow using chosen settings
"""

import sys
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---- deps ----
try:
    import pyarrow as pa
    import pyarrow.csv as pv
except Exception:
    sys.stderr.write("This app needs PyArrow. Install with: pip install pyarrow\n")
    raise

import pandas as pd
import numpy as np

# matplotlib is optional (only if "Plot after load" is checked)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ---------- small helpers ----------
def sniff_delimiter(sample_text: str) -> str:
    """Very small heuristic delimiter sniffer."""
    candidates = ["\t", ",", ";", "|", " "]
    best = "\t"
    best_hits = -1
    for d in candidates:
        hits = 0
        for line in sample_text.splitlines()[:30]:
            if d in line:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best = d
    return best


def parse_lines(lines, delimiter):
    """Split lines by delimiter into a list-of-lists (no quoting)."""
    rows = []
    for ln in lines:
        ln = ln.rstrip("\n").rstrip("\r")
        if ln == "":
            rows.append([])
        else:
            rows.append(ln.split(delimiter))
    return rows


def sanitize_to_temp(src_path: Path, data_start_line: int, expected_cols: int, delimiter: str, last_col_index: int):
    """
    Create a sanitized temp file ensuring each data row has exactly expected_cols fields.
    Overflow from the target column (see target_idx) to the end of the line is folded into that target cell.
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp_path = Path(tmp.name)
    tmp.close()

    # Choose a safe target column to absorb overflow:
    # - if the caller's index is bad or beyond the last valid index, use the final column in the schema
    target_idx = last_col_index if (0 <= last_col_index < expected_cols) else (expected_cols - 1)

    with src_path.open("r", encoding="utf-8", errors="ignore") as src, \
         tmp_path.open("w", encoding="utf-8", errors="ignore") as dst:

        for i, line in enumerate(src):
            # Copy the header untouched
            if i < data_start_line:
                dst.write(line)
                continue

            row = line.rstrip("\r\n")
            if row == "":
                continue

            parts = row.split(delimiter)

            # If there are too many parts, fold everything from target_idx onward into one cell
            if len(parts) > expected_cols:
                head = parts[:target_idx]
                tail_joined = delimiter.join(parts[target_idx:])  # collapse all remaining into one
                parts = head + [tail_joined]

            # If there are too few, pad
            if len(parts) < expected_cols:
                parts = parts + [""] * (expected_cols - len(parts))

            # If somehow still too many (shouldn't happen now), truncate
            if len(parts) > expected_cols:
                parts = parts[:expected_cols]

            dst.write(delimiter.join(parts) + "\n")

    return tmp_path


# ---------- main app ----------
class PreviewLoader(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Preview & Load (PyArrow)")
        self.geometry("1050x680")

        # state
        self.path_var = tk.StringVar()
        self.delim_mode = tk.StringVar(value="Tab (\\t)")
        self.custom_delim = tk.StringVar(value="")
        self.header_row_var = tk.StringVar(value="(none)")
        self.data_start_var = tk.StringVar(value="1")
        self.preview_rows = 150

        self.sanitize_var = tk.BooleanVar(value=False)
        self.plot_var = tk.BooleanVar(value=HAS_MPL)  # default on if mpl is available

        self.available_columns = []  # list[str]
        self.file_preview_rows = []  # list[list[str]]
        self.preview_delim = "\t"

        self._build_ui()

    def _build_ui(self):
        # ---- top bar: file + delimiter ----
        top = ttk.Frame(self); top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="File:").pack(side="left")
        ttk.Entry(top, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Browse…", command=self.pick_file).pack(side="left", padx=(0,8))
        ttk.Button(top, text="Preview", command=self.load_preview).pack(side="left")

        # delimiter row
        delim_row = ttk.Frame(self); delim_row.pack(fill="x", padx=10, pady=(0,10))
        ttk.Label(delim_row, text="Delimiter:").pack(side="left")
        cmb = ttk.Combobox(
            delim_row, state="readonly", width=18, textvariable=self.delim_mode,
            values=[
                "Auto-detect",
                "Tab (\\t)",
                "Comma (,)",
                "Semicolon (;)",
                "Pipe (|)",
                "Space ( )",
                "Custom"
            ]
        )
        cmb.pack(side="left", padx=6)
        ttk.Entry(delim_row, width=8, textvariable=self.custom_delim).pack(side="left")
        ttk.Label(delim_row, text="(Used only if 'Custom')").pack(side="left", padx=(4,0))

        # ---- preview table ----
        table_frame = ttk.LabelFrame(self, text="Preview (first ~150 lines parsed with the selected delimiter)")
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.tree = ttk.Treeview(table_frame, columns=(), show="headings", height=12)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side="left", fill="y")
        self.tree.configure(yscrollcommand=vsb.set)

        # ---- header / data start / options ----
        options = ttk.Frame(self); options.pack(fill="x", padx=10, pady=6)
        ttk.Label(options, text="Header row (1-based, '(none)' for no header):").pack(side="left")
        self.header_row_entry = ttk.Entry(options, width=10, textvariable=self.header_row_var)
        self.header_row_entry.pack(side="left", padx=(6,18))

        ttk.Label(options, text="Data starts at row (1-based):").pack(side="left")
        ttk.Entry(options, width=10, textvariable=self.data_start_var).pack(side="left", padx=(6,18))

        ttk.Checkbutton(options, text="Sanitize variable-width rows (fold overflow into last selected column)",
                        variable=self.sanitize_var).pack(side="left", padx=(0,18))
        ttk.Checkbutton(options, text="Plot after load", variable=self.plot_var).pack(side="left")

        # ---- columns selection ----
        cols_frame = ttk.LabelFrame(self, text="Columns of interest (Ctrl/Cmd + click to multi-select)")
        cols_frame.pack(fill="both", expand=False, padx=10, pady=10)

        self.lst_cols = tk.Listbox(cols_frame, selectmode="extended", height=10, exportselection=False)
        self.lst_cols.pack(side="left", fill="both", expand=True, padx=(8,4), pady=8)
        cols_vsb = ttk.Scrollbar(cols_frame, orient="vertical", command=self.lst_cols.yview)
        cols_vsb.pack(side="left", fill="y", pady=8)
        self.lst_cols.config(yscrollcommand=cols_vsb.set)

        # ---- actions ----
        actions = ttk.Frame(self); actions.pack(fill="x", padx=10, pady=10)
        ttk.Button(actions, text="Apply Header to Columns", command=self.apply_header_to_columns)\
            .pack(side="left")
        ttk.Button(actions, text="Load Full File with PyArrow", command=self.load_full_file)\
            .pack(side="right")

        self.status_var = tk.StringVar(value="Select a file and click Preview.")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=(0,10))

    # ---------- UI callbacks ----------
    def pick_file(self):
        p = filedialog.askopenfilename(
            title="Select delimited text file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.path_var.set(p)

    def _current_delimiter(self, sample_text=""):
        mode = self.delim_mode.get()
        if mode == "Auto-detect":
            d = sniff_delimiter(sample_text)
        elif mode == "Tab (\\t)":
            d = "\t"
        elif mode == "Comma (,)":
            d = ","
        elif mode == "Semicolon (;)":
            d = ";"
        elif mode == "Pipe (|)":
            d = "|"
        elif mode == "Space ( )":
            d = " "
        else:
            d = self.custom_delim.get() or "\t"
        return d

    def load_preview(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Pick a file first.")
            return

        src = Path(path)
        if not src.exists():
            messagebox.showerror("Not found", f"File not found:\n{src}")
            return

        # read a small chunk of the file for preview & auto-detect
        with src.open("r", encoding="utf-8", errors="ignore") as f:
            sample_text = "".join([next(f, "") for _ in range(self.preview_rows)])

        self.preview_delim = self._current_delimiter(sample_text)
        rows = parse_lines(sample_text.splitlines(), self.preview_delim)
        self.file_preview_rows = rows

        # Build columns count from the widest row in the preview
        max_cols = max((len(r) for r in rows), default=0)
        cols = [f"C{i+1}" for i in range(max_cols)]
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, anchor="w")

        # fill preview table
        for i in self.tree.get_children():
            self.tree.delete(i)
        for r in rows[:100]:  # show ~100 in the grid (we read up to ~150)
            padded = r + [""] * (max_cols - len(r))
            self.tree.insert("", "end", values=padded)

        # reset columns list with generic names until header is applied
        self.available_columns = cols[:]  # generic
        self._refresh_columns_listbox()

        self.status_var.set(f"Preview loaded with delimiter '{repr(self.preview_delim)}'. "
                            f"Max columns in preview: {max_cols}.")

    def _refresh_columns_listbox(self, preselect_first=True):
        self.lst_cols.delete(0, "end")
        for name in self.available_columns:
            self.lst_cols.insert("end", name)
        if preselect_first and self.available_columns:
            # preselect first column + any common signals if they exist
            self.lst_cols.selection_set(0)

    def apply_header_to_columns(self):
        """Use the chosen header row to label columns and repopulate the selectable list."""
        if not self.file_preview_rows:
            messagebox.showinfo("Preview first", "Load a preview before applying a header.")
            return

        hdr_txt = self.header_row_var.get().strip()
        if hdr_txt.lower() in ("none", "(none)", ""):
            # keep generic names
            self.available_columns = [f"C{i+1}" for i in range(len(self.available_columns))]
            self._refresh_columns_listbox(preselect_first=True)
            self.status_var.set("No header: using generic column names (C1, C2, …).")
            return

        if not hdr_txt.isdigit():
            messagebox.showerror("Invalid header row", "Enter a 1-based integer row index, or '(none)'.")
            return

        hdr_idx_1based = int(hdr_txt)
        if hdr_idx_1based < 1 or hdr_idx_1based > len(self.file_preview_rows):
            messagebox.showerror("Out of range", "Header row is outside the preview range.")
            return

        header_row = self.file_preview_rows[hdr_idx_1based - 1]
        max_cols = max(len(header_row), len(self.available_columns))
        header_row = header_row + [""] * (max_cols - len(header_row))
        if not any(header_row):
            messagebox.showwarning("Empty header row", "That row appears empty; keeping generic names.")
            self.available_columns = [f"C{i+1}" for i in range(max_cols)]
        else:
            # Replace blanks with Cn
            names = []
            for i, v in enumerate(header_row):
                v = v.strip()
                names.append(v if v else f"C{i+1}")
            self.available_columns = names

        self._refresh_columns_listbox(preselect_first=True)
        self.status_var.set(f"Header applied from row {hdr_idx_1based}.")

    def load_full_file(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Pick a file first.")
            return

        # delimiter
        # (If the user never hit Preview, still compute a delimiter.)
        src = Path(path)
        if not self.file_preview_rows:
            with src.open("r", encoding="utf-8", errors="ignore") as f:
                sample_text = "".join([next(f, "") for _ in range(self.preview_rows)])
            self.preview_delim = self._current_delimiter(sample_text)

        delimiter = self.preview_delim or "\t"

        # header row index (1-based), or None
        hdr_txt = self.header_row_var.get().strip()
        header_row_1based = None
        if hdr_txt.lower() not in ("none", "(none)", ""):
            if not hdr_txt.isdigit():
                messagebox.showerror("Invalid header row", "Enter a 1-based integer row index, or '(none)'.")
                return
            header_row_1based = int(hdr_txt)

        # data start row (1-based, first line PyArrow should parse as data)
        dstart_txt = self.data_start_var.get().strip()
        if not dstart_txt.isdigit():
            messagebox.showerror("Invalid data start", "Enter a 1-based integer for where data begins.")
            return
        data_start_1based = int(dstart_txt)
        if data_start_1based < 1:
            messagebox.showerror("Invalid data start", "Data start must be >= 1.")
            return

        # selected columns
        sel_idx = self.lst_cols.curselection()
        if not sel_idx:
            messagebox.showwarning("No columns", "Select at least one column to load.")
            return
        selected_cols = [self.available_columns[i] for i in sel_idx]

        # Build 'names' and skip_rows for PyArrow
        # PyArrow will skip data_start_1based-1 lines total (0-based index of first data line)
        skip_rows = data_start_1based - 1

        # Construct "names" mapping for the table: we don't rely on file headers;
        # we rename columns AFTER reading based on the width seen in the first data row.
        # To do that safely, we first need to know how many columns exist in the data.
        # We'll peek one data line from disk (starting at data_start_1based).
        with src.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(skip_rows):
                next(f, "")
            first_data = next(f, "").rstrip("\n").rstrip("\r")
        first_parts = first_data.split(delimiter) if first_data else []
        n_cols = max(len(first_parts), len(self.available_columns))
        # If user picked header names for fewer cols than present, pad generic names
        if len(self.available_columns) < n_cols:
            self.available_columns += [f"C{i+1}" for i in range(len(self.available_columns), n_cols)]

        names_all = self.available_columns[:n_cols]

        # Optionally sanitize variable-width rows into last selected column
        read_path = src
        tmp_path = None
        try:
            if self.sanitize_var.get():
                last_sel = selected_cols[-1]
                last_idx = names_all.index(last_sel)
                expected_cols = n_cols
                self.status_var.set("Sanitizing variable-width rows…"); self.update_idletasks()
                tmp_path = sanitize_to_temp(src, skip_rows, expected_cols, delimiter, last_idx)
                read_path = tmp_path

            # Read with PyArrow
            self.status_var.set("Reading full file with PyArrow…"); self.update_idletasks()
            table = pv.read_csv(
                str(read_path),
                read_options=pv.ReadOptions(skip_rows=skip_rows, autogenerate_column_names=True, use_threads=True),
                parse_options=pv.ParseOptions(delimiter=delimiter),
                convert_options=pv.ConvertOptions(strings_can_be_null=True, null_values=["NaN"])
            )

            # rename to our chosen names (pad or trim if needed)
            cur = table.num_columns
            want = len(names_all)
            if cur >= want:
                table = table.rename_columns(names_all + [f"EXTRA_{i}" for i in range(cur - want)])
            else:
                padded = names_all + [f"EXTRA_{i}" for i in range(want - cur)]
                table = table.rename_columns(padded[:cur])

            # Select only requested columns that actually exist
            sel_cols_final = [c for c in selected_cols if c in table.column_names]
            if not sel_cols_final:
                messagebox.showerror("No matching columns", "Selected columns not found after parsing.")
                return

            table = table.select(sel_cols_final)

            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            self.status_var.set(f"Loaded {len(df)} rows × {len(df.columns)} cols.")

            # Optional quick plot
            if self.plot_var.get():
                if not HAS_MPL:
                    messagebox.showinfo("Plot unavailable", "matplotlib not installed; skipping plot.")
                else:
                    # Use the first selected column as x if it looks like Time/Index, else x=0..N-1
                    x = None
                    for cand in sel_cols_final:
                        if cand.lower() in ("time", "index", "t", "sample", "timestamp"):
                            x = pd.to_numeric(df[cand], errors="coerce").to_numpy()
                            break
                    if x is None:
                        x = np.arange(len(df), dtype=np.int32)

                    step = max(1, len(x)//100_000)
                    x_ds = x[::step]

                    import matplotlib.pyplot as plt
                    plt.figure()
                    plotted = False
                    for c in sel_cols_final:
                        if c.lower() in ("time", "index", "t", "sample", "timestamp"):
                            continue
                        y = pd.to_numeric(df[c], errors="coerce").to_numpy()
                        plt.plot(x_ds, y[::step], label=c); plotted = True
                    if plotted:
                        plt.title(Path(path).name)
                        plt.xlabel("Time/Index" if x is not None else "Row")
                        plt.ylabel("Value")
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
                    else:
                        messagebox.showinfo("Loaded", "Loaded successfully (no numeric series to plot).")

        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.status_var.set("Load failed.")
        finally:
            if tmp_path is not None and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    app = PreviewLoader()
    app.mainloop()

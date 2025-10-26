"""BP Analyzer V1
===================

A command-line utility for streaming ingestion of delimited blood pressure files.

Features
--------
* Streaming parser with configurable delimiter, NA values, and first data row.
* Terminal preview that lets users map source columns to the expected schema.
* Persistence of the selected columns to a Parquet file plus a JSON manifest
  capturing the parsing configuration and resulting schema.

The module can be run as a script. Example::

    python BP_Analyzer_V1.py --file path/to/data.csv

"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional

import itertools
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ParserConfig:
    """Configuration parameters that drive parsing."""

    file_path: str
    delimiter: str = ","
    na_values: List[str] = None
    first_data_row: int = 1
    header_row: Optional[int] = None
    chunk_size: int = 5000
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if self.na_values is None:
            self.na_values = ["", "NA", "NaN"]
        if self.first_data_row < 1:
            raise ValueError("first_data_row must be >= 1")
        if self.header_row is not None and self.header_row < 1:
            raise ValueError("header_row must be >= 1 when provided")
        if self.header_row is not None and self.header_row >= self.first_data_row:
            # Ensure the header is not interpreted as data.
            print(
                "[Info] header_row occurs at or after first_data_row. "
                "Adjusting first_data_row to start after the header."
            )
            self.first_data_row = self.header_row + 1


def interpret_delimiter(raw: str) -> str:
    """Convert user provided delimiter strings such as "\\t" into literals."""

    escape_map = {"\\t": "\t", "\\n": "\n", "\\r": "\r"}
    return escape_map.get(raw, raw)


def read_header_row(config: ParserConfig) -> Optional[List[str]]:
    """Return column names from the configured header row if available."""

    if config.header_row is None:
        return None

    with open(config.file_path, "r", encoding=config.encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=config.delimiter)
        for index, row in enumerate(reader, start=1):
            if index == config.header_row:
                return [value.strip() for value in row]
    return None


def compute_skiprows(config: ParserConfig) -> List[int]:
    """Compute 0-based row numbers that should be skipped when reading data."""

    if config.first_data_row <= 1:
        return []
    return list(range(config.first_data_row - 1))


def stream_dataframe_chunks(
    config: ParserConfig, column_names: Optional[List[str]]
) -> Iterator[pd.DataFrame]:
    """Yield chunks of the dataset as pandas DataFrames."""

    skiprows = compute_skiprows(config)
    read_kwargs = {
        "sep": config.delimiter,
        "na_values": config.na_values,
        "chunksize": config.chunk_size,
        "skiprows": skiprows,
        "header": None,
        "encoding": config.encoding,
        "engine": "python",
    }
    if column_names:
        read_kwargs["names"] = column_names

    chunk_iterator = pd.read_csv(config.file_path, **read_kwargs)

    inferred_names: Optional[List[str]] = column_names
    for chunk in chunk_iterator:
        if inferred_names is None:
            inferred_names = [f"column_{idx + 1}" for idx in range(len(chunk.columns))]
        chunk.columns = inferred_names
        yield chunk


def display_preview(chunk: pd.DataFrame, max_rows: int = 10) -> None:
    """Pretty-print a sample of the chunk for terminal inspection."""

    preview = chunk.head(max_rows)
    print("\nPreview of the incoming data:")
    print(preview.to_string(index=False))


def prompt_column_mapping(column_names: List[str]) -> Dict[str, str]:
    """Interactively ask the user to map columns to the expected schema."""

    print("\nAvailable columns:")
    for idx, name in enumerate(column_names, start=1):
        print(f"  {idx}. {name}")

    required_fields = ["Time", "Blood Pressure"]
    optional_fields = ["Comments", "ECG"]
    mapping: Dict[str, Optional[str]] = {}

    def ask_for_field(field_name: str, required: bool) -> Optional[str]:
        prompt = (
            f"Enter the column name or number to use for '{field_name}'"
            f" ({'required' if required else 'optional'})."
        )
        while True:
            user_input = input(f"{prompt} [press Enter to skip]: ").strip()
            if not user_input and not required:
                return None
            if not user_input and required:
                print("This field is required. Please choose a column.")
                continue
            if user_input.isdigit():
                index = int(user_input) - 1
                if 0 <= index < len(column_names):
                    return column_names[index]
                print("Invalid column number; please try again.")
                continue
            if user_input in column_names:
                return user_input
            print("Unrecognized column; please enter a valid column name or number.")

    for field in required_fields:
        column = ask_for_field(field, required=True)
        mapping[field] = column

    for field in optional_fields:
        column = ask_for_field(field, required=False)
        if column:
            mapping[field] = column

    # Sanity check for duplicate assignments.
    chosen_columns = [col for col in mapping.values() if col]
    if len(chosen_columns) != len(set(chosen_columns)):
        raise ValueError("Each source column can only be assigned once.")

    return {k: v for k, v in mapping.items() if v}


def prompt_output_paths(default_path: str) -> tuple[str, str]:
    """Ask the user where output artifacts should be stored."""

    base_default = os.path.splitext(default_path)[0]
    default_parquet = f"{base_default}_selected.parquet"
    default_manifest = f"{base_default}_manifest.json"

    parquet_path = input(
        f"Enter output Parquet path [{default_parquet}]: "
    ).strip() or default_parquet
    manifest_path = input(
        f"Enter output manifest path [{default_manifest}]: "
    ).strip() or default_manifest
    return parquet_path, manifest_path


def write_selected_columns(
    first_chunk: pd.DataFrame,
    remaining_chunks: Iterable[pd.DataFrame],
    mapping: Dict[str, str],
    parquet_path: str,
) -> Dict[str, Any]:
    """Persist selected columns to Parquet and return simple metrics."""

    schema = None
    row_count = 0
    writer: Optional[pq.ParquetWriter] = None

    all_chunks = itertools.chain([first_chunk], remaining_chunks)
    selected_columns = list(mapping.values())
    rename_map = {source: target for target, source in mapping.items()}

    try:
        for chunk in all_chunks:
            subset = chunk[selected_columns].rename(columns=rename_map)
            table = pa.Table.from_pandas(subset, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(parquet_path, schema)
            writer.write_table(table)
            row_count += len(subset)
    finally:
        if writer:
            writer.close()

    return {
        "rows_written": row_count,
        "schema": schema,
    }


def build_manifest(
    config: ParserConfig,
    mapping: Dict[str, str],
    parquet_path: str,
    manifest_path: str,
    metrics: Dict[str, Any],
) -> None:
    """Write a JSON manifest describing the generated Parquet file."""

    schema = metrics.get("schema")
    schema_description = []
    if schema:
        schema_description = [
            {"name": field.name, "type": str(field.type)}
            for field in schema
        ]

    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_file": os.path.abspath(config.file_path),
        "parquet_file": os.path.abspath(parquet_path),
        "column_mapping": mapping,
        "rows_written": metrics.get("rows_written", 0),
        "parser_config": {
            "delimiter": config.delimiter,
            "na_values": config.na_values,
            "first_data_row": config.first_data_row,
            "header_row": config.header_row,
            "chunk_size": config.chunk_size,
            "encoding": config.encoding,
        },
        "schema": schema_description,
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nManifest written to {manifest_path}")


def run_interactive_session(config: ParserConfig) -> None:
    """Execute the main interactive parsing workflow."""

    column_names = read_header_row(config)
    chunk_iterator = stream_dataframe_chunks(config, column_names)
    try:
        first_chunk = next(chunk_iterator)
    except StopIteration:
        print("No data found with the provided configuration.")
        return

    # If no header row was provided, use column names inferred from the first chunk.
    column_names = list(first_chunk.columns)

    display_preview(first_chunk)
    mapping = prompt_column_mapping(column_names)
    parquet_path, manifest_path = prompt_output_paths(config.file_path)

    # Persist data to Parquet.
    metrics = write_selected_columns(first_chunk, chunk_iterator, mapping, parquet_path)
    print(f"\nParquet written to {parquet_path}")
    print(f"Rows written: {metrics.get('rows_written', 0)}")

    build_manifest(config, mapping, parquet_path, manifest_path, metrics)


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood Pressure streaming parser")
    parser.add_argument("--file", required=True, help="Path to the source data file")
    parser.add_argument(
        "--delimiter",
        help="Column delimiter (default: comma). Use \\t for tab.",
        default=",",
    )
    parser.add_argument(
        "--na",
        help="Comma-separated list of strings that should be interpreted as NA",
        default="",
    )
    parser.add_argument(
        "--first-data-row",
        type=int,
        default=1,
        help="1-based index for the first row containing data",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        help="1-based index for the header row (optional)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of rows to read per chunk",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding used by the input file",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ParserConfig:
    delimiter = interpret_delimiter(args.delimiter)
    na_values = [token.strip() for token in args.na.split(",") if token.strip()]

    return ParserConfig(
        file_path=args.file,
        delimiter=delimiter,
        na_values=na_values or None,
        first_data_row=args.first_data_row,
        header_row=args.header_row,
        chunk_size=args.chunk_size,
        encoding=args.encoding,
    )


def main() -> None:
    args = parse_cli_arguments()
    config = build_config_from_args(args)
    run_interactive_session(config)


if __name__ == "__main__":
    main()

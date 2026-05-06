"""
Export DeepFilterNet3 to ONNX.

Usage
-----
# Both model types (default):
python run_export.py

# Spectral model only  (spec-in / spec-out):
python run_export.py --spec-only

# Component models only (enc / erb_dec / df_dec):
python run_export.py --components-only

# Full waveform-feature model as well:
python run_export.py --export-full

Outputs land in ./onnx_export/<model_name>/  relative to this file.
"""

import argparse
import os
import sys

# Make sure the DeepFilterNet Python package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepFilterNet"))

from df.enhance import init_df
from df.scripts.export import export

MODEL_DIR = os.path.join(os.path.dirname(__file__), "_export_model", "DeepFilterNet3")
OUT_BASE  = os.path.join(os.path.dirname(__file__), "onnx_export")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=MODEL_DIR, help="Path to model directory (default: %(default)s)")
    ap.add_argument("--out-dir",   default=None,       help="Output directory (default: onnx_export/<model>)")
    ap.add_argument("--spec-only",       action="store_true", help="Export spectral model only")
    ap.add_argument("--components-only", action="store_true", help="Export component models only")
    ap.add_argument("--export-waveform", action="store_true", help="Export end-to-end waveform model (deepfilternet_v3.onnx, opset 17)")
    ap.add_argument("--export-full",     action="store_true", help="Also export full-feature monolithic model")
    ap.add_argument("--no-check",        action="store_true", help="Skip ORT parity check after export")
    ap.add_argument("--simplify",        action="store_true", help="Run onnxsim simplification pass")
    ap.add_argument("--opset",           type=int, default=14, help="ONNX opset version (default: %(default)s)")
    args = ap.parse_args()

    if args.spec_only and args.components_only:
        ap.error("--spec-only and --components-only are mutually exclusive")

    export_spec       = not args.components_only
    export_components = not args.spec_only

    import pathlib
    model_name = pathlib.Path(args.model_dir).name
    out_dir = args.out_dir or os.path.join(OUT_BASE, model_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model from {args.model_dir} ...")
    model, df_state, _, epoch = init_df(
        args.model_dir,
        log_level="WARNING",
        config_allow_defaults=True,
    )
    print(f"  Loaded epoch {epoch}")

    print(f"\nExporting to {out_dir}")
    print(f"  export_spec       = {export_spec}")
    print(f"  export_waveform   = {args.export_waveform}")
    print(f"  export_components = {export_components}")
    print(f"  export_full       = {args.export_full}")
    print(f"  opset             = {args.opset}\n")

    export(
        model,
        out_dir,
        df_state=df_state,
        check=not args.no_check,
        simplify=args.simplify,
        opset=args.opset,
        export_spec=export_spec,
        export_waveform=args.export_waveform,
        export_full=args.export_full,
        export_components=export_components,
    )

    print("\nDone. Output files:")
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        size  = os.path.getsize(fpath)
        print(f"  {f:45s}  {size/1024:8.1f} KB")


if __name__ == "__main__":
    main()

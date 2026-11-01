"""Command-line batch processing built on :mod:`pixelforge.core`.

Examples::

    python -m pixelforge.cli photo.jpg out.jpg --preset Vivid --resize 1600x
    python -m pixelforge.cli *.png --out-dir small --scale 0.5 --filter "Unsharp Mask:radius=1.5"
    python -m pixelforge.cli photo.jpg out.png --adjust contrast=20 --adjust temperature=15 --rotate 90
    python -m pixelforge.cli --list-filters
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import APP_NAME, __version__
from .core import io as pfio
from .core.adjustments import SLIDERS
from .core.document import Document, DocumentError
from .core.filters import FILTERS
from .core.presets import PRESETS, preset_adjustments


def parse_filter(text: str) -> tuple[str, dict]:
    """``"Gaussian Blur:radius=3,foo=bar"`` -> ``("Gaussian Blur", {"radius": "3", "foo": "bar"})``."""
    name, _, rest = text.partition(":")
    name = name.strip()
    if name not in FILTERS:
        raise argparse.ArgumentTypeError(f"unknown filter '{name}' (see --list-filters)")
    params: dict[str, str] = {}
    for item in filter(None, (p.strip() for p in rest.split(","))):
        key, eq, value = item.partition("=")
        if not eq:
            raise argparse.ArgumentTypeError(f"filter parameter '{item}' must look like key=value")
        params[key.strip()] = value.strip()
    return name, params


def parse_adjust(text: str) -> tuple[str, int]:
    key, eq, value = text.partition("=")
    key = key.strip().lower()
    if not eq or key not in SLIDERS:
        raise argparse.ArgumentTypeError(f"adjustment must be key=value with key in {', '.join(SLIDERS)}")
    try:
        return key, int(float(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"adjustment value for {key} must be a number") from None


def parse_size(text: str) -> tuple[int | None, int | None]:
    """``1600x900``, ``1600x`` (keep aspect) or ``x900``."""
    w, sep, h = text.lower().partition("x")
    if not sep:
        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT, WIDTHx or xHEIGHT")
    try:
        return (int(w) if w else None, int(h) if h else None)
    except ValueError:
        raise argparse.ArgumentTypeError("size values must be integers") from None


def parse_box(text: str) -> tuple[int, int, int, int]:
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("crop values must be integers") from None
    return x, y, x + w, y + h


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pixelforge-cli", description=f"{APP_NAME} {__version__} - batch image processing")
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="INPUT... [OUTPUT]",
        help="input image(s) or .pfp project(s); the last path is the output unless --out-dir/--info is used",
    )
    parser.add_argument("--out-dir", help="write outputs here, keeping the input file names")
    parser.add_argument("--format", help="output extension when using --out-dir (e.g. png, jpg, webp)")
    parser.add_argument("--preset", choices=list(PRESETS), help="apply an adjustment preset")
    parser.add_argument(
        "--adjust", action="append", type=parse_adjust, default=[], metavar="KEY=VALUE", help="set an adjustment slider (repeatable)"
    )
    parser.add_argument(
        "--filter", action="append", type=parse_filter, default=[], metavar="NAME[:k=v,...]", help="apply a filter (repeatable, in order)"
    )
    parser.add_argument("--resize", type=parse_size, metavar="WxH", help="resize to WxH, Wx or xH (keeps aspect when one side is empty)")
    parser.add_argument("--scale", type=float, help="scale by a factor, e.g. 0.5")
    parser.add_argument("--rotate", type=float, default=0, metavar="DEG", help="rotate counter-clockwise by DEG degrees")
    parser.add_argument("--flip", choices=["h", "v"], help="flip horizontally or vertically")
    parser.add_argument("--crop", type=parse_box, metavar="X,Y,W,H", help="crop rectangle")
    parser.add_argument("--trim", action="store_true", help="trim transparent borders")
    parser.add_argument("--quality", type=int, default=92, help="JPEG/WebP quality (default 92)")
    parser.add_argument("--strip-metadata", action="store_true", help="do not write the colour profile / EXIF data to the output")
    parser.add_argument("--no-color-management", action="store_true", help="keep pixel values as they are instead of converting to sRGB")
    parser.add_argument("--info", action="store_true", help="print image information instead of writing output")
    parser.add_argument("--list-filters", action="store_true", help="list available filters and parameters")
    parser.add_argument("--list-presets", action="store_true", help="list available presets")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    return parser


def list_filters(out=sys.stdout) -> None:
    for spec in FILTERS.values():
        params = ", ".join(
            f"{p.name}={p.default:g}[{p.min:g}..{p.max:g}]" if p.kind in ("int", "float") else f"{p.name}" for p in spec.params
        )
        out.write(f"{spec.name:<16} {spec.category:<8} {params}\n")


def list_presets(out=sys.stdout) -> None:
    for name, adj in PRESETS.items():
        out.write(f"{name:<14} {' '.join(f'{k}={v}' for k, v in adj.items()) or '(neutral)'}\n")


def process(doc: Document, args: argparse.Namespace) -> Document:
    if args.crop:
        doc.crop(args.crop)
    if args.trim:
        doc.trim()
    if args.rotate % 360:
        if args.rotate % 90 == 0:
            turns = int(args.rotate // 90) % 4
            for _ in range(turns):
                doc.rotate90(clockwise=False)
        else:
            doc.rotate(args.rotate)
    if args.flip:
        doc.flip(horizontal=args.flip == "h")
    if args.resize:
        w, h = args.resize
        if w is None and h is None:
            raise DocumentError("resize needs at least one dimension")
        if w is None:
            w = max(1, round(doc.width * h / doc.height))
        if h is None:
            h = max(1, round(doc.height * w / doc.width))
        doc.resize(w, h)
    if args.scale:
        doc.scale(args.scale)
    adj = preset_adjustments(args.preset) if args.preset else dict(doc.active_layer.adjustments)
    for key, value in args.adjust:
        adj[key] = value
    if args.preset or args.adjust:
        for index in range(len(doc.layers)):
            doc.set_adjustments(adj, index)
    for name, params in args.filter:
        doc.apply_filter(name, **params)
    return doc


def main(argv: list[str] | None = None, out=sys.stdout, err=sys.stderr) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_filters:
        list_filters(out)
        return 0
    if args.list_presets:
        list_presets(out)
        return 0
    if not args.paths:
        parser.print_usage(err)
        err.write("error: at least one input file is required\n")
        return 2

    inputs = list(args.paths)
    output: str | None = None
    if not args.info and not args.out_dir:
        if len(inputs) < 2:
            err.write("error: specify an output file or --out-dir\n")
            return 2
        output = inputs.pop()
        if len(inputs) > 1:
            err.write("error: several inputs need --out-dir\n")
            return 2

    status = 0
    for src in inputs:
        try:
            doc = pfio.open_document(src, color_manage=not args.no_color_management)
        except Exception as exc:
            err.write(f"{src}: cannot open ({exc})\n")
            status = 1
            continue
        if args.info:
            layers = ", ".join(f"{layer.name}[{layer.blend_mode}]" for layer in doc.layers)
            meta = doc.metadata.describe()
            out.write(f"{src}: {doc.width}x{doc.height} px, {len(doc.layers)} layer(s): {layers}" + (f"; {meta}" if meta else "") + "\n")
            continue
        try:
            process(doc, args)
        except (DocumentError, ValueError, KeyError) as exc:
            err.write(f"{src}: {exc}\n")
            status = 1
            continue
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ext = f".{args.format.lstrip('.')}" if args.format else Path(src).suffix
            if ext.lower() == pfio.PROJECT_EXT:
                ext = ".png"
            dest = out_dir / (Path(src).stem + ext)
        else:
            assert output is not None
            dest = Path(output)
        try:
            if pfio.is_project_file(dest):
                pfio.save_project(doc, dest)
            else:
                pfio.save_image(doc, dest, quality=args.quality, metadata=not args.strip_metadata)
        except Exception as exc:
            err.write(f"{src}: cannot save to {dest} ({exc})\n")
            status = 1
            continue
        out.write(f"{src} -> {dest} ({doc.width}x{doc.height})\n")
    return status


if __name__ == "__main__":
    sys.exit(main())

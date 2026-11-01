# PixelForge

A layer-based picture editor written in Python with **Pillow**, **NumPy** and a
thin **Qt (PySide6)** shell. Open a picture, adjust it non-destructively,
select, paint, retouch, add text and shapes, stack layers with masks, groups,
blend modes and effects, shape tones with curves and levels, apply filters,
undo anything, save a project or export the result with its colour profile.

```
uv run pixelforge [picture]            # the editor
uv run pixelforge-cli --help           # batch processing from the shell
```

## The interface

The whole UI is a single surface with floating **glass panels**: each panel is
rendered over a blurred copy of whatever is behind it, with rounded corners,
a soft edge highlight and a faint shadow. Every control is drawn by the app
itself with Pillow, in *device* pixels, and blitted by Qt with the screen's
device-pixel ratio - so on a Retina / HiDPI display the interface **and the
picture** are pixel-sharp, and alignment, hover states, tooltips and scrolling
behave exactly the same on macOS, Windows and Linux. On macOS you also get the
native menu bar with the standard ⌘ shortcuts; the UI is available in English,
German, French and Spanish (View ▸ Language) and in a dark, light or
high-contrast theme (View ▸ Theme). The interface ships its own font (DejaVu
Sans) so it looks the same everywhere, and it is exposed to screen readers as a
tree of panes and controls.

| Panel | What it does |
| --- | --- |
| Top-left | Open, Save project, Export picture, Undo, Redo, Revert, **Compare** with the original (or hold `\`) |
| Top-centre | Options of the active tool (brush size, selection mode, gradient type, …) |
| Top-right | File name, size, unsaved marker, **⋯** menu |
| Left strip | 18 tools in two columns, foreground/background colours, rotate and flip |
| Right panel | **Adjust** – histogram, *Auto*, 16 sliders in three groups · **Presets** – built-in looks and your own · **Layers** – thumbnails, masks, groups, visibility, blend mode, opacity, reorder, merge, effects, lock · **History** – steps and named snapshots |
| Bottom | Zoom controls (click the percentage for a menu), info pill with tool hints, cursor position and colour readout |

### Tools

| Tool | Key | Notes |
| --- | --- | --- |
| Move | `V` | drag the layer, arrow keys nudge, optional auto-select |
| Free Transform | `Ctrl+T` | scale / rotate / move / flip the layer or the selected pixels with handles and numeric fields; Shift keeps the aspect, Alt scales from the centre |
| Marquee / Lasso / Magic Wand | `M` `L` `W` | rectangle or ellipse, freehand, colour-based; Shift adds, Alt subtracts; feather, tolerance, contiguous; marching ants |
| Crop | `C` | handles, aspect ratios, rule-of-thirds, snapping to edges/thirds/grid, Enter/Esc |
| Brush / Eraser | `B` `E` | brush presets with tips (round, square, chalk, spatter, custom from a selection), spacing, angle, roundness, scatter, size/opacity jitter, flow and count; `[` `]` size, Shift-click lines; paints on the layer mask when that is the edit target |
| Clone Stamp / Healing Brush | `S` `J` | Alt-click picks the source; aligned or not, from one or all layers; the healing brush keeps the surroundings' colour and spot-heals without a source |
| Paint Bucket / Gradient | `G` `⇧G` | tolerance fill or whole selection; linear, radial, reflected gradients |
| Text | `T` | system fonts, size, colour, outline; text layers stay editable (double-click in Layers) |
| Shape | `U` | rectangle, ellipse, line with fill and stroke |
| 3D | `K` | rotates (drag), moves (Shift) and scales (Alt) the 3D object of the active layer; double-click opens all settings |
| Eyedropper | `I` | picks the foreground (Alt: background) colour |
| Zoom / Hand | `Z` `H` | plus Space-drag, pinch and ⌘-wheel everywhere |

Selections restrict painting, fills, filters and *Apply Adjustments to Pixels*;
the Select menu offers feather, grow, shrink, border, invert and crop-to-selection.

### Layers, masks, groups and effects

* **Masks** – *Layer ▸ Layer Mask* (or the mask button) adds a reveal-all,
  hide-all or from-selection mask. Click the mask thumbnail (or press `Ctrl+\`)
  to paint on the mask instead of the pixels; a red overlay shows what is
  hidden. Masks can be disabled, inverted, loaded as a selection, applied or
  deleted, and they follow every move, transform, crop and rotation.
* **Adjustment layers** affect everything below them – the Adjust sliders,
  *Curves…* and *Levels…* all edit the adjustment layer non-destructively
  through its mask, opacity and blend mode.
* **Groups** (`Ctrl+G`, `Ctrl+Shift+G`) can be nested, collapsed, masked and
  given effects; *Move Up/Down* steps a layer into and out of a group, *Merge
  Down* on a group flattens it to one layer.
* **Effects** (`fx`) – drop shadow, outer glow and stroke are rendered live and
  stay editable; they are stored with the layer in project files.
* **Tones** – *Image ▸ Curves…* (`Ctrl+M`) edits a master RGB curve and the
  three channels over the layer's histogram; *Levels…* (`Ctrl+L`) has draggable
  black / grey / white input markers and an output range.

### Brushes

The brush button in the options bar of the Brush, Eraser, Clone Stamp and
Healing Brush opens the brush picker: eleven built-in presets (hard and soft
round, airbrush, pencil, marker, calligraphy, square, chalk, spatter,
watercolour, stipple) with a stroke sample each, plus your own. Every preset
has a tip shape, size, hardness, spacing, angle, roundness, scatter, size and
opacity jitter, flow (low flow builds up like an airbrush) and dabs per step.
*Tip from selection* turns the selected pixels into a custom tip; presets are
saved to `~/.pixelforge/brushes.json`.

### 3D objects

*Layer ▸ New 3D Object…* adds a 3D layer - cube, sphere, cylinder, cone, torus
or pyramid - rendered by PixelForge's own software renderer (perspective,
smooth Blinn-Phong shading, adjustable light, anti-aliasing, optional
wireframe). *Extrude Layer to 3D* turns the opaque pixels of any layer - a
text layer, a logo, a selection - into a textured 3D extrusion. The 3D tool
(`K`) rotates, moves and scales the object directly on the canvas with a quick
preview rendered in the background; the object stays editable on its layer and
is stored in project files. 3D and text layers are *smart*: filters and curves
applied to them are kept as a stack and re-applied whenever the layer is
re-rendered, Move and Free Transform change the object rather than its pixels,
and *Layer ▸ Rasterize Layer* converts them to ordinary pixels when you want
to paint on them.

### Menus and sheets

The **Filter** menu (and *Filters…* in the ⋯ menu) offers 30 filters in six
groups with live-preview sheets built from each filter's parameters. **Image**
has straighten (with grid and automatic cropping), resize, canvas size, trim,
flatten and **Auto Enhance**; **View** toggles the grid, rulers and crop
snapping. Projects are saved as `.pfp` files that keep layers, blend modes,
adjustments and the selection; *Export Picture…* flattens to PNG/JPEG/WebP/
TIFF/BMP with remembered quality and matte options. *Apply This Look to Other
Pictures…* batch-exports the current adjustments, and recent files appear in
the File menu and on the start card.

### Safety nets

* **Autosave** – unsaved work is written to `~/.pixelforge/autosave.pfp` every two
  minutes; after a crash the next start offers to recover it.
* **Bounded memory** – undo states store only the pixels that changed (a brush
  stroke costs a small patch, not a full copy of the layer) and the oldest
  states are dropped once the history holds more than about 1.5 GB.
* **Responsive** – large pictures are composited and filter previews are
  computed on a background thread; while a new frame renders, the previous one
  is re-projected so panning and zooming never stall.
* **Colour-managed** – pictures with an embedded ICC profile are converted to
  sRGB when opened so they look right on screen; the profile, EXIF data and
  resolution are kept in project files and written back on export (the export
  sheet has a toggle, the CLI `--strip-metadata`).
* **No dead sessions** – an unexpected error in a tool or filter becomes a notice
  in the info pill (and a traceback on stderr); the document is never lost.
* Window size and position, zoom per picture, export options, grid/rulers/snap
  and the UI language are remembered between sessions.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). Python 3.10+ is
required; uv will fetch one if needed.

```bash
uv sync                          # creates .venv with Pillow, NumPy, PySide6 (+ dev tools)
uv run pixelforge photo.jpg      # start the editor
uv run pixelforge-cli --help     # batch processing
```

Without uv, a plain `pip install .` into any virtual environment works as well
(`python -m pixelforge` / `python picture_editor.py` start the editor).

### Resolution / HiDPI

The UI scale follows the display's device-pixel ratio automatically (2× on a
Retina Mac, whatever Windows/Linux report), and the interface is re-rendered
when the window moves to a screen with a different ratio. If you want the UI
larger or smaller on top of that:

```bash
uv run pixelforge --scale 1.25 photo.jpg   # 25 % larger controls
```

### Standalone app

```bash
uv sync --group build
uv run python packaging/build.py           # → dist/PixelForge (or PixelForge.app on macOS)
```

`packaging/pixelforge.spec` bundles only the Qt modules the editor uses.

The picture itself is never affected by the UI scale: 100 % zoom is always one
image pixel per physical screen pixel, like in Photoshop.

## Keyboard shortcuts

| Keys | Action |
| --- | --- |
| `Ctrl+O` / `Ctrl+S` / `Ctrl+E` | Open / save project / export picture |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo |
| `V M L W C B E G T U I Z H` | Tools (see above) |
| `X` / `D` | Swap / reset colours |
| `Ctrl+A` / `Ctrl+D` / `Ctrl+Shift+I` | Select all / deselect / invert |
| `Delete` · `Alt+Backspace` · `Ctrl+Backspace` | Clear / fill with foreground / background |
| `Ctrl+Shift+N` / `Ctrl+J` / `Ctrl+Shift+E` | New layer / duplicate / merge down |
| `Ctrl+G` / `Ctrl+Shift+G` | Group layer / ungroup |
| `Ctrl+\` | Edit the layer mask / the layer pixels |
| `Ctrl+T` | Free Transform |
| `Ctrl+L` / `Ctrl+M` | Levels / Curves |
| `S` / `J` | Clone Stamp / Healing Brush (Alt-click picks the source) |
| `K` | 3D tool (drag rotates, Shift moves, Alt scales) |
| `Ctrl+[` / `Ctrl+]` | Rotate left / right |
| `Enter` / `Esc` | Apply / cancel the current tool action or sheet |
| `Ctrl++` / `Ctrl+-` / `Ctrl+0` / `Ctrl+1` / `Ctrl+2` | Zoom in / out / fit / 100 % / fit width |
| `Ctrl+'` / `Ctrl+R` | Grid / rulers |
| `\` (hold) | Compare with the original |
| `↑` `↓` over a slider | Nudge by 1 (Shift: 10) |
| `Tab` / `Ctrl+.` | Hide / show the panels |

On macOS use `⌘` instead of `Ctrl`.

## Engine and CLI

`pixelforge.core` is GUI-independent: documents, layers, blend modes,
selections, painting, filters, history and project files. The command line
uses it for batch work:

```bash
uv run pixelforge-cli photo.jpg out.jpg --preset Vivid --adjust temperature=15 --resize 1600x
uv run pixelforge-cli photos/*.jpg --out-dir thumbs --format webp --scale 0.25
uv run pixelforge-cli --list-presets
uv run pixelforge-cli photo.jpg --info                 # size, layers, colour profile, EXIF
uv run pixelforge-cli in.jpg out.jpg --strip-metadata  # export without profile / EXIF
```

```python
from pixelforge.core import Document, open_image, save_image

doc = Document.from_image(open_image("photo.jpg"))
doc.set_adjustments({"contrast": 20, "vibrance": 30}, commit=True)
doc.crop((100, 80, 1500, 1000))
save_image(doc, "out.png")
```

## Project layout

```
pixelforge/
├── core/          image engine (no GUI): document, layers/masks/groups, adjustments, curves,
│                  effects, brushes, painting (strokes, clone/heal, transform), three_d (meshes,
│                  software renderer, extrusions), colorspace, history, io, ...
├── gui/
│   ├── theme.py   colours, fonts, metrics, UI scale
│   ├── glass.py   anti-aliased shapes, frosted-glass panels, vector icons (Pillow)
│   ├── ui.py      widget kit: Column/Row, Label, Button, IconButton, Segmented, Slider,
│   │              Toggle, ListItem, NumberField, Panel
│   ├── surface.py picture rendering, panel compositing, event routing, tooltips
│   ├── tools.py   the 18 interactive tools
│   ├── worker.py  background thread for composites and previews (latest request wins)
│   ├── accessibility.py  accessible-object tree for assistive technology
│   ├── fonts/     bundled DejaVu Sans (see LICENSE-DejaVu.txt)
│   ├── editor.py  panels, commands, actions (toolkit independent)
│   ├── sheets.py  glass dialogs and popovers
│   ├── i18n.py    translations (locales/*.json)
│   ├── qt.py      Qt shell: HiDPI painting, native menus & dialogs, gestures
│   └── app.py     entry point
└── cli.py
tests/             pytest suite: engine, CLI and GUI (Qt offscreen - no display needed), Hypothesis properties
packaging/         PyInstaller spec and build script for a standalone app
.github/workflows  CI: lint, type-check and the test-suite on Linux/macOS/Windows × Python 3.10–3.13
pyproject.toml     project metadata, dependencies, dev dependency group, tool config
uv.lock            locked dependency versions
```

## Development

```bash
uv run pytest                    # everything; the GUI tests use Qt's offscreen platform
uv run pytest -m "not gui"       # engine + CLI only
uv run pytest --cov              # with a coverage report
uv run ruff check . && uv run ruff format .
uv run mypy                      # static types for the package and the (fully annotated) tests
uv run pre-commit install        # lint, format, type-check and run the engine tests on every commit
uv lock --upgrade                # refresh uv.lock
```

See `CHANGELOG.md` for what changed between versions.

The same checks run in GitHub Actions on every push and pull request
(`.github/workflows/ci.yml`): Ruff, mypy and pytest across Ubuntu, macOS and
Windows with Python 3.10 through 3.13.

The GUI tests cover the things that are easy to get wrong in a hand-drawn UI:
alignment of every slider row, tooltip appearance on hover, trackpad and wheel
scrolling over sliders, pinch zoom, crop handle dragging with real mouse
events, modal sheets, keyboard focus, glyph fallback for symbols and rendering
at a 2× device-pixel ratio.

# Changelog

All notable changes to PixelForge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [2.1.0] – Unreleased

### Added
- **Layer masks**: reveal-all / hide-all / from-selection masks on any layer, mask editing target
  (mask thumbnail, `⌘\` / `Ctrl+\`), rubylith overlay, invert / apply / delete / load as selection;
  masks follow moves, transforms, crops and rotations and are saved in project files.
- **Free Transform** (`⌘T` / `Ctrl+T`): scale, rotate, move and flip the active layer – or only the
  selected pixels – with handles, numeric fields and a live preview; premultiplied resampling keeps
  edges clean.
- **Curves** (`⌘M`) with a master RGB curve plus per-channel curves (monotone cubic interpolation,
  histogram backdrop, keyboard nudging) and a graphical **Levels** dialog (`⌘L`) with draggable
  black / grey / white and output markers.
- **Clone Stamp** (`S`) and **Healing Brush** (`J`) tools: aligned / unaligned sampling from the
  active or all layers, spot healing without a source, texture-preserving healing with one.
- **Adjustment layers**: sliders, curves and levels applied non-destructively to every layer below,
  through the layer's mask, opacity and blend mode.
- **Layer effects**: live drop shadow, outer glow and stroke (outside / inside / centre) stored with
  the layer and re-editable at any time; an `fx` button and *Layer Effects…* dialog.
- **Layer groups**: group / ungroup (`⌘G` / `⌘⇧G`), nested groups, group masks and effects,
  collapsible groups in the Layers tab, Move Up / Down steps into and out of groups, Merge Group.
- **Background rendering**: composites of large pictures and all live previews run on a worker
  thread ("latest request wins"); the previous frame is re-projected while a new one renders.
- **Memory-lean history**: undo states keep only the changed region of a layer as a patch, and the
  history is trimmed by a memory budget instead of a fixed step count.
- **Bundled UI font** (DejaVu Sans) for an identical look on every platform
  (`PIXELFORGE_SYSTEM_FONTS=1` prefers the system font).
- **Colour management**: pictures with an embedded ICC profile are converted to sRGB on open; the
  profile, EXIF data (orientation reset) and resolution are kept in project files and written on
  export (toggle in the export sheet, `--strip-metadata` / `--no-color-management` in the CLI).
- **Themes**: light and high-contrast looks next to the dark default (View ▸ Theme).
- **Locales**: French and Spanish translations.
- **Accessibility**: the painted interface is exposed to assistive technology as a tree of panes and
  controls with roles, names, states and on-screen rectangles; notifications are announced.
- **Packaging**: a PyInstaller spec and `packaging/build.py` produce a standalone app bundle.
- **Development**: pre-commit configuration, Hypothesis property-based tests for blend modes,
  selection algebra, curves, transforms and history patches.

### Changed
- The Layers tab's buttons are arranged in two rows (new: adjustment layer, group, mask, effects).
- Tooltips of tools without a single-letter key show their menu shortcut.

## [2.0.0]

- Rewrite on a layer-based engine with a Pillow-rendered glass interface in a Qt shell; tools,
  selections, non-destructive adjustments, presets, filters, snapshots, project files, CLI.

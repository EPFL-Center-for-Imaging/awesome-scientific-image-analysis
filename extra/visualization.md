# 3D Image Visualization software tools

This page provides a general comparison of free and open-source software tools for 3D image rendering.

## Features

| Software | 3D+time | Multichannel | Large data | Volume rendering | Projections (mip) | Isosurface / meshes | Glyphs | Intuitive | Scriptable |
|----------|-------- | ------------ | ---------- | ---------------- | ----------------- | ------------------- | ------ | --------- | ---------- |
| Fiji Volume Viewer | - | - | - | ✅ | ✅ | - | - | ✅ | - |
| Fiji 3D Viewer     | ✅ | ✅ | - | ✅ | - | ✅ | - | ✅ | - |
| Napari             | ✅ | ✅ | ✔️ | ✔️ | ✅ | ✅ | ✅ | ✅ | ✅ |
| PyVista            | - | ✔️ | ✔️ | ✅ | - | ✅ | ✅ | ✔️ | ✅ |
| Neuroglancer       | ✅ | ✅ | ✅ | ✔️ | ✔️ | ✅ | ✔️ | - | ✔️ |
| Paraview           | ✅ | ✔️ | ✔️ | ✅ | - | ✅ | ✅ | - | ✔️ |

✅ Yes ✔️ Sort of

## Strengths & Limitations

**Fiji Volume Viewer**

- ✅ Ideal for Fiji users
- ✅ Good control over the 3D rendering
- 🔴 No glyphs or overlays (masks, points, vectors...)
- 🔴 4D (3D+time or multichannel) not supported (?)
- 🔴 Not controllable programmatically

**Napari**

- ✅ Ideal for nD: 3D+time, multichannel
- ✅ Ideal for overlays: masks, points, vectors...
- ✅ Controllable programmatically
- 🔴 No fine control over the transfer function

**PyVista**

- ✅ Ideal for reproducible visualizations in Python
- ✅ Good control over the 3D rendering
- ✅ Desktop or web-based
- 🔴 Not as interactive as other tools

**Neuroglancer**

- ✅ Ideal for Zarr and large images
- ✅ Visualizations can be shared simply with a URL
- 🔴 Not as intuitive as other tools
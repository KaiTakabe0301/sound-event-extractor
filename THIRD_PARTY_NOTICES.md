# Third-Party Notices

The source code of this project is licensed under the MIT License (see
[LICENSE](LICENSE)). The standalone application built from this repository
additionally bundles the third-party components listed below. This file is
included in the release archives.

## Components bundled in the standalone application

### Python (CPython runtime)
- License: Python Software Foundation License (PSF-2.0)
- https://docs.python.org/3/license.html

### Qt / PySide6
- License: GNU Lesser General Public License v3.0 (LGPL-3.0-only)
- Copyright (C) The Qt Company Ltd.
- https://www.gnu.org/licenses/lgpl-3.0.html / https://pypi.org/project/PySide6/
- The Qt libraries are included as separate, unmodified shared libraries and
  can be replaced by the user, as permitted by LGPL §4. Qt Multimedia bundles
  its own FFmpeg libraries built under LGPL-2.1-or-later.

### FFmpeg executable (bundled via imageio-ffmpeg)
- License: GNU General Public License (GPL build; configured with
  `--enable-gpl --enable-libx264`)
- Copyright (c) 2000-2024 the FFmpeg developers
- https://www.gnu.org/licenses/gpl-3.0.html
- The ffmpeg executable is an independent program invoked as a separate
  process (subprocess); it is aggregated with, not linked into, this
  application. Source code: https://ffmpeg.org/download.html — binary
  provenance and build scripts: https://github.com/imageio/imageio-ffmpeg

### TensorFlow, tf-keras, TensorFlow Hub
- License: Apache License 2.0
- Copyright The TensorFlow Authors / Google LLC
- https://www.apache.org/licenses/LICENSE-2.0

### NumPy
- License: BSD-3-Clause (with vendored components under 0BSD / MIT / Zlib /
  BSD-2-Clause)
- https://numpy.org/doc/stable/license.html

### imageio-ffmpeg (Python package)
- License: BSD-2-Clause
- https://github.com/imageio/imageio-ffmpeg

### setuptools (pkg_resources)
- License: MIT
- https://github.com/pypa/setuptools

### PyInstaller bootloader
- License: GPL-2.0-or-later with the PyInstaller Bootloader Exception, which
  permits distributing applications of any license built with PyInstaller
- https://pyinstaller.org/en/stable/license.html

## Model downloaded at runtime (not distributed)

### YAMNet
- License: Apache License 2.0 (Google)
- https://tfhub.dev/google/yamnet/1
- The model (~20 MB) is downloaded directly by the end user from TensorFlow
  Hub on first analysis and cached locally. It is not distributed with this
  repository or with the release archives.

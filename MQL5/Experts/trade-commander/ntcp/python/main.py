"""
NTCP Entry Point — launches the PyQt6 cockpit GUI.

Run: python main.py  (from ntcp/python/ directory)
 or: python -m python.main  (from ntcp/ directory)
"""

import os
import sys
from pathlib import Path

# Ensure the ntcp/ parent is on sys.path so relative imports within
# the 'python' package resolve correctly when running as a script.
_PACKAGE_DIR = Path(__file__).resolve().parent
_NTCP_DIR = _PACKAGE_DIR.parent
if str(_NTCP_DIR) not in sys.path:
    sys.path.insert(0, str(_NTCP_DIR))

# Register PyTorch's DLL directory so Windows resolves c10.dll correctly,
# regardless of import order between torch and PyQt6.
if sys.platform == "win32":
    import torch  # noqa: F401, E402
    _torch_lib = Path(torch.__file__).parent / "lib"
    if _torch_lib.is_dir():
        os.add_dll_directory(str(_torch_lib))
else:
    import torch  # noqa: F401, E402

from PyQt6.QtWidgets import QApplication  # noqa: E402

from python.gui.main_window import NTCPMainWindow  # noqa: E402

STYLE_PATH = _PACKAGE_DIR / "gui" / "style.qss"


def main() -> None:
    app = QApplication(sys.argv)

    # Load dark theme
    if STYLE_PATH.exists():
        app.setStyleSheet(STYLE_PATH.read_text(encoding="utf-8"))

    window = NTCPMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""Entry point for MAFilter NN App (Qt6). Run from MAFilter: python python/main.py"""

from __future__ import annotations

if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from python.app import main
    main()
else:
    from .app import main

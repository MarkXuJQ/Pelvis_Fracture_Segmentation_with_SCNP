"""Training package marker for project utilities and experiment entrypoints."""

import sys


# Keep the local package importable through both `training` and `Training`
# so the copied nnU-Net experiment modules continue to work on Windows while
# remaining self-contained inside this repository.
sys.modules.setdefault("Training", sys.modules[__name__])

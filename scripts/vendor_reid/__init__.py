"""Vendored ReID model definitions, copied verbatim from the research project.

Source: /home/inomjon/projectAI/phd/dissertatsiya2/tools/reid/
        reid_model.py, vendor/osnet.py, vendor/resnet_ibn_a.py

Copied rather than imported so that exporting a checkpoint does not depend on
that tree staying where it is, or on its `sys.path` conventions. Nothing here is
used at runtime - `scripts/export_reid.py` needs it to rebuild the graph once,
and the deployed pipeline then loads only the resulting ONNX file.

Do not edit these. If the research model definition changes, re-copy it, or the
exported weights will silently land in the wrong layers.
"""

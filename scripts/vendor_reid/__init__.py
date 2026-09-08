"""Vendored ReID model definitions, copied verbatim from the research project.

Source: /home/inomjon/projectAI/phd/dissertatsiya2/tools/reid/
        reid_model.py, boshlar.py, vendor/osnet.py, vendor/resnet_ibn_a.py
Refreshed 2026-09-08 from integration/model_def/, which is the copy that
exported the deployed OSNet checkpoint.

Copied rather than imported so that exporting a checkpoint does not depend on
that tree staying where it is, or on its `sys.path` conventions. Nothing here is
used at runtime - `scripts/export_reid.py` needs it to rebuild the graph once,
and the deployed pipeline then loads only the resulting ONNX file.

WHY IT HAD TO BE REFRESHED. The previous copy predated the training heads.
`ReIDNet` had no `head` argument, so a circle-loss or AdaFace checkpoint - which
every current one is - rebuilt into a graph with no `head` submodule and its
`head.*` keys came back "unexpected". `export_reid.py` refuses on an
inexact state_dict, correctly, so the deployed model had to be exported out of
band by the research tree instead. It is exportable from here again.

The head takes part in TRAINING ONLY. Inference reads the BNNeck feature, so
which head a checkpoint carries changes nothing about the exported graph - it
only has to be constructible for the weights to load exactly.

Do not edit these. If the research model definition changes, re-copy it, or the
exported weights will silently land in the wrong layers.
"""

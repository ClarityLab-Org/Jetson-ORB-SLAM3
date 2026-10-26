#!/usr/bin/env python3
"""Export the CosPlace ResNet-50 (512-d) place-recognition network to ONNX, for the paper's
CNN loop closure (Sec. 3.3; plan/paper_runbook.md, Part R3).

Run this on a PC with PyTorch (not needed on the Jetson):
    pip install torch torchvision onnx onnxscript
    python3 export_cosplace_onnx.py cosplace_r50_512.onnx

Then copy the .onnx to the Jetson and build the TensorRT engine there (engines are specific to
one GPU and TensorRT version):
    /usr/src/tensorrt/bin/trtexec --onnx=cosplace_r50_512.onnx \
        --saveEngine=cosplace_r50_512.fp16.trt --fp16

Input/output match what src/CNNLoopDetector.cpp feeds and reads: one 1x3x224x224 float image
(grayscale replicated to RGB, ImageNet-normalised) -> a 512-d descriptor (L2-normalised by the
C++ code). Weights are the published CosPlace model from https://github.com/gmberton/CosPlace.
"""
import sys

import torch


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "cosplace_r50_512.onnx"
    model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                           backbone="ResNet50", fc_output_dim=512)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        d = model(dummy)
    assert d.shape == (1, 512), d.shape
    # Weights stored inside the file (no external .data file): the paper notes externally
    # referenced weights are what made ONNX-Runtime's TensorRT EP hang on the Orin.
    # PyTorch >= 2.6 writes weights to a side file by default; older versions lack the flag.
    kw = dict(input_names=["input"], output_names=["descriptor"],
              opset_version=17, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, out, external_data=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, out, **kw)
    print("wrote %s (input 1x3x224x224, output 1x512)" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

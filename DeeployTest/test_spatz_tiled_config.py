"""Test configuration for Snitch platform."""

# Snitch platform supports gvsoc, vsim simulators

KERNEL_TESTS = [
    "Kernels/FP32/MatMul",
    "Kernels/FP32/TopK/TopK128L2048",
    "Kernels/FP32/Gather",
    "Kernels/FP32/Softmax/Regular2D",
    "Kernels/FP32/TopKAttention/TopKAttention_1.64.2048_k10",
]

MODEL_TESTS = []
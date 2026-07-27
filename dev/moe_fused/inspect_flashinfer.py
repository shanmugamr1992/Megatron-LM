#!/usr/bin/env python3
"""Stage-2 pre-flight: what does the installed flashinfer actually expose?

Prints version, the presence + signature of every MoE entry point mcore or vLLM
could use, and whether the JIT cubin cache packages are installed (their absence
means a multi-minute JIT compile on first call, which must be budgeted for
inside CUDA-graph warmup).
"""
import importlib
import inspect

import torch


def show(mod, name):
    fn = getattr(mod, name, None)
    if fn is None:
        print(f"  {name:<34} MISSING")
        return
    try:
        sig = str(inspect.signature(fn))
    except (TypeError, ValueError):
        sig = "<no signature>"
    print(f"  {name:<34} PRESENT {sig}")


def main():
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    try:
        print(f"device {torch.cuda.get_device_name()} "
              f"sm{''.join(str(x) for x in torch.cuda.get_device_capability())}")
    except Exception as exc:
        print(f"device query failed: {exc}")

    try:
        import flashinfer
    except Exception as exc:
        print(f"IMPORT FAILED: flashinfer: {type(exc).__name__}: {exc}")
        return
    print(f"\nflashinfer {getattr(flashinfer, '__version__', '?')} at {flashinfer.__file__}")

    for pkg in ("flashinfer_jit_cache", "flashinfer_cubin", "flashinfer_python"):
        try:
            m = importlib.import_module(pkg)
            print(f"  {pkg:<22} installed ({getattr(m, '__version__', '?')})")
        except Exception:
            print(f"  {pkg:<22} NOT installed")

    from flashinfer import fused_moe
    print(f"\nflashinfer.fused_moe at {fused_moe.__file__}")
    print("entry points:")
    for name in (
        "cutlass_fused_moe",
        "trtllm_bf16_moe",
        "trtllm_bf16_routed_moe",
        "trtllm_fp8_block_scale_moe",
        "trtllm_fp4_block_scale_moe",
        "RoutingMethodType",
        "WeightLayout",
        "GatedActType",
        "ActivationType",
    ):
        show(fused_moe, name)

    for enum_name in ("RoutingMethodType", "WeightLayout", "GatedActType"):
        enum = getattr(fused_moe, enum_name, None)
        if enum is None:
            continue
        try:
            members = [m for m in dir(enum) if not m.startswith("_")]
            print(f"\n{enum_name} members: {members}")
        except Exception as exc:
            print(f"\n{enum_name} introspection failed: {exc}")

    print("\nall public names in flashinfer.fused_moe:")
    print("  " + ", ".join(n for n in dir(fused_moe) if not n.startswith("_")))


if __name__ == "__main__":
    main()

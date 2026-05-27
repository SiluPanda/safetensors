#!/usr/bin/env python3
"""Wall-time + memory benchmark for `from_pretrained` onto MPS.

Compares loading WITH vs WITHOUT this PR by running the *same* script in two
environments:

  without PR:  pip-installed `safetensors`
  with PR:     `maturin develop` build of this repo

It prints the `safetensors` / `transformers` versions so you can tell which
build is active, then times one
`AutoModelForCausalLM.from_pretrained(..., device_map="mps")` and reports wall
time plus several memory signals.

On MPS, `ps -o rss` undercounts badly: MTLBuffer/IOKit memory is accounted
outside the process resident set, and mmap file-cache pages aren't stably
resident under pressure. So we sample three things and keep their peaks:

  rss     - `ps -o rss` (kept for reference; misleading on MPS)
  phys    - `proc_pid_rusage` ri_phys_footprint (Activity Monitor's "Memory")
  wired   - system-wide `vm_stat` wired pages (MTLBuffers tend to land here)
  comp    - system-wide compressed pages (memory pressure)
  swap    - `sysctl vm.swapusage` used

Run it in-situ (don't close your other apps) on a model sized near your RAM so
swap pressure is realistic.

Usage:
    python bench_mps_load.py <hf-repo-or-path>
    python bench_mps_load.py <model> --dtype float16 --device-map mps
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import subprocess
import threading
import time

import torch

GB = 1024**3


# --- per-process memory signals -------------------------------------------


def rss_bytes(pid: int) -> int:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
    return int(out.strip()) * 1024  # macOS `ps` reports RSS in KB


class _RUsageInfoV2(ctypes.Structure):
    # <libproc.h> rusage_info_v2, truncated after the field we read.
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


_RUSAGE_INFO_V2 = 2
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_libproc.proc_pid_rusage.restype = ctypes.c_int


def phys_footprint_bytes(pid: int) -> int:
    """`ri_phys_footprint`: the footprint Activity Monitor shows for the
    process, including IOKit/GPU (MTLBuffer) and compressed memory."""
    info = _RUsageInfoV2()
    rc = _libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V2, ctypes.byref(info))
    if rc != 0:
        return 0
    return int(info.ri_phys_footprint)


# --- system-wide memory signals -------------------------------------------


def swap_used_bytes() -> int:
    # vm.swapusage: "total = 3072.00M  used = 1234.50M  free = 1837.50M ..."
    out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode()
    tok = out.split("used =")[1].split()[0]  # e.g. "1234.50M"
    return int(float(tok[:-1]) * {"K": 1024, "M": 1024**2, "G": 1024**3}[tok[-1]])


def vm_stat_bytes() -> dict[str, int]:
    """Parse `vm_stat`. Returns wired/compressed/free in bytes."""
    out = subprocess.check_output(["vm_stat"]).decode()
    page = 4096
    first = out.splitlines()[0]
    if "page size of" in first:
        page = int(first.split("page size of")[1].split("bytes")[0].strip())
    vals = {}
    for line in out.splitlines()[1:]:
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        rest = rest.strip().rstrip(".")
        if rest.isdigit():
            vals[key.strip()] = int(rest) * page
    return {
        "wired": vals.get("Pages wired down", 0),
        "compressed": vals.get("Pages occupied by compressor", 0),
        "free": vals.get("Pages free", 0),
    }


class PeakSampler(threading.Thread):
    """Polls per-process and system memory every 50ms, keeping the max."""

    def __init__(self):
        super().__init__(daemon=True)
        self.pid = os.getpid()
        self.peak_rss = 0
        self.peak_phys = 0
        self.peak_swap = 0
        self.peak_wired = 0
        self.peak_comp = 0
        self._stop = False

    def run(self):
        while not self._stop:
            self.peak_rss = max(self.peak_rss, rss_bytes(self.pid))
            self.peak_phys = max(self.peak_phys, phys_footprint_bytes(self.pid))
            self.peak_swap = max(self.peak_swap, swap_used_bytes())
            vm = vm_stat_bytes()
            self.peak_wired = max(self.peak_wired, vm["wired"])
            self.peak_comp = max(self.peak_comp, vm["compressed"])
            time.sleep(0.05)

    def stop(self):
        self._stop = True
        self.join()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model", help="HF repo id or local model directory")
    ap.add_argument(
        "--dtype", default="bfloat16", help="torch dtype (default: bfloat16)"
    )
    ap.add_argument(
        "--device-map", default="mps", help="from_pretrained device_map (default: mps)"
    )
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    import safetensors
    import transformers
    from transformers import AutoModelForCausalLM

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("MPS not available")

    print(
        f"safetensors {safetensors.__version__}   transformers {transformers.__version__}"
    )
    print(f"model: {args.model}   dtype: {args.dtype}   device_map: {args.device_map}")
    vm0 = vm_stat_bytes()
    print(
        f"at start: swap {swap_used_bytes() / GB:.2f}G  "
        f"wired {vm0['wired'] / GB:.2f}G  compressed {vm0['compressed'] / GB:.2f}G\n"
    )

    dtype = getattr(torch, args.dtype)
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()

    sampler = PeakSampler()
    sampler.start()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    sampler.stop()

    nparams = sum(p.numel() for p in model.parameters())
    print(
        f"  from_pretrained {dt:7.2f}s   params {nparams / 1e9:4.1f}B\n"
        f"  peak  rss {sampler.peak_rss / GB:5.1f}G  phys {sampler.peak_phys / GB:5.1f}G  "
        f"wired {sampler.peak_wired / GB:5.1f}G  comp {sampler.peak_comp / GB:5.1f}G  "
        f"swap {sampler.peak_swap / GB:5.1f}G\n"
        f"  MPS driver {torch.mps.driver_allocated_memory() / GB:5.1f}G  "
        f"current {torch.mps.current_allocated_memory() / GB:5.1f}G"
    )


if __name__ == "__main__":
    main()

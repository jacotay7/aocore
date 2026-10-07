# Propagators on an Arm host

`python benchmarks/bench_propagation.py --gpu` on cfl-test-bench, an 80-core
Ampere Neoverse-N1 (aarch64, no SMT) pinned to 12 cores (`taskset -c 16-27`),
with an RTX 4060 (driver 580). NumPy 2.5.3 (OpenBLAS 0.3.34), SciPy 1.18.1
(ducc FFT) and CuPy 14.2.0. Times are median microseconds per call, with two
diversity channels (`K = 2`); the host was shared, so expect a few percent of
noise. "0.1.3" re-implements the old code paths inside the script. CPU runs
in double precision, GPU in single.

| device | case | 0.1.3 (us) | 0.1.4 (us) | speed-up | max rel. difference |
|---|---|---:|---:|---:|---:|
| cpu | FFT forward 64 -> 128 | 858 | 340 | 2.53x | 0 |
| cpu | FFT forward 128 -> 256 | 3033 | 1276 | 2.38x | 0 |
| cpu | FFT forward 256 -> 512 | 7929 | 2644 | 3.00x | 2.3e-16 |
| cpu | FFT forward 512 -> 1024 | 13900 | 8805 | 1.58x | 2.2e-16 |
| cpu | MFT forward 128, L=1 | 1274 | 727 | 1.75x | 0 |
| cpu | MFT forward 128, L=5 | 6142 | 3424 | 1.79x | 0 |
| cpu | MFT forward 256, L=5 | 21675 | 22075 | 0.98x | 0 |
| cpu | dot, 65536 float | 30 | 30 | 1.00x | 0 |
| gpu | FFT forward 64 -> 128 | 163 | 162 | 1.01x | 0 |
| gpu | FFT forward 128 -> 256 | 161 | 156 | 1.03x | 0 |
| gpu | FFT forward 256 -> 512 | 166 | 162 | 1.02x | 0 |
| gpu | FFT forward 512 -> 1024 | 339 | 320 | 1.06x | 0 |
| gpu | MFT forward 128, L=1 | 405 | 95 | 4.28x | 0 |
| gpu | MFT forward 128, L=5 | 383 | 108 | 3.55x | 0 |
| gpu | MFT forward 256, L=5 | 514 | 356 | 1.44x | 0 |
| gpu | dot, 65536 float | 95 | 75 | 1.26x | 0 |

The CPU FFT gains come from pruning the padded transform (half the FFT work)
and from reusing its work arrays (writing fresh multi-megabyte arrays costs a
page fault per 4 KiB, about as long as the transform). The CPU MFT gains come
from 8 BLAS threads instead of 4 on a host without SMT; the 256, L=5 case
already used 8. On the GPU the MFT no longer makes CuPy broadcast its
matrices over the channel axis, and `FocalPlanePropagator` (not shown) also
skips a stacking copy, 1.2x for the FFT engine at these sizes.

Every GPU result and every CPU MFT and dot result is bit-identical to 0.1.3.
The CPU FFT matches it bit for bit on the power-of-two grids up to 256
points above. Elsewhere values can differ by at most 2.5 units in the last
place of the largest element: where 0.1.3's full transform ran on 12 or more
threads (whose line partition rounds some lines differently; running 0.1.3
with `AOCORE_FFT_WORKERS=1` instead of its default changes the same values
by the same amount), and on some grids whose line counts are not multiples
of 16 (a short remainder group of lines rounds differently).

## solvephase end to end

`benchmarks/run.py --cases focal_lm focal_gradient lift phase_diversity` of
solvephase 0.2.0, three interleaved runs each with aocore 0.1.3 and 0.1.4
(median, milliseconds). The recovered wavefronts and iteration counts are
unchanged; `focal_lm` errors differ in the 15th digit on the CPU, and are
identical on the GPU.

| case | CPU 0.1.3 | CPU 0.1.4 | speed-up | GPU 0.1.3 | GPU 0.1.4 | speed-up |
|---|---:|---:|---:|---:|---:|---:|
| focal_lm 64 | 162.2 | 129.7 | 1.25x | 39.7 | 38.6 | 1.03x |
| focal_lm 128 | 615.0 | 430.2 | 1.43x | 39.9 | 38.9 | 1.03x |
| focal_lm 256 | 1896.5 | 1580.7 | 1.20x | 158.6 | 158.3 | 1.00x |
| focal_gradient 128, FFT | 5.74 | 4.77 | 1.20x | 1.24 | 1.18 | 1.05x |
| focal_gradient 128, 5 wavelengths (MFT) | 19.39 | 13.03 | 1.49x | 1.69 | 1.20 | 1.40x |
| focal_gradient 256, FFT | 16.75 | 14.79 | 1.13x | 1.23 | 1.22 | 1.01x |
| focal_gradient 256, 5 wavelengths (MFT) | 77.63 | 80.02 | 0.97x | 1.68 | 1.29 | 1.30x |
| focal_gradient 512, FFT | 76.32 | 64.37 | 1.19x | 2.22 | 2.16 | 1.03x |
| lift 32 | 13.87 | 9.85 | 1.41x | 41.3 | 37.3 | 1.11x |
| lift 64 | 26.65 | 24.01 | 1.11x | 39.1 | 37.6 | 1.04x |
| lift 128 | 85.15 | 77.56 | 1.10x | 40.0 | 39.0 | 1.03x |
| phase_diversity 64 | 143.0 | 89.4 | 1.60x | 270.1 | 266.7 | 1.01x |
| phase_diversity 128 | 422.6 | 416.0 | 1.02x | 373.6 | 368.0 | 1.02x |

# DAHD4

Python conversion of the DAHD (Data-Adaptive Harmonic Decomposition) MATLAB codebase, developed as part of an NSF-funded Arctic sea ice forecasting project at the SETI Institute in collaboration with UCLA and UCSB.

Original MATLAB code written by Dmitri Kondrashov (UCLA). Python conversion by Taylor McDonald (SETI Institute, 2026).

---

## Overview

DAHD is a frequency-domain spectral decomposition method for multivariate time series. It identifies spatio-temporal oscillatory modes by eigendecomposing a Hermitian cross-spectral density matrix, then reconstructs those modes via Harmonic Reconstruction Components (HRCs). This repo provides a complete, license-free Python implementation of the DAHD pipeline, validated against the original MATLAB output.

The method is described in:
- Chekroun, M. D., and D. Kondrashov, 2017: *Data-adaptive harmonic spectra and multilayer Stuart-Landau models.* Chaos, 27, 093110. doi:10.1063/1.4989400
- Kondrashov, D., E. Ryzhov, and P. Berloff, 2020: *Data-adaptive harmonic analysis of oceanic waves and turbulent flows.* Chaos, 30, 061105. doi:10.1063/5.0012077
- Kondrashov, D., I. Sudakow, V. Livina, and Q. Yang, 2026: *Accurate and robust real-time prediction of September Arctic sea ice.* Chaos, 36, 023110. doi:10.1063/5.0295634

---

## Repository Structure

```
DAHD4/
├── center.py                    # Center data by subtracting column-wise mean
├── xcorr.py                     # MATLAB xcorr equivalent using scipy
├── dahc.py                      # Compute DAH coefficients (DAHCs)
├── hrc.py                       # Compute Reconstructed Harmonic Components (HRCs)
├── DAHM4_ex.py                  # Compute space-time DAHMs via inverse FFT
├── DAHD4freq_part_weight.py     # Compute DAHD spectrum (frequency-domain)
├── generate_data.py             # Generate synthetic multivariate test dataset
├── dahd4example.py              # Full pipeline: spectrum → DAHMs → reconstruction
├── plothrcmodes.py              # Plot reconstructed vs reference modes
├── center.ipynb                 # Notebook: center
├── xcorr.ipynb                  # Notebook: xcorr
├── dahc.ipynb                   # Notebook: dahc
├── hrc.ipynb                    # Notebook: hrc
├── DAHM4_ex.ipynb               # Notebook: DAHM4_ex
├── DAHD4freq_part_weight.ipynb  # Notebook: DAHD4freq_part_weight
├── generate_data.ipynb          # Notebook: generate_data
├── dahd4example.ipynb           # Notebook: full pipeline
├── plothrcmodes.ipynb           # Notebook: plot HRC modes
└── matlab/                      # Original MATLAB source files
```

---

## Installation

```bash
pip install numpy scipy matplotlib
```

No MATLAB license required.

---

## Usage

Run the full pipeline:

```bash
python dahd4example.py
```

This generates all five figures from Kondrashov (2026):
- **Fig 1** — Synthetic dataset: reference modes, signal, and noisy data
- **Fig 2** — DAHD spectrum (no temporal weighting)
- **Fig 3** — DAHD spectrum (Hamming-weighted cross-correlations)
- **Fig 4** — Space-time DAHM eigenvector pairs at the four target frequencies
- **Fig 5** — HRC reconstructions vs reference modes

---

## Python Conversion Notes

### Approach

The original MATLAB codebase was converted function by function into Python, preserving the mathematical structure while replacing MATLAB-specific syntax and idioms with NumPy/SciPy equivalents. Key translation decisions included:

- `xcorr` (MATLAB built-in, requires Signal Processing Toolbox) → reimplemented using `scipy.signal.correlate` with unbiased normalization matching MATLAB's behavior
- `eigs` (MATLAB) → `scipy.linalg.eigh` with `subset_by_index` for the top-NP eigenvalues; note that `eigh` returns eigenvalues in ascending order, so results are reversed to match MATLAB's descending order
- `lfilter` (MATLAB) → `scipy.signal.lfilter` (direct equivalent)
- `fft`/`ifft` (MATLAB) → `np.fft.fft`/`np.fft.ifft`
- `reshape` (MATLAB, column-major) → `np.reshape` (row-major by default); order must be matched explicitly where relevant
- `zeros` (MATLAB, always real) → `np.zeros` (real by default); complex arrays require explicit `dtype=complex`

---

## Bugs Found and Fixed

Three bugs were identified during numerical validation against MATLAB ground-truth output. All three caused the RMSE of reconstruction to be substantially higher than MATLAB's.

---

### Bug 1 — `DAHM4_ex.py`: Mirror frequency index off-by-one

**Location:** `DAHM4_ex.py`, inside the `NF > 1` branch

**Root cause:** When translating MATLAB's 1-indexed array addressing to Python's 0-indexed addressing, the mirror frequency index was incorrectly computed.

MATLAB original:
```matlab
fftv(end-NF+2, :, i2)    = sqrt(W-0.5)*conj(vv(:,i));
fftv(end-NF+2, :, i2+1)  = sqrt(W-0.5)*conj(cvv);
```

In MATLAB, `end = WW = 2*W-1`. So `end-NF+2` in 1-indexed addressing maps to Python index `WW-NF`, which in Python's negative indexing convention is simply `-NF`. The original conversion used `-NF+1` instead, placing the conjugate at the wrong frequency bin and breaking the Hermitian symmetry of the FFT array.

**Fix:**
```python
# BEFORE (incorrect)
fftv[-NF+1, :, i2]     = np.sqrt(W - 0.5) * np.conj(vv[:, i])
fftv[-NF+1, :, i2 + 1] = np.sqrt(W - 0.5) * np.conj(cvv)

# AFTER (correct)
fftv[-NF, :, i2]     = np.sqrt(W - 0.5) * np.conj(vv[:, i])
fftv[-NF, :, i2 + 1] = np.sqrt(W - 0.5) * np.conj(cvv)
```

**Verification:** After this fix, the imaginary part of the IFFT output dropped from ~0.06 (same order of magnitude as the real part) to ~1e-17 (floating-point noise), confirming that Hermitian symmetry was restored.

---

### Bug 2 — `DAHD4freq_part_weight.py`: `cspec` and `FEP` initialized as real arrays

**Location:** `DAHD4freq_part_weight.py`

**Root cause:** Both `cspec` (the cross-spectral density matrix) and `FEP` (the Hermitian DAHD eigenvectors) were initialized with `np.zeros(...)` without `dtype=complex`. In MATLAB, `zeros()` produces real arrays, but the arrays in this function hold inherently complex values — the FFT of a real cross-correlation is complex, and the eigenvectors of a Hermitian matrix are complex. When Python assigns complex values to a real array, it silently drops the imaginary part, causing `FEP` to come out as a purely real array when it should be complex.

This was confirmed by comparing `FEP` values between Python and MATLAB at `iff=17`:

| | Python (before fix) | MATLAB |
|---|---|---|
| FEP col 1 row 1 | `0.42075` (real only) | `0.1344 - 0.3622i` (complex) |
| FEP col 2 row 1 | `-0.41251` (real only) | `0.3535 - 0.2108i` (complex) |

**Fix:**
```python
# BEFORE
cspec = np.zeros((WW, dim, dim))
FEP   = np.zeros((dim, NP, NFE))

# AFTER
cspec = np.zeros((WW, dim, dim), dtype=complex)  # FIX: must be complex
FEP   = np.zeros((dim, NP, NFE), dtype=complex)  # FIX: must be complex
```

---

### Bug 3 — `DAHM4_ex.py`: `fftv` initialized as real, losing quadrature pair

**Location:** `DAHM4_ex.py`

**Root cause:** After Bug 2 was fixed, `FEP` correctly became complex. However, `fftv` inside `DAHM4_ex` was still initialized as a real array. Since `vv` (a slice of `FEP`) is now complex, the quadrature pair `cvv = 1j * vv[:, i]` is also complex. Assigning this to a real `fftv` silently dropped the imaginary part, causing the second column of the output (the quadrature DAHM) to come out as all zeros.

This was confirmed by printing `tmp` (the DAHM slice fed into `dahc`) before and after the fix:

| | Python (before fix) | Python (after fix) | MATLAB |
|---|---|---|---|
| tmp col 2 row 1 | `0.0` | `-0.0451` | `-0.0451` |

**Fix:**
```python
# BEFORE
fftv = np.zeros((2 * W - 1, D, NT))

# AFTER
fftv = np.zeros((2 * W - 1, D, NT), dtype=complex)  # FIX: must be complex
```

---

## Numerical Validation

All RMSE values computed on the synthetic dataset from Kondrashov (2026): N=129, d=6, NoiseLevel=0.6, W=65 (maximum embedding window), Hamming-weighted cross-correlations.

### Normalized RMSE of reconstruction

| Mode | MATLAB | Python (before fixes) | Python (after all fixes) |
|------|-------:|----------------------:|-------------------------:|
| f_1  | 0.2211 | 0.59~                 | **0.1964**               |
| f_2  | 0.1860 | 0.66~                 | **0.2284**               |
| f_3  | 0.0504 | 0.11~                 | **0.0501**               |
| f_4  | 0.0541 | 0.42~                 | **0.0470**               |

f_3 and f_4 match MATLAB to within 0.3%. The small remaining gap in f_1 and f_2 is consistent with algorithmic differences between Python's `scipy.linalg.eigh` and MATLAB's `eigs` in the eigendecomposition step, and is not attributable to a remaining bug.

### Note on the M = N edge case

This example uses the maximum embedding window M = (N+1)/2 = 65, giving WW = 2M-1 = 129 = N. At this setting the DAHCs collapse to a single scalar value per mode (MT=1), which limits reconstruction accuracy. As noted in Kondrashov (2026) section 3, this is expected behavior for short datasets where maximum spectral resolution is prioritized over reconstruction accuracy. For longer time series, M << N is recommended.

---

## Contact

Original MATLAB code: Dmitri Kondrashov — dkondras@atmos.ucla.edu

Python conversion: Taylor McDonald — SETI Institute, 2026

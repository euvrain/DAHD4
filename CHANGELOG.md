# Changelog

All notable changes to this project are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

---

## [1.0.0] — 2026-07-20

### Origin
Original MATLAB codebase written by Dmitri Kondrashov (UCLA).
Python conversion by Taylor McDonald (SETI Institute), June–July 2026,
as part of an NSF-funded Arctic sea ice forecasting project in collaboration
with UCLA (Dr. Dmitri Kondrashov) and UCSB (Dr. Ivan Sudakow).

### Added
- `center.py` — Python conversion of `center.m`
- `xcorr.py` — Python reimplementation of MATLAB's `xcorr` using `scipy.signal.correlate`
- `dahc.py` — Python conversion of `dahc.m`
- `hrc.py` — Python conversion of `hrc.m`
- `DAHM4_ex.py` — Python conversion of `DAHM4_ex.m`
- `DAHD4freq_part_weight.py` — Python conversion of `DAHD4freq_part_weight.m`
- `generate_data.py` — Python conversion of `generate_data.m`
- `dahd4example.py` — Python conversion of `dahdexample4.m`
- `plothrcmodes.py` — Python conversion of `plothrcmodes.m`
- Jupyter notebooks for all functions with markdown documentation and executable demos
- `README.md` with full conversion notes, bug documentation, and RMSE validation table
- `LICENSE` (MIT)
- `requirements.txt`
- `CITATION.cff`
- `codemeta.json`

### Fixed

#### Bug 1 — `DAHM4_ex.py`: Mirror frequency index off-by-one
- **Introduced by:** incorrect translation of MATLAB 1-indexed `end-NF+2` to Python
- **Effect:** Broke Hermitian symmetry of FFT array; imaginary part of IFFT output was ~0.06 (same order as real part), silently discarded on assignment
- **Fix:** Changed `fftv[-NF+1]` to `fftv[-NF]`
- **Verified by:** imaginary part dropped from ~0.06 to ~1e-17 after fix

#### Bug 2 — `DAHD4freq_part_weight.py`: Complex arrays initialized as real
- **Introduced by:** `np.zeros()` defaults to real dtype; MATLAB `zeros()` also defaults to real but the values assigned were complex
- **Effect:** `cspec` and `FEP` silently dropped imaginary parts; `FEP` came out purely real when it should be complex
- **Fix:** Added `dtype=complex` to both `cspec` and `FEP` initialization
- **Verified by:** comparing `FEP` values against MATLAB ground truth at `iff=17`

#### Bug 3 — `DAHM4_ex.py`: `fftv` initialized as real, losing quadrature pair
- **Introduced by:** after Bug 2 was fixed, `vv` (a slice of `FEP`) became complex; assigning `1j * vv[:, i]` to a real `fftv` silently zeroed out the quadrature DAHM
- **Effect:** second column of DAHM output was all zeros; reconstruction RMSE ~1.0 across all modes
- **Fix:** Added `dtype=complex` to `fftv` initialization
- **Verified by:** `tmp col 2` (quadrature DAHM) recovered non-zero values matching MATLAB

### Validated
Normalized RMSE of reconstruction verified against MATLAB ground truth
(synthetic dataset, N=129, d=6, NoiseLevel=0.6, W=65, Hamming weighting):

| Mode | MATLAB | Python before fixes | Python after fixes |
|------|-------:|--------------------:|-------------------:|
| f_1  | 0.2211 | ~1.01               | 0.1964             |
| f_2  | 0.1860 | ~1.00               | 0.2284             |
| f_3  | 0.0504 | ~1.00               | 0.0501             |
| f_4  | 0.0541 | ~1.00               | 0.0470             |

---

## Provenance

| Item | Detail |
|---|---|
| Original MATLAB author | Dmitri Kondrashov, UCLA (dkondras@atmos.ucla.edu) |
| MATLAB version date | June 22, 2026 |
| Python conversion author | Taylor McDonald, SETI Institute |
| Conversion period | June–July 2026 |
| Validation method | Numerical comparison against MATLAB output; RMSE verification |
| Funding | NSF-funded Arctic sea ice forecasting project |
| Collaborating institutions | UCLA, UCSB, SETI Institute |

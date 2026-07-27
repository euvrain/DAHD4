# Changelog

All notable changes to this project are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

---

## [1.1.0] — 2026-07-27

### Added
- `runMSLM_ice.py` — Python conversion of `runMSLM_ice.m`: full forecasting driver that loads NSIDC Sea Ice Index regional data, computes DAHD/HRC harmonic reconstructions, and drives the MSLM ensemble forecast
- `MSLM_FCST.py` — Python conversion of `MSLM_FCST.m`: multi-level stochastic Stuart-Landau model ensemble forecast engine
- `runMSLM_ice.ipynb`, `runMSLM_ice_test.ipynb` — notebook versions of the forecasting pipeline, producing the 8 diagnostic/forecast figures (regional forecasts, pan-Arctic reconstruction, September SIE ensemble prediction, etc.)
- `robustness_test.py`, `robustness_test.ipynb` — training-window sensitivity analysis: re-runs the September sea-ice-extent forecast on 10/12/14/16/18-year training windows to check forecast stability
- `validate_translation.py` — automated validation suite covering (1) unit tests of `dahc`/`hrc`/`MSLM_FCST` physical/mathematical constraints, (2) data-alignment correlation checks against MATLAB reference output (`ROBSF.mat`, `ROBSFF.mat`), and (3) forecast-quality/physical-sanity checks (boundary continuity, ensemble spread, September SIE plausibility)

### Fixed

#### Bug 4 — `runMSLM_ice.py`: year-offset bug in raw data alignment
- **Introduced by:** the row offset used to align the NSIDC regional-data spreadsheet to the model's 2006 epoch origin was hardcoded (`1404`), based on an assumption that the spreadsheet's first year-column was 1979. The actual spreadsheet begins at 1978 (47 year-columns), so the hardcoded offset pointed one full year early (at 2005 instead of 2006).
- **Effect:** every downstream anomaly/reconstruction was computed against the wrong calendar year, throwing off any comparison to the reference MATLAB output.
- **Fix:** compute the offset from the spreadsheet's actual header row: `OFFSET_2006 = (2006 - first_year) * 52`, giving `1456` instead of `1404`.
- **Verified by:** correlation of recomputed anomalies against the reference `ROBSRF.mat` observations rose from 0.0–0.3 (effectively uncorrelated) to 0.87–0.98 across all 7 regions after the fix.

#### Bug 5 — `runMSLM_ice.py`: seasonal-cycle-corrected `RXRC` reconstruction pass was never implemented
- **Introduced by:** the MATLAB source (`runMSLM_ice.m`, lines ~487–536) computes `RXRC` in two passes — a first pass that is immediately discarded, and a second pass (`RXRC = RX00`) that adds the seasonal-mean ice extent back in, clips negative absolute extent values to physical zero, subtracts the seasonal mean back off, and re-subtracts the seasonal cycle. The original Python conversion only implemented the discarded first pass.
- **Effect:** `RXRC`/`RXC`/`RXRMC`/`RXMC` retained the seasonal cycle when plotted against de-seasonalized `ROBSRF` observations, producing a visible region-dependent jump at the forecast boundary.
- **Fix:** implemented the full add-clip-subtract-reseasonalize sequence (`RX00`) matching the MATLAB second pass.

#### Bug 6 — `runMSLM_ice.py`: forecast-boundary index off-by-one
- **Introduced by:** `NET` (MATLAB's 1-indexed last-training-week row number) was translated to 0-indexed Python as `NA - 1` instead of `NET - 1`, confusing a related but distinct embedding-window index with the training-window boundary.
- **Effect:** the forecast boundary used for the regional-forecast figure and for locating the September SIE prediction was offset by ~76 weeks (roughly 1.5 years), pulled from the wrong point in the reconstructed series — including, in some cases, the amplified values near the reconstruction edge produced by `hrc.m`'s edge-tapering correction.
- **Fix:** `NET_plot = NET - 1`.

#### Bug 7 — Figure 6: tick/label count mismatch
- **Effect:** `matplotlib` raised `ValueError: The number of FixedLocator locations (2) ... does not match the number of labels (3)` when the visible x-range excluded one of the preset tick positions.
- **Fix:** filter tick positions and labels as a matched pair, keeping only ticks that fall within the plotted range, before calling `set_xticks`/`set_xticklabels`.

#### Bug 8 — September SIE prediction: seasonal-mean not added back before reporting
- **Effect:** the September sea-ice-extent point prediction and ensemble were reported in de-seasonalized anomaly units instead of absolute ice-extent units (M km²), making the forecast numbers not directly comparable to the observational reference values.
- **Fix:** added the seasonal-mean ice extent (`icemean`) back to the anomaly before reporting the September prediction, consistent with the `RX00` sequence in Bug 5.

#### Bug 9 — `robustness_test.py`: same off-by-one embedding-window bug as Bug 6
- **Effect:** identical to Bug 6, but independently present in the standalone robustness-test script (`NET_s = NA_s - 1` instead of `NET_s = n_weeks - 1`), producing forecast boundaries offset by ~76 weeks in every training-window variant tested.
- **Fix:** `NET_s = n_weeks - 1`.

#### Bug 10 — `robustness_test.py`: seasonal mean double-counted
- **Effect:** `RX0_s` added `icemean_sub` back and clipped to zero (per Bug 5's sequence) but never subtracted it back off, while the September-reporting step separately added `icemean_sub` again — double-counting the seasonal mean and inflating every reported September SIE value.
- **Fix:** added the missing subtract-back-off step so `RX0_s` matches the exact MATLAB `RX00`→`RX0` sequence.

#### Bug 11 — `robustness_test.py`: training-window subsetting used the wrong end of the data (dominant bug)
- **Introduced by:** shorter "training length" variants were built by taking the *first* N years of data starting at 2006 (`iceextobs[:n_weeks, :]`) rather than the *last* N years ending at the same real-data cutoff used by the full run. This meant a 10-year training-window variant, for example, was actually forecasting into the middle of the historical record (~2016) instead of the target period (~2024).
- **Effect:** September SIE predictions across training-window lengths were physically nonsensical (~12–13 M km², versus a ~4.40 M km² reference).
- **Fix:** `start_idx = KT_TRUNC - n_weeks; iceext_sub = iceextobs[start_idx:KT_TRUNC, :]`, holding the forecast target period fixed and varying only the training length, with a calendar-time lookup independent of the truncated series length.
- **Verified by:** re-running all five training-window lengths (10/12/14/16/18 years) post-fix gives September SIE = 4.66, 4.50, 4.52, 4.53, 4.53 M km² respectively — consistent with each other and with the ~4.40 M km² reference, versus ~12–13 M km² before the fix.

### Validated
`validate_translation.py` (new in this release) ran 15 automated checks spanning unit-level physics constraints, MATLAB data-alignment correlation, and forecast-quality sanity checks. All 15 passed on the current build.

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
| Validation method | Numerical comparison against MATLAB output; RMSE verification; automated validation suite (v1.1.0) |
| Funding | NSF-funded Arctic sea ice forecasting project |
| Collaborating institutions | UCLA, UCSB, SETI Institute |

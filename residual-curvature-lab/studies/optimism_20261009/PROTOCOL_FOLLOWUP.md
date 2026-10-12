# Follow-up protocol: refined theory, remedies and PARSEC-L

Written after RESULTS.md, before running `scripts/followup.py`. Same six banked states, operators and
gates as PROTOCOL.md. New candidates and readings below; nothing in the first protocol changes.

## Additions

- **PARSEC-L operator** (`parsec_l`). The damping-calibrated rank-16 core with exact eigenpairs (the
  offline-tested operator):
  - A⁻¹ = F0 (I − W diag(κ/(1+κ)) Wᵀ) F0ᵀ per tensor, with β = 1/(μ·scale) and μ as for exact GGN;
  - its r and predicted shares are computed like the other operators'.
- **Fixed-step candidates (no search):** native SOAP's step size s applied to the SOAP map and to
  PARSEC-L. This is the step each optimizer actually takes; momentum RHS and gradient RHS.
  - s comes from the stored native step: −s P u must equal `raw_soap` (gate: cosine ≥ 1 − 1e-10).
  - Added before the run, because norm blend is to become standard: `parsec_l_blend1`, PARSEC-L's fixed
    step rescaled globally to SOAP's fixed-step norm (norm_blend = 1).
- **Step-size policies** for every searched candidate, measured on panels 2–4:
  - `same`: as stored (search on the construction batch);
  - `second`: search on panel 1;
  - `corrected` (gradient RHS, label-free operators only): search on the construction batch with
    objective L_B(θ + s d) + s·O, where O = tr(A⁻¹Σ̂)/B for d = −A⁻¹g_B;
  - `oracle`: search on panels 2–4 (an optimistic upper bound).
- **Independent mechanism check:** ρ₁₂ = ḡ₁₂ᵀD / g_BᵀD from panels 1–2; measured share from panels 3–4.
- **Momentum staleness:** metric cosine between the momentum and the held-out gradient,
  cos_A = ḡᵀA⁻¹m / √(S · mᵀA⁻¹m), with S cross-fitted.

## Pre-declared readings

1. **EF ceiling:** g_bᵀ(F_b + μ_F Ω_b)⁻¹ g_b ≤ 1 for every tensor b at every state (up to 1e-12).
2. **Independent mechanism:** Spearman(2ρ₁₂ − 1, share on panels 3–4) over the 24 gradient points of
   the first protocol ≥ 0.6.
3. **Second-batch step size:** the mean held-out decrease on panels 2–4 is higher with `second` than
   with `same` for exact-GGN gradient steps in ≥ 5 of 6 states. Also reported: the count over all
   searched candidates.
4. **Optimism-corrected step size:** `corrected` beats `same` on panels 2–4 in ≥ 12 of the 18
   label-free gradient points.
5. **Drift:** cos_A(m, ḡ) is lower at GGN-25 and GGN-50 than at SOAP-25 and SOAP-50 (pairwise by
   step) for each of SGD, SOAP map and exact GGN.
6. **PARSEC-L:**
   - (a) r(soap_map) < r(parsec_l) < r(exact_ggn) in ≥ 5 of 6 states;
   - (b) "keeps SOAP's protection" if its fixed-step momentum share is at least SOAP's fixed-step
     momentum share − 0.1 in ≥ 5 of 6 states;
   - also reported: its fixed-step held-out decrease against SOAP's.
   - (c) the same two comparisons for `parsec_l_blend1`.

Exploratory (post hoc, labelled as such): calibration restricted to the label-free operators (SGD,
SOAP map, exact GGN, PARSEC-L); absolute held-out decreases for momentum steps.

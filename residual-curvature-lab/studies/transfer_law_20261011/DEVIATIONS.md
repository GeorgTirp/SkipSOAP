# Deviations

## 1. Part B stopped at the Wiener gate (2026-10-11, 20:18:59Z; written before the rerun)

- **What failed:** on the first state (`searched_soap-10`), the Wiener construction with φ ≡ 1 differed from
  `exact_ggn`'s direction by 1.96e-8 relative. The protocol gate is ≤ 1e-8 (section 6). All 12 reference gates
  (same, second, oracle × four operators) passed at 0.0. No Part B result was written.
- **Diagnosis** (same state, per tensor):
  - **Both directions solve the same symmetrized system to roundoff:** relative residual ‖Ax + g‖/‖g‖ is
    1.1e-14 for the reference Cholesky solve and 3.8e-14 for the eigenbasis form.
  - **The difference sits in one tensor,** `input.projection.weight` (784 parameters, 6.0e-8). There,
    cond(A) = 1.6e10 and cond(Ω) = 7.1e6.
  - **Elsewhere it is 2e-10 or less.**
  - The forward difference between two backward-stable solvers is bounded by about cond(A) × residual, roughly
    1e-4 here. A forward-agreement gate of 1e-8 is therefore not attainable in general for this block.
- **Handling (gate not loosened):**
  - **Reading B2** (Wiener step) is reported as *not evaluated, gate failed*.
  - **The rest of Part B:** the script is rerun with this gate recorded instead of stopping the run, so that the
    readings protected by passing gates (B1 and the reported step-size policies) are produced. All other gates
    still stop the run.
  - **Wiener values** are shown only as labelled diagnostics, together with each state's identity difference
    and residuals.
  - Precedent: stage D's failed SOAP gate (that arm reported as failed, the study continued).
- **For the owner to decide:** whether a well-posed replacement gate (residual ≤ 1e-12 against the linear
  system) should be adopted for future Wiener steps. If so, B2 can be re-evaluated from the stored values.

## 2. Part B rerun stopped in a reported-only policy (20:23:15Z; written before the second rerun)

- **What stopped:** at the third state (`searched_soap-50`), the lab's `search_candidate` raised for
  `wiener_search` (the Wiener direction with same-batch search). Along that direction the construction loss was
  lowest at the largest grid point, 2¹⁶ native radii. The two completed states passed every gate at 0.0, with
  Wiener identity differences of 2.0e-8 and 1.5e-8 (DEVIATIONS #1). No results file was written.
- **Handling:** the boundary rule that Part A pre-declared (record the case, exclude it, report the count) is
  applied to the two reported-only searched policies, `wiener_search` and `exact_ggn_xc`. A boundary search gives
  no step and is flagged `boundary`. The gated reference searches (same, second, oracle) keep raising. No
  pre-declared reading depends on these two policies.

## 3. Bug in Part B's Ô for `soap_map`, found after the run (20:47Z; fixed and rerun)

- **Symptom:** γ̂ = 1.00 at all six `soap_map` points (Ô/q ≤ 0.02), although r(soap_map) = 0.16–0.55 in the optimism
  study.
- **Cause:**
  - The `soap_map` direction is check 2c's solve of (μΩ)δ = −g, so q = g_BᵀA⁻¹g_B and Ŝ carry a factor 1/μ.
  - Ô used the follow-up's explicit inverse Ω⁻¹, which is unscaled. There the scale did not matter, because r
    used the same inverse for O and S.
  - Ô was therefore μ times too small relative to q and Ŝ, and the γ̂ correction was switched off for this
    operator.
  - The other three operators use one consistent A for the direction and for Ô.
- **Affected:** `gamma_x` and `gamma_b` at the six `soap_map` points only. Unaffected: the gates, `same`, `second`,
  `oracle`, `slope_x`, the Wiener diagnostics, `exact_ggn_xc` and all of Part A.
- **Fix,** as the protocol defines Ô (with the A of the direction): Ô is multiplied by q / g_BᵀA⁻¹g_B, the
  operator's scale. This is 1/μ for `soap_map` and 1 (to roundoff) otherwise.
- **Record:** the pre-fix outputs are kept as `results/steps_prefix_bug.json` and `results/summary_prefix_bug.json`.
  Both B1 values are reported.

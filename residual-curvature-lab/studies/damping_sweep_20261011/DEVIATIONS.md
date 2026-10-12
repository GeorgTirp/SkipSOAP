# Deviations and clarifications

## 1. Part 2, lag-10 prediction (written 2026-10-11 ~14:30Z, before the OU check was run)

The protocol compares the measured lag-10 correlation of y = w_kᵀF̂⁻¹(θ − θ̄) against "the mean predicted ρ₁₀",
where ρ is the model autocorrelation of the iterate x itself. But y is the deviation of x from its EMA, not x.
The EMA removes the slow part of x, so y decorrelates faster than x whenever the correlation time is comparable to
the EMA window (about 50 steps at d = 0.98). Under the top-free brake every top direction has correlation time
≥ 64 steps (η'λ ≤ 1/64), so the literal comparison would test the wrong quantity.

**Clarification, fixed before any part-2 result exists:**
- **O1 lag criterion (primary):** measured pooled lag-10 correlation against the model's lag-10 correlation of
  the residual process r = x − EMA_d(x). It is computed exactly from ρ as Σ_m A(m)ρ_|10−m| / Σ_m A(m)ρ_|m|, with
  A the autocorrelation of the residual filter h₀ = d, h_j = −(1−d)d^j. It is pooled like the measurement:
  Σ predicted covariance at lag 10 / Σ predicted variance.
- **Literal reading (reported alongside):** the protocol's unweighted mean ρ₁₀ of x, with its own O1 outcome. If
  the two O1 outcomes differ, both are reported and the primary one is used for the Kalman decision.
- **Unchanged:** the spread criterion (stationary EMA factor, bins, [0.5, 2]).
- **Reported only:** the same readings restricted to t ≥ 100, where the bias-corrected EMA is close to its
  stationary form.

## 2. Part 2, implementation detail

- SOAP's ε is not in `configs/mini_vit_allocation.yaml`. It is read from the checkpoint's optimizer `param_groups`
  (default 1e-8).
- F̂ is built from the checkpoint state with the lab's `rotate`. It is checked against upstream
  `SOAP.project_back` on one random vector per tensor (relative difference ≤ 1e-12, else the script stops).

## 3. Part 2, first launch stopped at a self-check (14:24:52Z; no result produced)

- **What stopped:** the script's own round-trip check F̂⁻¹(F̂v) = v (tolerance 1e-10, set in the code, not in the
  protocol) failed on the first tensor of the first state (`input.cls_token`, 3.1e-7). The forward check against
  upstream `project_back` passed (2.2e-16).
- **Cause:** upstream SOAP computes its eigenbases in float32 (`third_party/soap/soap.py` lines 346–350,
  388–396), so |QᵀQ − I| ≈ 3–5e-7 in the checkpoints. The inverse used Qᵀ·√d/s, which is only F̂⁻¹ to that
  precision. The condition number of √d (up to about 5e3) amplifies the gap in the round trip, up to 2e-5.
- **Change (before any result):** F̂⁻¹ is now an exact LU solve of the dense F̂. It is checked by its residual
  ‖F̂F̂⁻¹v − v‖/‖v‖ ≤ 1e-12 per tensor, a tighter and well-posed check. The forward check is unchanged.
- **Effect on readings:** none expected. The Qᵀ-based coordinates differ from exact ones by about 1e-7 relative
  (cond-amplified up to 2e-5), far below the factor-2 spread tolerance.

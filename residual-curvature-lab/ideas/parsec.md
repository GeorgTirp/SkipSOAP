# PARSEC: insights, design rules and idea status

Living document, started 8 October 2026. It banks what the lab has *learned* about PARSEC
(PARtial-Spectrum Eigen-Correction) and the status of every idea. The original proposals, with their
derivations and kill criteria, are in [PARSEC_ideas.md](PARSEC_ideas.md); this file does not replace
them.

**Companion file.** Batch-noise theory and evidence (noise scales, transfer, pooling, queue effects) are kept
separately in [batch_noise.md](batch_noise.md).

**How to maintain this file.**

- Update it after every study that touches PARSEC: the design rules, the idea-status table and the
  changelog.
- Every claim names its source study. Every claim carries one evidence level:
  - **T**: training result, fresh seeds 1–3, tuned baselines;
  - **L**: local, fixed-state result (one step at saved states);
  - **H**: hypothesis, not tested.
- Never delete a negative result or a killed idea; mark it and say why.
- A new idea gets the next free ID and a first test and kill criterion before anyone builds it.

Workload behind all numbers unless stated: ViT width 16, 2 blocks, MNIST 28x28, batch 512, CPU
float64. Training rows: seeds 1–3, 500 consumed steps, each arm tuned on seed 0.

## 1. What PARSEC-L is (current form)

Per tensor b, with SOAP's frozen factor F0_b (SOAP's preconditioner is P_b = F0_b F0_bᵀ), damping
μ = max(γ tr C / D, 1e-8), β = 1/(μ · scale) and SOAP's effective momentum u:

    Δ_b = −η F0_b [ I − Σ_{k≤r} (κ_k / (1 + κ_k)) w_k w_kᵀ ] F0_bᵀ u_b

(w_k, κ_k) are the top eigenpairs of the whitened GGN block K_C,b = β F0_bᵀ C_bb F0_b. This is SOAP's
native step, shrunk along the r directions where the curvature is large compared with SOAP's own
metric. With r = 0 it is plain SOAP. With exact eigenpairs it equals the capacity screen's
damping-calibrated `teacher_r` core.

**Growing noise scales stop full Newton early; a fractional power keeps a one-step gain (2026-10-21, [L], 6
replayed CIFAR runs, steps 10–400).**
- **Setup:**
  - SOAP and top-free, seeds 1–3, replayed bitwise.
  - Steps −A^{−ν}z in SOAP-whitened coordinates, with A the damped per-tensor GGN (4,096 reference examples) and
    ν ∈ {0, ¼, ½, ¾, 1}.
  - Noise from 43 batch gradients of 1,024. Decreases are scored on an independent evaluation batch, with the step
    size selected on another batch.
- **Noise scales:**
  - Newton's K₁ grows from 1,340–2,170 to 17,300–20,400 examples between steps 10 and 400. At B = 2,048 its
    noise-to-signal ratio goes from 0.63–1.01 to 8.1–9.5.
  - K₁ > K½ holds everywhere. K½ > K₀ fails in two step-10 states (N1 not supported).
- **Break:** full Newton stops paying at B = 2,048 from step 25–50 (N3), and ¾ beats 1 in 35 of 36 states.
- **Fractional optimum:** interior in 35 of 36 states, about ¾ early and ½ late (F1, F2). Late, ν = ½ gives 1.3–2.0×
  the one-step decrease of SOAP's direction.
- **Theory:**
  - [batch_noise.md](batch_noise.md)'s tuned-step law gets the trend right and puts the optimum within one grid step
    (P1, 32 of 36).
  - But its optimum is about ¼ too high, and it overstates the gains (1.5–3.8× at ν = ½).
- **Identity:** the family is the geodesic from SOAP's metric to the damped GGN, so ν = ½ is their matrix geometric
  mean.

([../studies/noise_scale_20261021/RESULTS.md](../studies/noise_scale_20261021/RESULTS.md))

**For curvature estimates the data queue does not matter, but staleness and pooling do (2026-10-20, [L], top-free
replays, seeds 1–3, steps 50–400).**
- **Visit phase:** under random reshuffling, per-example loss rises with the time since the last visit, by about
  0.04–0.08 nats from step 100 on (placebo flat). The declared reading fails (absent at step 50).
- **Sources:** curvature from the next 8, the last 8 or 8 random batches at current weights gives the same subspaces
  (differences ≤ 0.007 capture) and the same steps.
- **Staleness and pooling:** probing at old weights costs 0.03–0.09 capture, while pooling 8 batches gains
  0.014–0.073 over 1.
- **Trailing pool vs fresh batch:** the trailing pool loses to a fresh batch early and wins after about step 250. The
  pre-declared floor-vs-drift rule put the break-even too high.
- **Probe-row Krylov:** 8-step CG on probe-row Grams roughly doubles the one-step held-out decrease of the plain
  whitened step (median 2.1×).

([../studies/queue_curvature_20261020/RESULTS.md](../studies/queue_curvature_20261020/RESULTS.md))

**Late in training the GGN spectrum is a fixed power law with rotating eigenvectors; early, both change
(2026-10-20, [L], 6 replayed CIFAR runs, steps 10–500).**
- **Setup:** SOAP and top-free, seeds 1–3, replayed bitwise. The raw per-tensor GGN (top 200) was measured on a fixed
  batch at 14 states.
- **Late (steps 250–500):** the spectrum's shape changes by only 0.03–0.08 (log units) over 10–50 steps. A
  10-step-old top-16 subspace captures 0.84–0.88 of the current top-16 curvature, a 50-step-old one 0.70–0.81.
  Verdict for both optimizers: *stable spectrum, rotating eigenvectors*.
- **Early (steps 10–100):** both change. The shape distance over 10→60 is 1.1–1.6 and the capture 0.16–0.32. The
  power law straightens from α ≈ 1.8–2.1 (kinked) to α ≈ 0.9–1.1 (R² 0.99).
- **Batch noise vs rotation:** at fixed weights, another batch's top-16 subspace captures 0.93–0.999. The
  noise/drift ratio stays at 0.02–0.6, so rotation, not batch noise, limits a cached subspace.
- **Faster rotation:** top-free rotates faster than SOAP, consistent with its 3× larger lr.

([../studies/spectral_rotation_20261020/RESULTS.md](../studies/spectral_rotation_20261020/RESULTS.md))

**A spectrum alone cannot precondition the bulk; a few deflated Krylov products can (2026-10-14, [L], 12 CIFAR
top-free states, steps 10–100).**
- **Spectrum:** the whitened per-tensor GGN spectrum is roughly a power law, but broken early. The top 16 fall as
  about k^−1, ranks 17–200 about twice as steeply (α_bulk 1.8–2.1 at steps 10–25), and it straightens by step 100.
  The probes measure λ₁ and the trace accurately (within 0.91–1.02 and 0.97–1.02, medians). A single law fitted from
  them underestimates λ₁₇ by 1.6–2.4× (P1, P2 not supported).
- **Spectrum only (owner's power-law idea, N25):** a power-law diagonal assigned by the probe diagonal closes a
  median 6% of the gap to the exact per-tensor GGN step, the same as the exact (Jacobi) diagonal (K1). That is a
  baseline, not a ceiling for every diagonal rule.
- **Directions by products (N26):** the top-16 probe subspace plus deflated Krylov closes medians of 35%, 51% and 70%
  of the gap with 2, 4 and 8 products (67–76% at 8, K2). On held-out loss the 4- and 8-product steps match or beat
  the exact step (medians 0.075, 0.078 vs 0.068; top-free 0.024).
- **Failed variants:**
  - Chebyshev with the probe-predicted interval: never a descent step (K3).
  - 5-step full-GGN CG on 256 examples: equals the exact step on held-out loss at median, but beats top-free in only
    7 of 9 states (K4). It costs 0.07 s per product against a 0.22 s SOAP step.

([../studies/bulk_krylov_20261014/RESULTS.md](../studies/bulk_krylov_20261014/RESULTS.md))

**At matched tuning, the OT metric adds only a little over GGN probes; tuning the brake and lr gives most of the gain
(2026-10-14, [T], CIFAR, 1,000 steps, seeds 1–3).**
- **Setup:** GGN-probe top-free tuned on seed 0 over lr {3e-2, 9e-2} × c {64, 256, 1024} (plus lr 0.27). It selected
  lr 9e-2, c = 256, EMA 0.98, interior in both. That matches OT's c = 4096 in brake strength (about GGN c = 450).
- **F1:** OT top-free beats tuned GGN top-free in every seed: EMA 1000 0.460 vs 0.466 (−0.007, −0.008, −0.002).
  By the declared reading *the OT metric matters*, but the margin is about the seed-to-seed spread, and the test is
  a sign test on three seeds.
- **F2:** tuned GGN top-free vs tuned SOAP is mixed: 0.466 vs 0.483 (−0.026, +0.001, −0.025).
- **Reading:** tuned GGN top-free reaches 74% of OT top-free's gain over SOAP. Both top-free arms cost about 1.65×
  SOAP's wall time per step, probe steps included.

([../studies/ggn_ot_fairness_20261014/RESULTS.md](../studies/ggn_ot_fairness_20261014/RESULTS.md))

**Tuned OT-probe top-free beats tuned SOAP at 1,000 steps (2026-10-14, [T], CIFAR, seeds 1–3).**
- **Setup:** both arms tuned on seed 0. OT selected lr 9e-2 and c = 4096; SOAP selected lr 3e-2.
- **Result:** EMA at step 1000 0.460 vs 0.483 and raw 0.484 vs 0.608, both better in every seed. Accuracy 0.841 vs
  0.828.
- **The gain rests on the lr:** on seed 0 at matched lr, SOAP is slightly better at 3e-2 (0.475 vs 0.483), while OT
  is far better at 9e-2 (0.468 vs 0.544). The OT-metric brake lets a 3× larger lr stay productive.
- **Open:** 2,000 steps, where SOAP itself used 9e-2 well; and metric vs c (N23, answered by `ggn_ot_fairness` above).

([../studies/ot_tuned_1000_20261014/RESULTS.md](../studies/ot_tuned_1000_20261014/RESULTS.md))

**Top-free's gain does not survive a longer run (2026-10-13, [T], CIFAR, 2,000 steps, seeds 1–5); the owner stops
building on this PARSEC idea.**
- **Setup:** every arm retuned on seed 0. All selected lr 9e-2, the top edge after one extension.
- **Result:** top-free + EMA 0.462 vs SOAP + EMA 0.457, lower in only 1 of 5 seeds (R1 mixed). Raw losses tied
  (0.488 vs 0.489). Accuracy higher in all 5 (0.847 vs 0.844).
- **Top-free + N9:** mixed against both.
- **Along the run:** top-free leads SOAP's raw loss mid-run (step 500: 0.761 vs 0.844), and the lead is gone by
  step 1500.
- **Reading:** the 500-step advantage was a mid-run effect, not a better end point.
- **Owner's decision (13 October):** "this is not enough to argue for building on this exact PARSEC idea". The
  remaining run and the OT fairness control were stopped or cancelled.

([../studies/long_run_20261013/RESULTS.md](../studies/long_run_20261013/RESULTS.md))

**PARSEC-L is not a GGN approximation (2026-10-13, [L], ladder at CIFAR top-free states): "close the GGN gap" is
formally dropped.**
- **Where the gap lives:** 56–80% of the gap between top-free's step and the exact per-tensor GGN step, on the
  damped per-tensor quadratic at steps 10–50, lies outside a rank-16 top space plus momentum and gradient (9 of 9).
- **The smaller parts:**
  - better coefficients in the held space: 10–28%;
  - fresher probes: 3–17%;
  - exact instead of probe top directions: ≤ 3%.
- **Held out:** at step 10, a reduced solve in the fresh subspace would recover 60–87% of the exact GGN's
  single-step held-out advantage. That fades by step 25–50.
- **N21:** the ten-step staleness along the step is mostly selection (11 of 18 points), with global sharpening
  around step 60.

([../studies/ladder_mechanism_20261013/RESULTS.md](../studies/ladder_mechanism_20261013/RESULTS.md))

**Probe variants (2026-10-13, [T] but confounded): OT-metric probes give the best PARSEC-L so far; probes from another
batch do nothing.**
- **OT probes:** top-free with OT-metric probes (the lab's graph-Wasserstein pullback sketched by probes, c = 4096
  tuned on seed 0) beats GGN-probe top-free (c = 64) in every seed at raw steps 50 and 200 and at EMA 500 (0.521 vs
  0.541; SOAP 0.554).
  - **Confound:** GGN top-free's c was never tuned on CIFAR. The control `ggn_c_control_20261013` is queued.
  - **Scale:** OT's curvature is about 9× smaller, so OT at c = 4096 brakes about 7× harder than GGN at c = 64.
- **Probes from an independent batch:** mixed (0.537 vs 0.541).

([../studies/probe_variants_20261013/RESULTS.md](../studies/probe_variants_20261013/RESULTS.md))

**Offline tests of N9, triggered refresh and I11 (2026-10-13, [L], CIFAR states): the probe-GGN basis is a much
better SOAP basis; no refresh trigger beats the clock; top-free brakes its cleanest directions.**
- **N9:** at SOAP states (steps 10–250, seeds 1–3), the eigenbasis of the probe factors S_L = Σ G_jG_jᵀ,
  S_R = Σ G_jᵀG_j (s = 1 sampled-label per-example gradients) with a diagonal recovers 12–18% of the exact per-tensor
  GGN's advantage over SOAP's map at steps 10–50. SOAP's own basis with the exact diagonal recovers 2–5% (9 of 9).
  - The probe-estimated diagonal is as good as the exact one.
  - The gain is largest for the head (47–77%) and the attention query and key weights.
  - Post hoc, with step sizes chosen on a separate batch: on held-out loss, the deployable version beats SOAP's map
    in 9 of 9 states at steps 10–50, and recovers 31–73% of the exact GGN's held-out advantage at steps 10–25. Not
    at steps 100–250.
  - Diagonal models in any Kronecker basis recover under 20% of the per-tensor GGN advantage. The Frobenius-nearest
    Kronecker basis of the exact block is worse than the probe basis.
- **Refresh (I10, I15a, N8):** the benefit of refreshing SOAP's basis now grows with its age in 9 of 9 windows
  (median relative benefit 0.08 at ages 1–5 to 0.22 at ages 21–30). SOAP's single QR step leaves its basis well
  short of its own factors' eigenbasis.
  - No signal beats the trivial "steps since refresh": factor rotation passes 6 of 9 windows, probe ρ rise 5, and
    probe spread 0.
- **I11:** top-free's braked directions have *higher* SNR than random whitened directions, and brake strength rises
  with SNR (both readings reversed, 0 of 9).
  - In weight matrices, the 16 braked directions (0.4% of the dimensions) carry 29–57% of the population gradient's
    signal energy and 8–32% of its noise.
  - Top-free moves along them at about 1/60 of the static SNR-optimal response.
  - The static noise-filtering explanation of top-free's gain is refuted.
([../studies/basis_refresh_snr_20261013/RESULTS.md](../studies/basis_refresh_snr_20261013/RESULTS.md))

**SOAP ± SN on CIFAR-10 (2026-10-12, [T] for the loss, descriptive otherwise): SN (c = 1) makes SOAP worse and
does not improve the quadratic model.**
- **Workload:** that of the CIFAR scale-up below. SOAP at lr 1e-2 against SOAP + SN; SN clips all 36 branch weights
  from step 50 on, and its tuned lr is also 1e-2. Seeds 1–3.
- **Reproduction:** the SOAP units reproduce the CIFAR scale-up's SOAP units bitwise.
- **Loss:** SN is worse at raw steps 50 and 200 and at EMA step 500 in every seed (0.580 vs 0.554). The raw weights
  end level, so most of the gap comes from a smaller EMA gain. SN costs 2× per step.
- **SN hypotheses (pre-declared; none shown):**
  - distance preservation is better in 3 of 4 parts, mainly in shallow blocks;
  - full-GGN model accuracy is not better;
  - cross-tensor coupling along the step is **higher** (17–19 vs 13–15);
  - network nonlinearity at a matched logit step is about equal (1.2% at RMS logit step 0.1, 11–17% at 1.0);
  - ten-step curvature persistence is about equal.
- **Descriptive:**
  - SN slows representation drift and deformation at every depth (drift after block 6: 0.29–0.30 vs 0.37–0.39 per
    ten steps), and keeps the blocks near identity.
  - Neither helps the quadratic model or training.
- **Theory draft's prediction 3** (linearization error tracks quadratic-model error better than distortion):
  weakly, in 3 of 6 units; not shown.
- **Curvature staleness (both arms):** along the step just taken, the GGN curvature ten steps earlier was only
  32–42% (steps 20–100) and 56–62% (steps 110–500) of its current value; one step later it is 93–101% (N21).
([../studies/cifar_sn_soap_20261012/RESULTS.md](../studies/cifar_sn_soap_20261012/RESULTS.md))

**CIFAR-10 scale-up (2026-10-12, [T] for top-free vs SOAP; the GGN arms stopped at a gate): top-free beats tuned
SOAP on a model 54× larger.**
- **Workload:** CIFAR-10 (45k/5k split of the training set), depth-6 ViT (width 64, MLP 256, 0.31M parameters),
  batch 2048, 500 steps, GPU float32. Every arm uses crop + flip, warm-up + WSD, weight decay 0.05 and weight EMA.
- **R2 (better):** top-free c = 64 (lr 3e-2) + EMA reached 0.5408 against tuned SOAP (lr 1e-2) + EMA at 0.5536. The
  EMA d = 0.98 validation loss at step 500 was lower in each of seeds 1–3, by 0.007, 0.019 and 0.012. The raw
  weights at step 500 were also lower in all three seeds. Two qualifications:
  - top-free's raw weights were *higher* at steps 50 and 200 in all three seeds;
  - on tuning seed 0 the two arms tied (0.5427 vs 0.5430).
- **Not answered:** whether SN changes GGN training, distance preservation or the quadratic model (Q1, Q3, Q4).
  - The exact per-tensor GGN arms (γ = 1e-2, untuned) stopped at the step-100 action audit in `ggn` seed 1
    (`head.weight` 3.6e-4 vs 1e-4).
  - Float64 recomputation showed that the solved block is accurate (7e-7). The audit's float32 reference, formed in
    original coordinates, was off (2.7e-4).
  - The owner kept the gate and stopped the GGN arms.
- **GGN seed 1, steps 0–75 (descriptive):** validation loss 1.18 at step 50 against SOAP's 1.43. SOAP needed about
  125 steps to match GGN's step 75. GGN costs 24 s per step against 0.22 s, and sees a second batch per step for its
  step search.
- **Quadratic models along the steps taken (descriptive medians):**
  - Along SOAP and top-free steps after step 100, the next-batch full-GGN and Hessian models predict the loss
    change well (ratio 0.97–1.06). The per-tensor model does not (−0.03 to 0.15): cross-tensor coupling is 15–21×.
  - Along GGN steps (seed 1, steps 10–70): full 0.75, Hessian 0.98, per-tensor 0.47; the Hessian term is 1.4× the
    GGN term.
- **Distance preservation (descriptive):** local distortion rises with depth in every arm. Under SOAP the deep
  blocks drift from identity over training (RMS ‖R_k − I‖ of blocks 3–6: 0.5–0.7 at step 500). GGN reaches 0.7–0.85 by step 50.
  The relation between distortion and quadratic-model error is weak (Spearman −0.09 pooled, 7 units).
([../studies/cifar_ggn_20261012/RESULTS.md](../studies/cifar_ggn_20261012/RESULTS.md))

**GPU port (2026-10-12, [T] replication): the top-free result holds on another machine and in float32.**
- **Replication:** on the GPU machine, top-free c = 64 + EMA is below SOAP + EMA at step 500 in each of seeds 4–6:
  - CPU float64: 0.123 vs 0.133;
  - GPU float32: 0.123 vs 0.136.
- **Port:** the device-ready copy (`studies/gpu_port_20261012/gpu_parsec`) is the original bitwise on the CPU. One
  GPU float64 step agrees with the CPU to 3.3e-10.
- **Chaos floor:** device, precision (float32 vs float64) and machine change single step-500 EMA losses by
  0.002–0.008. Float32 is adequate by the pre-declared rule.
- **Cost on a 0.2M-parameter CIFAR ViT (batch 512):**
  - top-free costs about 3× SOAP per step: non-probe steps about 2×, probe steps 0.36 s;
  - the probe path needs several times the raw n × D storage, and runs out of memory at s = 4 on 0.2M parameters
    and at s = 1 on 1.2M.
([../studies/gpu_port_20261012/RESULTS.md](../studies/gpu_port_20261012/RESULTS.md))

**Damping sweep and noise-model check (2026-10-11, [T] sweep, [L] OU check): no Kalman version; c = 64 stays.**
- **Sweep** (seeds 4–6, EMA d = 0.98, step 500): more damping than c = 64 does not help.
  - Every c vs c = 64 is mixed.
  - Seed means rise from c = 64 to freezing: 0.121, 0.123, 0.126, 0.128, 0.130.
  - Freezing the top directions (c = 10⁹) is worst by mean, and its sharpness climbs to 112.
  - The best fixed c is near 64.
- **OU check** (seed-4 checkpoints): the gradient-noise model of the deviation from the weight EMA along the exact
  top GGN directions is not adequate for any arm.
  - Measured deviations are slow (lag-10 correlation 0.81–0.93; SOAP's prediction is 0.42, or −0.62 in the
    sharpest bin).
  - Under top-free they are 16–110 times the damped-noise prediction.
  - The model is not stationary over training.
  - Candidates for the missing slow component: drift, rotating directions, or leakage of the exact directions
    out of the optimizer's held subspace.
- **Decision** by the pre-declared rule: no Kalman filter; revise the noise model first. The noise-only
  explanation of the top-free gain is not supported, though the gain itself stands.
([../studies/damping_sweep_20261011/RESULTS.md](../studies/damping_sweep_20261011/RESULTS.md))

**Top-free confirmation (2026-10-11, [T], fresh seeds 4–6): in-loop damping of the top directions beats weight
averaging.** The momentum-free top-space brake (`top_rhs='gradient'`, c = 64, lr 1e-2) + EMA reaches 0.121, against
SOAP + EMA 0.135 and SOAP + final decay + EMA 0.135. It is better in all seeds; both pre-declared readings passed.
More damping keeps helping (c = 64 > 16 > 4 > 1); the optimum is not yet found. *(Damping sweep: no gain beyond
c = 64; see above.)* Its trajectories are sharper
(27–59 vs SOAP's 10–23, limit 78). **Current best PARSEC-L: lr calibration, `top_rhs='gradient'`, c = 64, plus
weight EMA.** ([../studies/topfree_confirm_20261011/RESULTS.md](../studies/topfree_confirm_20261011/RESULTS.md))

**Noise controls (2026-10-11, [T]): the brake's gain is noise removal and does not survive standard practice.**
With a 100-step final decay (WSD) SOAP reaches 0.129, and on an EMA of its weights 0.127. Both beat the constant-lr
brake (0.149) in all seeds. With decay or averaging added, the brake's extra effect is mixed (0.130 vs 0.129,
0.123 vs 0.127). The statistics effect, edge of stability and early transient explanations are not supported.
Remaining candidate: the momentum-free top space (`top_rhs='gradient'`, c = 16) beat the brake at step 500 in all
seeds (0.140), and with EMA reached 0.119, the lowest here; not yet tested against SOAP + WSD or SOAP + EMA.
**Rule from now on: every optimizer comparison includes decay and weight averaging for all arms.**
([../studies/noise_controls_20261011/RESULTS.md](../studies/noise_controls_20261011/RESULTS.md))

**Stage D result (2026-10-11, [T], seeds 1–3, 500 steps):** with learning-rate calibration c = 4 at lr 1e-2,
`norm_blend = 0` and no search, PARSEC-L beats tuned SOAP at step 500 in all seeds (0.149 vs 0.161). It
reaches the targets in fewer steps (T50 117 vs 142) but is slower in CPU wall-clock (about 2× SOAP's cost per
step). This "brake" is the first PARSEC variant to beat tuned SOAP
([../studies/stage_d_20261009/RESULTS.md](../studies/stage_d_20261009/RESULTS.md)). Recommended spec v2,
pending the owner's decision on defaults: `calibration='lr', strength=4, norm_blend=0`. The current code
default is `norm_blend = 1`, which stage D did not test.

**Working spec v1 (implemented 2026-10-09 in [../studies/parsec_l_20261009/](../studies/parsec_l_20261009/README.md);
not yet tested in training):**

- Curvature: s = 4 model-sampled labels per example, as ordinary backward passes. Randomized SVD of
  √β F0_bᵀ U_bᵀ with 2 power steps, per tensor, re-extracted at each refresh.
- Rank r = 16 (8 is the cheaper fallback).
- **Standard since 2026-10-09 (owner decision, one-seed smoke evidence):** `norm_blend = 1` (the step keeps
  SOAP's global norm) and residual-branch scaling 1/depth (`with_branch_scaling`, exponent 1). Both have
  an off-setting (0) that stage D must keep as an ablation. Evidence against them so far:
  - norm blend 1 slightly hurt at step 100 in the smoke run (0.356 vs 0.343);
  - offline it did not raise PARSEC-L's held-out progress (2 of 6 states, as without it);
  - branch scaling hurt at depth 2 (0.384 vs 0.343).
- μ from the probe trace, tr Ĉ = ‖U‖²_F.
- One tuned learning rate, SOAP's momentum, no line search on the training batch.

The implementation (`PARSECL`, a torch optimizer on upstream SOAP) passes 11 tests. Without curvature
it is bitwise upstream SOAP. At full rank its step solves the lab's damped per-tensor system in SOAP's
metric to a 1.1e-12 residual. Truncated, it equals the exact top-r shrinkage. Tracking, ρ-triggered
per-side refresh and the stop rule are switches, off by default.

Smoke run (one seed, untuned, lr 1e-2, not an evaluation): validation loss 0.72 / 0.34 at steps 50 / 100
versus native SOAP's 0.88 / 0.44, at 0.085 versus 0.041 s per step. Ten steps after a refresh, the held
subspace's invariance residual is 0.5–0.86.

## 2. Design rules supported by evidence

1. **Correct SOAP; don't replace it.** [T, gap audit and geometry follow-up] Tuned native SOAP is the bar
   at step 500 (0.161). Tuned Adam (0.202), tuned PARSEC default (0.186) and tuned PARSEC split (0.192)
   stay behind it. The default branch feeds a transformed gradient into Adam, whose normalization partly
   undoes the transform. The tuned split branch is effectively Adam: its top-space step averages 0.67% of
   the complement step.
2. **Per-tensor scope, not global and not full.** [T, geometry follow-up] At matched damping, metric and
   search, full (cross-tensor) GGN is worse than per-tensor GGN at steps 50, 200 and 500 in all seeds
   (0.330 versus 0.182 at step 500). That holds although the cross-tensor blocks carry 54–92% of the
   curvature energy. Full OT is better than per-tensor OT only at step 50. Caveat: damping was tuned for
   the per-tensor versions.
3. **Shrink the SOAP step; never invert or add a separate Newton step.** [T for the failure modes, L for
   the remedy] The split branch's Newton step diverged at lr_top ≥ 0.1 and was negligible at stable
   sizes. Exact inversion gains early, then mostly fits each minibatch (rule 8). The shrinkage form can
   only shorten SOAP's step, and does so exactly along directions where SOAP's step is too long for the
   curvature. That is a direction-wise answer to edge-of-stability concerns, without per-layer learning
   rates from λ_max.
4. **Calibrate against damping, in SOAP-whitened coordinates.** [L, capacity screen] The same rank-16
   correction recovers 56.1% of local headroom with damping calibration and 15.4% with trace calibration
   (six saved states). The scale of the uncorrected part matters more than extra rank.
5. **Precondition SOAP's momentum; one learning rate, one momentum.** [T for momentum, L for the pairing]
   Removing SOAP's momentum ruins training (0.495 versus 0.238 for default searched SOAP at step 500, gap
   audit 2b). Exact GGN applied to the momentum is poor locally (0.032 mean decrease, against 0.030 for the
   SOAP map on the gradient). PARSEC-L applied to the momentum is as
   good as on the gradient (about 63% of exact-GGN headroom, PARSEC offline).
6. **Sampled-label probes are a sufficient curvature source; true labels are not.** [L, PARSEC offline;
   T for the negative]
   - s = 4 sampled labels keep 97.8% of exact PARSEC-L's local gain (r = 16, momentum, 72 states);
     every setting keeps 93–104%, even s = 1.
   - True-label empirical Fisher trained worse than tuned searched SOAP (0.306 versus 0.210 at step 500,
     gap audit 2a).
   - A probe row is a plain backprop gradient with a sampled label, so s backward passes give every
     tensor's probes at once (study test).
7. **Re-extract at each refresh; don't track the subspace.** [L, PARSEC offline] One power step per
   update tracks worse than a fresh probe extraction at r = 4, 8 and 16 (median sin²Θ 0.088 / 0.081 /
   0.077 versus 0.044 / 0.055 / 0.073). The whitened top subspaces rotate about 0.05–0.06 per step.
8. **No line search on the batch being fitted.** [T for the effect: post-hoc analysis of the stored
   seed 1–3 training logs, 8 October; not in a RESULTS file. H for the remedy] Share of each step's loss
   drop on its own batch that shows up on the next batch:

   | Steps | 2–50 | 51–200 | 201–499 |
   | --- | ---: | ---: | ---: |
   | Native SOAP | 76% | 24% | 3% |
   | Searched SOAP, tuned | 57% | 7% | 0.7% |
   | Per-tensor GGN (γ 0.001) | 20% | 0.7% | 0.1% |
   | Per-tensor OT (γ 0.1) | 33% | 1.5% | 0.2% |

   After step 50, curvature steps chosen by search on the same minibatch mostly fit that minibatch.
   Early, GGN's 20% is still the most real progress per step of any arm. Remedy (untested, N1): a fixed
   tuned learning rate or a step size chosen on another batch. Heavier damping also helps late: OT at
   γ = 0.1 transfers more than GGN and ended best among curvature methods (0.165).

   Remedy tested offline [L, optimism follow-up, six states], against same-batch search:
   - a step size chosen on a second 512-example batch gives a higher held-out decrease for 50 of 60
     searched candidates;
   - mean held-out decrease 0.0363 vs 0.0259, recovering 92% of the oracle's gain over same-batch
     search;
   - subtracting the estimated optimism α·tr(A⁻¹Σ̂)/B wins in 14 of 18 label-free cases;
   - PARSEC-L with same-batch search had the worst late held-out decrease (−0.035 at SOAP-50).

   Any step-length adaptation in PARSEC-L must use another batch.

   Loss-free step sizes tested [L, transfer-law study, six states]. A step α = γ̂·q/κ̂ is set from noise
   statistics and the GGN curvature along the step, with no loss evaluations:
   - overall it reaches 52% of second-batch selection, below even same-batch search (72%);
   - at steps 25 and 50 it reaches 96% and beats same-batch search in 14 of 16 points;
   - at step 10 it overshoots (8%), because the GGN quadratic underestimates the curvature along the step by up
     to 2.5×;
   - a full Newton or Wiener step without search is catastrophic (held-out loss −3.5 to −20.7).

   Keep loss evaluations on a second batch for the step length. Noise statistics can correct a searched step,
   not replace the search.
9. **Curvature pays per step early, not per second or late.** [T] Exact GGN reaches the step-50 GGN
   target in 58 steps (native SOAP 142) but needs 350 s against SOAP's 15 s. Tuned searched SOAP's loss
   is 1.8–2.1 times GGN's at step 50 and 1.09–1.20 times at step 500. PARSEC-L must
   keep its overhead within a small multiple of SOAP's 0.11 s per step.

   Mechanism [L, noise_scale, CIFAR, 36 states]:
   - The noise scale of strongly preconditioned steps grows about tenfold during training (Newton's K₁ 1,340–2,170
     → 17,300–20,400 examples).
   - So at B = 2,048 the full GGN step stops paying from step 25–50, while a fractional power (about ¾ early, ½ late)
     keeps a one-step gain over SOAP's direction (1.3–2.0× late).
   - The training consequence is untested (N29).
10. **Rank around 16, with diminishing returns.** [L] Share of exact-GGN headroom: r = 8 about 57–61%,
    r = 16 about 63%. In PARSEC default training, a larger rank and more frequent refreshes helped
    (seed-0 tuning: rank 8 0.2126, 16 0.1826, 32 0.1738).

11. **Compare by seeds, not single runs; differences below about 0.008 at step 500 are numerical chaos.** [T, GPU
    port] On the MNIST workload, identical runs on another device, in another precision or on another machine end
    0.002–0.008 apart in EMA validation loss. Trajectories agree to about 1e-8 at step 10 and separate from step 25
    on; SOAP's float32 basis refreshes differ across devices by about 1e-5. Bitwise gates are only meaningful on
    one machine and one device.
12. **In float32, compute and check SOAP-whitened quantities in whitened coordinates, or check them in float64.**
    [L, cifar_ggn D1] SOAP's denominators spread over training: `head.weight` spans 6.1 decades by step 51. A
    float32 product that goes through original coordinates, F0ᵀ·C·(F0u), then loses about four digits. The same
    block accumulated from whitened rows stays accurate to 7e-7. The original-coordinate systems (condition numbers
    1.7e7–1.8e8 at step 2) are not solvable in float32 at all.
13. **Do not develop PARSEC-L as an approximation of the exact per-tensor GGN step.** [L, ladder_mechanism] At early
    CIFAR states, 56–80% of the exact step's per-tensor advantage over top-free lies outside any rank-16 top space
    plus momentum and gradient. Coefficients, freshness and estimation together account for the rest. PARSEC-L's
    training gains (top-free, and the OT-probe variant) must be explained and improved as SOAP modifications in
    their own right.

## 3. Reference numbers

| Seeds 1–3 | Loss 50 | Loss 500 | s/step (measured, 4 workers) |
| --- | ---: | ---: | ---: |
| Native SOAP, lr 1e-2 | 0.904 | **0.161** | 0.11 |
| OT per-tensor, γ 0.1 | 0.451 | 0.165 | 6.6 |
| Exact per-tensor GGN, γ 0.001 | **0.355** | 0.182 | 6.5 |
| PARSEC default (rank 32, refresh 10, lr 3e-3) | 1.380 | 0.186 | 0.83 |
| PARSEC split (rank 16, refresh 100, lr 1e-2) | 1.077 | 0.192 | 0.16 |
| Adam, lr 1e-2 | 1.113 | 0.202 | 0.09 |
| Searched SOAP, tuned | 0.697 | 0.210 | 1.37 |
| Full OT / full GGN | 0.386 / 0.670 | 0.276 / 0.330 | 9.0 / 9.3 |
| Dense empirical Fisher | 0.705 | 0.306 | 1.86 |
| SOAP map without momentum | 1.474 | 0.495 | 1.32 |

CIFAR-10, depth-6 ViT (0.31M parameters), batch 2048, GPU float32, seeds 1–3, crop + flip, warm-up + WSD, weight
decay 0.05 ([cifar_ggn](../studies/cifar_ggn_20261012/RESULTS.md)):

| Seeds 1–3 | Raw loss 50 | Raw loss 500 | EMA 0.98 loss 500 | s/step (RTX 3070) |
| --- | ---: | ---: | ---: | ---: |
| SOAP, lr 1e-2 | 1.458 | 0.588 | 0.554 | 0.22 |
| Top-free c = 64, lr 3e-2 | 1.547 | 0.567 | **0.541** | 0.29–0.31 |
| Exact per-tensor GGN, γ 1e-2 (seed 1 only, stopped at step 100) | 1.177 | — | — | 24.2 |

Local share of exact per-tensor GGN headroom over the SOAP map (gradient RHS):

| Family | Six main ViT states (2c) | 69 states across ViT architectures (2c + partial 4b) | Source |
| --- | ---: | ---: | --- |
| PARSEC-L exact, r = 16 | 56.1% | 63.0% | capacity screen; PARSEC offline |
| PARSEC-L probes, s = 4, r = 16 | — | 64.4% | PARSEC offline |
| Full-GGN CG, 5 matvecs | 67.1% | 68–114% by architecture | gap audit 2c/4b |
| Diagonal in nearest-Kronecker eigenbasis | 46.0% | 36–75% by architecture | gap audit 2c/4b |
| Kronecker sums, rank 1–3 | 15–19% | — | gap audit 2c |
| Exact diagonal in SOAP's basis | 21.0% | 9–13% by architecture | gap audit 2c/4b |

Gap-audit 4b stopped at 21 of 22 units on its own solver-agreement assertion, so its ranges are partial.

## 4. Idea status

IDs I1–I15 and SN-1 refer to [PARSEC_ideas.md](PARSEC_ideas.md); N-IDs are added here.

| ID | Idea | Status | Evidence / next test |
| --- | --- | --- | --- |
| I1 | Per-example sampled-label probes as the curvature source | **Supported locally** (statistics) | I2's result covers the statistics. The training arm (dense MC-GGN, s ∈ {1, 4}) has not run; it is optional now that PARSEC-L itself can be tested. |
| I2 | Per-tensor extraction in SOAP-whitened coordinates | **Passed** (97.8% ≥ 70%) | PARSEC offline B1 |
| I3 | Persistent one-step tracking instead of refreshes | **Killed** | PARSEC offline B2: tracking worse than fresh extraction for every r. Revisit only with several power steps per update. |
| I4 | SOAP bulk, not Adam bulk | **Adopted** (rule 1) | Tuned original PARSEC branches stay behind native SOAP |
| I5 | One lr, one momentum; precondition the momentum | **Adopted** (rule 5) | Two-lr variants: lr_top ≥ 0.1 diverged |
| I6 | Fix split-branch leakage | **Moot** under PARSEC-L | Measured leakage: up to 36% of the complement step lies in the top space |
| I7 | Early-phase schedule (raise μ or stop refreshing) | **Open**, motivated by rules 8 and 9; implemented as `probe_until` | Smoke (1 seed): hand-over at 30 or 50 made step 100 worse (0.375 / 0.399 vs 0.343). Test in stage D: hand-over N ∈ {100, 200, ∞} |
| I8 | Per-layer μ from probe traces; evidence-based damping | **Open** | Probe trace error 2.8% at s = 4 (PARSEC offline) |
| I9 | Step-size safety (λ_max cap, spectral trust region) | **Partly built in** (shrinkage only) | Test only if stage D shows instability |
| I10 | Adaptive SOAP basis refresh | **Staleness confirmed; no trigger found** [L, basis_refresh_snr] | Refresh every step: better at step 50, worse at 500 (seed-0 tuning). CIFAR, offline: the benefit of refreshing now grows with basis age from the first step on (9 of 9 windows). The probe-spread signal never tracks it better than age (0 of 9). SOAP's single QR step leaves the basis short of its factors' eigenbasis (2–5% vs 4–10% of the GGN headroom with an exact diagonal). More refresh work may beat any trigger. |
| I11 | Gradient-noise scale and NI-SOAP noise from the same probes | **Static noise filtering refuted for top-free** [L, basis_refresh_snr] | Top-free's braked directions have higher SNR than random directions, and brake strength rises with SNR (0 of 9 for both pre-declared readings, both reversed). In weight matrices they hold 29–57% of the signal energy and 8–32% of the noise in 0.4% of the dimensions. The response is about 1/60 of the static SNR optimum. |
| I12 | Reduced cross-layer coupling | **Parked for the optimizer** (rule 2) | Keep for UQ only |
| I13 / SN-1 | SN as architecture-side conditioning | **S2 not confirmed** (partial test); **no optimizer effect at matched settings** [T] | Stream-writer Kronecker error not consistently lower (ViT 5 of 6 pairs). GGN and SOAP train the same with and without SN at c = 0.95 (see below the table). The full SN-1 test (paper recipe, c ∈ {0.5, 1, 3}) has not run. The CIFAR GGN ± SN (c = 1) arms did not run (cifar_ggn D1). **CIFAR SOAP ± SN (c = 1) [T]: SN worse in every seed (EMA 0.580 vs 0.554). It raises cross-tensor coupling (17–19 vs 13–15), does not improve the quadratic model, and slows representation drift (cifar_sn_soap).** |
| I14 | Laplace posterior as a byproduct | **Open** | — |
| I15 | Cheaper SOAP from shared, forward-computable statistics | **Open**; I15a's factor-rotation trigger not supported [L, basis_refresh_snr] | Zero-cost first check on saved states (cross-layer off-diagonal energy) not yet run. As a refresh trigger, the rotation of SOAP's own factors in the held basis tracked the refresh benefit at ≥ 0.3 in 6 of 9 windows and beat age in 5 (needed 7 and 7). It was strongest late (steps 251–281). |
| N1 | Same-batch line search makes curvature steps fit the minibatch | **Established** [T] (rule 8) | Test: GGN or PARSEC-L with the step size chosen on a different batch |
| N2 | Full GGN rank ≤ B(K−1) = 4,608 < 5,738 parameters | Fact | Bears on any full-matrix variant at batch 512 |
| N3 | Tight SN keeps residual branches alive and lowers early sharpness | **Observed** [L, exploratory] | MLP under searched SOAP, step 200: without SN every branch is dead (0% active GELU units); at c = 6 almost every branch is dead (0–41% per block); at c ≤ 2 34–62% stay active. ViT at c = 0.95: top GGN eigenvalue 74 versus 179 at step 10; diagonal and empirical-Fisher inverse errors lower in all six pairs. Untested in PARSEC training. |
| N4 | Check branch activity before reading late curvature diagnostics | Practice | Dead branches make Kronecker errors trivially 0 |
| N5 | SOAP as two half-steps: z = m̃ ⊘ √d, shrink z along w_k, unwhiten | **Same operator as section 1** | Algebraically the PARSEC-L step with λ = μ · scale. Only an implementation layout; nothing new to test. |
| N6 | Warm-started tracker step every k steps (W ← orth(Y Yᵀ W)) | **Conflicts with I3**; implemented as `refresh='track'` (with Rayleigh–Ritz) | One power step every step already tracked worse than fresh extraction (B2). Every k steps the subspace moves k times as far. Use re-extraction (randomized SVD, 2 power steps), optionally seeded with the old W (untested). The whitened coordinates also move every step as ṽ changes, not only at basis refreshes. |
| N7 | Rank-one probe columns y_j = (Q_Lᵀδ_j)(Q_Rᵀa_j)ᵀ ⊘ √d; rank-one storage of w_k | **Wrong for transformers as stated** | A per-example gradient of a token-shared weight is Σ_t δ_t a_tᵀ, rank up to T, so each probe column is dense or token-factored (the offline study used dense per-example gradients). Storing w_k as u_k v_kᵀ is an extra approximation; untested. |
| N8 | Per-side SOAP refresh when ρ = ‖offdiag(S)‖/‖S‖ of the probe partial traces has *risen* by τ | **Open** (extends I10, I15); implemented as `soap_refresh='triggered'/'both'`, `rho_threshold` | Validate offline first: does high ρ predict a better step from a refreshed basis, and does it fire more often early? Use S = Σ_j G_j G_jᵀ with dense G_j for token-shared weights. CIFAR offline [L, basis_refresh_snr]: the probe ρ rise tracked SOAP's refresh benefit at ≥ 0.3 in 5 of 9 windows and beat age in 3 (not shown). |
| N9 | Build SOAP's bases from the probe GGN (S_L, S_R) instead of GGᵀ | **Offline yes; in training no for SOAP, modest for top-free** [L, basis_refresh_snr; T, n9_training] | Offline (CIFAR SOAP states): with a diagonal in that basis, 12–18% of the exact per-tensor GGN's advantage over SOAP's map at steps 10–50, against 2–5% in SOAP's basis (9 of 9); held-out positive post hoc at steps 10–50. Training (seeds 1–3, 500 steps; rebuilt every 10 steps, momentum and v carried): SOAP + N9 is worse raw at steps 25, 50 and 200 in every seed at its selected lr 3e-2, and mixed at EMA 500 (0.563 vs 0.554). The own-factor rebuild control is the same. Top-free + N9 is better raw at steps 25 and 200 in every seed, has a lower EMA 500 in 2 of 3 (0.531 vs 0.541) and higher accuracy in all 3. Next: more seeds or a longer run for top-free + N9. |
| N10 | Tracker residual e = ‖K̂W − W diag κ‖/‖K̂W‖ as a re-extraction trigger | **Open**, cheap; implemented as `residual_threshold`, always logged | In the supplied source, `innovation_residual` is only a state field (`eigentools_excerpt.py`); `parsec_pasted.py` never sets or reads it. |
| N11 | Correction energy c = 1 − ‖z′‖²/‖z‖²: stop probing when it stays low | **Open** (automates I7); implemented as `stop_energy`, c logged every step | Log c in stage D first; decide the rule afterwards. |
| N12 | Rescale PARSEC-L's step globally to SOAP's step norm | **Open** | Lets SOAP's tuned lr carry over; no longer "only shrinks". Never per layer (allocation matters). Implemented as `norm_blend` (0 = off). Smoke: PARSEC-L's step is only about 12% shorter than SOAP's on average; a = 1 helped at step 50 (0.672 vs 0.717) and slightly hurt at 100. |
| N13 | Fixed depth branch scaling s = α/depth^p (residual h + s·branch) as architecture-side conditioning instead of SN | **Open**; implemented as `with_branch_scaling(config, exponent, alpha)`, exponent 0 = plain; block-Jacobian instrument `block_jacobian_deviation` implemented. Smoke (1 seed, PARSEC-L, 100 steps): plain ViT blocks are far from identity (first block RMS ‖R−I‖ 1.7 at init, 2.5–3.0 after training); at depth 4, 1/depth scaling keeps all blocks near 0.6 and gave the best loss (0.319 vs 0.396 plain); at depth 2 it did not help | Targets the block Jacobians R_k = I + s·DF_k that the transport argument needs. Weight SN does not bound LayerNorm or attention, and c = 0.95 per layer bounds a GELU branch only by about 1.02. |
| N14 | Depth test: depth {2, 4, 8}, ViT and MLP, width 16, plain vs branch scaling (SN as reference), with per-block ‖DF_k‖ from random JVP probes | **Proposed** | Prediction: Jacobian deviation, Kronecker error and the SOAP–GGN gap grow with depth in plain networks and branch scaling flattens it. Kill: Jacobian deviation stays small with depth even when plain. Rationale (other assistant; paper table not verified here): depth, not training length, correlates with worse distance preservation. First depth-6 data (cifar_ggn, CIFAR-10, plain ViT, descriptive): local distortion rises with depth (SOAP step 500: 1.5 after block 1, 1.7–1.8 after block 6, 3.5–4.0 at the head input). Deep-block deviation grows with training under SOAP (blocks 2–6: 0.12–0.62 at step 50, 0.5–0.87 at step 500), and exact GGN reaches 0.7–0.85 by step 50. Along SOAP's step, the representations' linearization error grows with depth: about 3% after block 1, 8–9% after block 6, 11% at the head input (cifar_sn_soap). |
| N16 | Signal/optimism theory of transfer: share = (1 − r)/(1 + r) under same-batch search, r = tr(A⁻¹Σ)/(B ḡᵀA⁻¹ḡ) (B·r = noise scale in the optimizer's metric) | **Partly supported** [L, optimism study] | Banked states, 24 points: calibration partial (Spearman 0.56, MAE 0.24); r ordered SGD < SOAP map < exact GGN in 6/6 states and rising with training (exact GGN 0.35–0.58 at step 10, 1.36–1.52 at step 50); EF-Newton breaks the predicted ordering (its A is built from the batch's own gradients); the realized line-search mechanism explains shares (Spearman 0.86); momentum shares are not predicted by the expectation. Next: noise-corrected line search or second-batch step size (N1). |
| N16 follow-up | Refined theory and PARSEC-L under it | [L, optimism follow-up] | EF ceiling holds (≤ 0.81, so batch-built preconditioners self-normalize). The mechanism holds when measured independently (Spearman 0.80). Label-free calibration (post hoc): Spearman 0.70, MAE 0.23. Momentum staleness is higher on the large-step GGN trajectory (SOAP-map and GGN metrics, not SGD's). **PARSEC-L (damping-calibrated, γ 0.01) has the largest r of all operators (0.41–3.33).** Its κ of 1,500–7,300 removes the top-16 whitened directions, which carry most per-step progress at these step sizes. At SOAP's real step it keeps 79–91% of SOAP's norm and SOAP's transfer share (5 of 6 states), but beats SOAP's held-out decrease only where SOAP's own step hurts (2 of 6). |
| N17 | Learning-rate calibration β = c·η_t instead of β = 1/(μ·scale): SOAP's step is a proximal step for ‖ζ‖²/(2η) in whitened coordinates, so adding curvature gives κ_k = c·η·λ_k, the edge-of-stability ratio; no γ | **Supported in training** [T, stage D]: c = 4 at lr 1e-2 beats tuned SOAP at step 500 in all seeds. In the seed-0 grid, c = 1 ≈ SOAP and c = 30 / damping are in between; every brake beats SOAP at lr 3e-2, but the optimum stays at lr 1e-2. Per tensor, SOAP's sharpest directions run at ηλ_max ≈ 4. The full-network value is 10–27, far below the EMA-momentum stability limit of 78, so this is not edge of stability. Mechanism (brake_mechanism study): gradient noise is concentrated in underdamped sharp directions (57–70% in 11% of dimensions), and the brake lowers the noise floor there without slowing them. | At the six states κ_max = 0.03–0.66 (no overshoot by the quadratic model). The step keeps 96–100% of SOAP's norm and its held-out decrease is within 0.0015 of SOAP's where SOAP helps. It helps less than damping calibration at GGN-10, where SOAP's harm is not curvature overshoot. Needs a stage-D comparison: calibration ∈ {damping γ 0.01, lr c ∈ {1, 4}}. |
| N15 | Spectral-norm trust region on the curvature step (I9) to get SN's early batch-fit reduction directly | **Proposed** | SN raised GGN's steps-1–50 transfer from 19% to 32% by shrinking the own-batch decrease; a step-level bound would act on the same same-batch fitting (N1) without changing the architecture. |
| N18 | Momentum-free top space (`top_rhs='gradient'`): inside the held subspace, replace SOAP's momentum by the gradient, then brake with κ = c·η·λ | **Supported in training, on MNIST and on CIFAR-10** [T, topfree_confirm; damping sweep; GPU port; cifar_ggn] | CIFAR-10 depth-6 ViT (0.31M parameters), seeds 1–3, every arm with WSD + EMA + augmentation: 0.541 vs SOAP 0.554, better in all seeds (margins 0.007–0.019; tuning seed 0 tied; raw weights higher mid-training). Fresh seeds 4–6: c = 64 + EMA 0.121 vs SOAP + EMA 0.135 and SOAP + WSD + EMA 0.135, better in all seeds. Sweep: c ∈ {256, 1024, 4096, 10⁹} all mixed vs 64, and seed means get worse with c; freezing is worst. The mechanism is open: the noise-only story failed the OU check (N19). **2,000 steps (long_run, seeds 1–5, all arms retuned to lr 9e-2): no longer better than SOAP (EMA 0.462 vs 0.457, lower in 1 of 5); the 500-step gain is a mid-run effect.** |
| N19 | Per-direction Kalman gain for the top coordinates, from an OU noise model with σ_k² and λ_k per direction | **Not built** [L, damping sweep] | Gate failed: S2 flat or undetermined, O1 not adequate (measured deviations slow, 16–110× the damped-noise prediction under top-free, non-stationary). Next: measure capture of the exact top directions by the held subspace, and distance to the valley floor g_k/λ_k, on the saved checkpoints; refit the model with leakage and drift. |
| N20 | Transfer law for batch-fitted steps with the finite-corpus correction r = (1 − f)·B_noise/B, the new-data gap r_pop = B_noise/B, and statistics-only step sizes (γ̂/λ̂, Wiener) | **Partly supported; step rule fails** [L, transfer-law study] | Corpus share: Spearman 0.96, MAE 0.18 (partial). Accurate for SGD and the SOAP map (MAE 0.07 / 0.11) and for every metric at B ≥ 1024. Overstates harm for exact GGN at small B (predicted −0.5 to −0.9, measured about 0). The noise part is exact; the miss is curvature: along a batch-fitted step the batch's own curvature is up to 4× the corpus curvature (κ ratio 0.25 at B = 64), so the batch search self-limits. The new-data gap is real (exact GGN at f = 0.82: corpus 0.92–0.99, new data 0.57–0.95). B_noise for exact GGN is 195–1,204 examples. Loss-free γ̂/λ̂ step: 52% of second-batch selection (fails), 96% late, 8% early. Wiener full step: catastrophic; gate failed, so B2 is not evaluated. The proposal's "curvature optimism" is not supported: another batch's curvature changes the share by at most 0.06. |
| N21 | Curvature along the current step is stale on SOAP's ten-step refresh scale | **Observed** [descriptive, cifar_sn_soap, both arms] | dᵀC(θ_{t−10})d / dᵀC(θ_t)d along the step just taken: 0.32–0.42 over steps 20–100, 0.56–0.62 over 110–500; one step later 0.93–1.01. Bears on SOAP's refresh interval and PARSEC-L's `probe_every` 10 (section 6, item 2). Next: a random-direction control, to separate curvature change from the step's selection of currently sharp directions, and the same ratio for PARSEC-L's held subspace. Control (ladder_mechanism, 18 points): mostly selection; in 11 of 18, random directions change at most half as much as the step. Around step 60 random directions also lose 30–60%. |
| N22 | Spatial distance preservation, temporal representation drift and parameter nonlinearity are separate quantities. Only the last bears on the GGN quadratic model (owner's theory draft, 2026-10-12, prediction 3). | **Partly supported** [descriptive, cifar_sn_soap; cifar_ggn R6] | SN changes preservation and drift clearly but nonlinearity barely, and the quadratic model not at all. Quadratic-model error correlates with logit-linearization error weakly (Spearman −0.02 to 0.39) and with head distortion not at all (−0.31 to 0.07); the pre-declared "better predictor in every unit" holds in 3 of 6. Along SOAP's own steps, network nonlinearity and the loss's beyond-second-order terms share the model error about equally. |
| N23 | PARSEC-L curvature from the OT metric (graph-Wasserstein pullback) instead of the GGN, sketched by probes: J_zᵀ H R⁻ᵀ ξ with (L + 11ᵀ/K) = RRᵀ | **Best variant so far; small metric effect at matched tuning** [T, probe_variants, ot_tuned_1000, ggn_ot_fairness] | Top-free + OT probes at c = 4096 (tuned) vs GGN at c = 64: better in every seed at raw steps 50 and 200 and at EMA 500 (0.521 vs 0.541). GGN's c is untuned on CIFAR, so the control `ggn_c_control_20261013` is pending. Exact OT probes reproduce the lab's dense Wasserstein blocks to 1e-9. The control was cancelled by the owner before running (ggn_c_control D1), so the confound stays unresolved. The variant was not run at 2,000 steps. **1,000 steps, both tuned (ot_tuned_1000): OT top-free (lr 9e-2, c 4096) beats SOAP (lr 3e-2) in every seed, EMA 0.460 vs 0.483. The gain comes with a 3× larger usable lr.** **Matched tuning (ggn_ot_fairness): GGN top-free tuned in lr and c (lr 9e-2, c 256) gets EMA 0.466; OT is better in every seed but by only 0.006 (F1 "the OT metric matters" by the declared sign reading), and tuned GGN reaches 74% of OT's gain over SOAP (F2 mixed).** |
| N24 | Probe curvature on an independent batch, not the step's batch (owner's idea, 2026-10-13) | **Not shown** [T, probe_variants] | Mixed at every point (EMA 0.537 vs 0.541). The brake's curvature scale is unchanged (η·λ_max about 9.7). |
| N25 | Power-law spectrum fitted from the top-k probes and the probe trace, used to set the bulk preconditioning without eigenvectors (owner's idea, 2026-10-14, after Meterez et al. 2026, not verified here) | **Not useful as stated** [L, bulk_krylov] | The spectrum is a broken power law early (top about k^−1, bulk about k^−2), so top-16 plus trace underestimates λ₁₇ by 1.6–2.4× (P2 3–15% of tensors). Assigned by the probe diagonal, the power-law diagonal closes a median 6% of the per-tensor GGN gap, the same as the exact (Jacobi) diagonal (K1). Without eigenvectors or curvature products, a spectrum can only set a diagonal in some fixed basis. |
| N26 | Bulk correction by a few curvature products: top-16 probe deflation plus Krylov (CG) in the deflated bulk; online, matrix-free full-GGN CG on a sub-batch | **Supported locally, untested in training** [L, bulk_krylov] | Deflated Krylov: 2/4/8 products close medians of 35/51/70% of the per-tensor gap, and the 4- and 8-product steps beat the exact step on held-out loss at median. Chebyshev with a power-law interval fails. 5-step full-GGN CG on 256 examples matches the exact step held out at median (7/9 vs top-free), at 0.07 s per product. Next: an interval from Lanczos rather than the power law, then a training test against tuned SOAP and top-free with decay and EMA. |
| N27 | Calibrate spectral quantities rarely and refresh directions often: late in training the per-tensor GGN spectrum is stationary while its eigenvectors rotate (owner's Meterez question, 2026-10-20) | **Supported offline, late training only** [L, spectral_rotation] | Late shape change is 0.03–0.08 over 10–50 steps; top-16 capture by a cached subspace is 0.84–0.88 after 10 steps and 0.70–0.81 after 50. Early (steps 10–100) the spectrum itself changes. Rotation exceeds batch noise (ratio 0.02–0.6), so pooling batches is not the lever; refresh is. The queue test of pooling sources is `queue_curvature_20261020`. |
| N28 | Curvature along the data queue (owner's queue theory, 2026-10-20): pool probe statistics over the coming or past batches; the window is set by drift vs batch noise, plus random-reshuffling corrections | **Queue corrections negligible; drift-vs-noise trade-off confirmed** [L, queue_curvature] | The visit phase shows in per-example loss (0.04–0.08 nats) but not in curvature subspaces. Next/last/random 8 batches are equivalent at current weights. Staleness costs 0.03–0.09 capture; pooling gains up to 0.07. The trailing pool beats a fresh batch only late (after about step 250). Probe-row CG doubles the one-step held-out gain over the plain step. Rademacher probes are marginally better than sampled labels. |
| N29 | Fractional curvature power on top of SOAP: steps −A^{−ν}z in whitened coordinates (A the damped per-tensor GGN), with ν set by the noise-scale law of batch_noise.md T1–T2 (owner's batch-noise discussion, 2026-10-21) | **Supported offline, untested in training** [L, noise_scale] | Best ν is interior in 35 of 36 states, about ¾ early and ½ late. ¾ beats full Newton in 35 of 36 states, and ½ beats ν = 0 in 35 of 36. The late gain is 1.3–2.0× SOAP's direction, one step ahead. Newton's noise scale grows about tenfold (K₁ 1,340–2,170 → 17,300–20,400 examples), which moves the optimum. The tuned-step law predicts the optimum about ¼ too high and overstates the gains (1.5–3.8× at ν = ½). ν = ½ is the matrix geometric mean of SOAP's metric and the damped GGN. Next: a curvature-estimation-noise term in T1 (offline), then a training test with falling ν against SOAP with WSD and EMA. Online cost needs Lanczos on probe rows (N26, N28). |

**Does SN help optimizers? Not at matched settings (first direct test).** [T, SN x GGN study]
Per-tensor GGN (γ 0.001, same-batch search) and native SOAP (lr 1e-2), each with and without SN at
c = 0.95, main ViT, seeds 1–3, 250 steps, settings tuned without SN:

- All nine readings are mixed: SN versus no SN at steps 50, 100 and 250 for each optimizer, and the
  SOAP/GGN loss ratio. Step-250 means: GGN 0.212 versus 0.213; SOAP 0.213 versus 0.221.
- GGN's advantage is unchanged: T50 in 58 steps versus SOAP's 142, with and without SN.
- SN raises GGN's next-batch transfer in steps 1–50 from 19% to 32% in every seed. It does so by
  shrinking the own-batch decrease (0.21 to 0.14 per update); real next-batch progress is unchanged
  (0.040 versus 0.044 ± 0.005). After step 50, GGN updates make no measurable next-batch progress
  with or without SN.
- In raw parameters SN halves the GGN trace and doubles the raw step norm and spectral norms.

Earlier indirect signals (SN curvature study, searched SOAP, 2 seeds, 200 steps): early sharpness
halves (top GGN eigenvalue 74 versus 179 at step 10); exact GGN's local headroom over SOAP at step 10
shrinks (0.118 versus 0.198); diagonal and empirical-Fisher estimates approximate the damped inverse
better in all six states; searched SOAP's next-batch transfer in steps 51–199 is 14% versus 8%
(post-hoc); validation loss at step 200 is about the same (0.274 versus 0.265).

**CIFAR-10, SOAP ± SN at c = 1 (cifar_sn_soap, [T]): SN hurts.** SN's lr was tuned (1e-2, equal to SOAP's), and
SN is worse in every seed and reading. It lowers shallow-layer distortion and representation drift, but leaves the
quadratic model unimproved and raises cross-tensor coupling. At c = 1 every branch weight is clipped all run; a
looser bound remains untested on CIFAR.

Still open: SN after retuning γ or lr; longer runs, where c = 2 activates; SN with PARSEC-L; SN's
effect on Laplace or OOD quality, which these in-distribution runs do not measure.

## 5. Parked or rejected

- **Global Lanczos ranks** (original PARSEC): only match per-tensor GGN at step 500 and are slowest
  early; a global basis spends its rank on a few sharp tensors.
- **Split Newton top step with its own learning rate:** unstable at useful sizes, negligible at stable
  ones.
- **Cross-tensor coupling in the optimizer** (rule 2); **true-label empirical Fisher** (rule 6).
- **Kronecker sums fitted in Frobenius norm** (15–19%); **allocation-only variants** (local gains did not
  survive training).
- **Learned GNN predictor:** its gate failed; revisit only if a cheap family needs predicting. PARSEC-L
  probes compute the operator directly.
- **One-step subspace tracking** (I3, killed).

## 6. Open questions and next tests (in order)

**Current state (2026-10-12):** the CIFAR-10 scale-up (`studies/cifar_ggn_20261012`) closed early, and the SN
question was then run for SOAP (`studies/cifar_sn_soap_20261012`).
- **Done:**
  - Top-free beats tuned SOAP on CIFAR (N18).
  - SN (c = 1) makes SOAP worse there and does not improve its quadratic model (I13, N22).
  - The GGN ± SN arms stopped at the float32 action audit (D1, owner decision).
- **Done (2026-10-13), offline at CIFAR states (`basis_refresh_snr_20261013`):**
  - N9 supported. Its next step is a training test.
  - Refresh staleness confirmed, but no trigger beats the clock.
  - Top-free brakes its highest-SNR directions, so it is not a static noise filter.
- **Done (2026-10-13), later the same day:**
  - N9 in training: no gain for SOAP, a modest gain under top-free.
  - The ladder: the goal "close the GGN gap" is formally dropped (rule 13).
  - OT probes beat GGN probes, but c was tuned for OT only.
  - Probes from another batch: no effect.
- **Concluded (13 October, evening):**
  - `long_run_20261013`: top-free's advantage does not survive 2,000 steps (R1 mixed, 1 of 5); N9 is mixed.
  - `ggn_c_control_20261013`: cancelled before running.
  - **The owner stops building on this PARSEC idea.** The open items below are kept for the record; none is
    scheduled.
- **Done (2026-10-14):**
  - `ot_tuned_1000`: tuned OT top-free beats tuned SOAP at 1,000 steps in every seed.
  - `ggn_ot_fairness`: at matched tuning, OT adds only 0.006 over GGN probes (N23).
  - `bulk_krylov`: a spectrum alone cannot precondition the bulk (N25), but a few deflated Krylov products can, locally
    (N26).
- **Done (2026-10-20 and 21), offline analysis studies after the stop:**
  - `spectral_rotation`: late in training the spectrum is stable while its eigenvectors rotate (N27).
  - `queue_curvature`: queue position does not matter for curvature; staleness and pooling do (N28).
  - `noise_scale`: noise scales grow about tenfold during training, and full Newton stops paying at B = 2,048 from
    step 25–50. The best power is interior and falls from ¾ to ½ (N29).
- **Still open:**
  - a **training test of N9**: SOAP, and top-free PARSEC-L, with probe-factor bases (done: n9_training);
  - cheaper or more refresh work for SOAP's basis (more power steps), since staleness starts at once;
  - a dynamic explanation of top-free's gain (tracking, N21 curvature drift, sharpness-dependent annealing);
  - the GGN ± SN questions on CIFAR. A float64 recheck of failing audit tensors would make the GGN arms runnable at
    a cost of seconds per audit step (rule 12);
  - N21's random-direction control, and the curvature staleness of PARSEC-L's held subspace;
  - CIFAR's run-to-run floor;
  - longer runs;
  - the offline noise-model revision below.

**Earlier next test (2026-10-12, now run):** CIFAR-10 scale-up on the GPU (owner decisions of 12 October):
- the depth-6 M1 ViT (width 64, MLP 256, about 0.31M parameters), batch 2048, float32;
- exact per-tensor GGN with and without SN (c = 1), tuned SOAP and top-free PARSEC-L;
- distance preservation by depth, and quadratic-model fidelity (full GGN, per-tensor GGN, Hessian) along the steps
  taken.

**Earlier next test (2026-10-11):** revise the noise model offline on the `topfree_confirm` seed-4 checkpoints.
Measure how much of each exact top direction the held subspace captures (leakage) and the distance to the valley
floor g_k/λ_k (drift), before any filter (N19). Then run c = 64 top-free on longer horizons and other models. The
list below predates stage D and is kept for its remaining items.

1. **Stage D: PARSEC-L training** (spec v1) against tuned native SOAP and per-tensor GGN, seeds 1–3, 500
   steps, steps and seconds to target. Tune lr and γ on seed 0. Ablations: r ∈ {8, 16}, refresh interval,
   I7 hand-over. This is decisive, because local gains have repeatedly failed to survive training
   (allocation, empirical Fisher, full GGN).
2. **Refresh interval:** how fast does a k-step-old extraction lose its local value? This can be measured
   offline along a saved every-step trajectory. The smoke run's held-subspace residual (0.5–0.86 after ten
   steps) suggests probe_every 10 may be too long.
3. **Cost:** time s = 4 backward passes plus the per-tensor SVDs against SOAP's step. The parallel
   speedup across tensors needs GPU batching.
4. **N1 remedy:** step size from a held-out batch for curvature methods.
5. **SN, remaining:** retuned or longer runs and SN with PARSEC-L (matched settings showed no effect;
   section 4); **I15 zero-cost check;** the full **SN-1** test with the paper's recipe; **longer
   horizons** (the evidence here ends at 500 steps).

## 7. Sources

- Gap audit (checks 1–4): [../studies/gap_audit_20261007/RESULTS.md](../studies/gap_audit_20261007/RESULTS.md)
- Geometry follow-up (tuned PARSEC, OT, full versus per-tensor): [../studies/geometry_followup_20261008/RESULTS.md](../studies/geometry_followup_20261008/RESULTS.md)
- SN curvature (A–C): [../studies/sn_curvature_20261008/RESULTS.md](../studies/sn_curvature_20261008/RESULTS.md)
- PARSEC offline (I2, I3, SN-1 partial): [../studies/parsec_offline_20261008/RESULTS.md](../studies/parsec_offline_20261008/RESULTS.md)
- SN x GGN training (SN with GGN and SOAP, 250 steps): [../studies/sn_ggn_20261008/RESULTS.md](../studies/sn_ggn_20261008/RESULTS.md)
- Capacity screen: [../docs/learnable_capacity_results_20261007.md](../docs/learnable_capacity_results_20261007.md)
- Signal/optimism test (N16): [../studies/optimism_20261009/RESULTS.md](../studies/optimism_20261009/RESULTS.md)
- PARSEC-L implementation and smoke run: [../studies/parsec_l_20261009/README.md](../studies/parsec_l_20261009/README.md)
- GPU port (bitwise port, float32, chaos floor): [../studies/gpu_port_20261012/RESULTS.md](../studies/gpu_port_20261012/RESULTS.md)
- CIFAR-10 scale-up (top-free vs SOAP; GGN stopped at D1): [../studies/cifar_ggn_20261012/RESULTS.md](../studies/cifar_ggn_20261012/RESULTS.md)
- SOAP ± SN on CIFAR-10 (SN, nonlinearity, drift, prediction 3): [../studies/cifar_sn_soap_20261012/RESULTS.md](../studies/cifar_sn_soap_20261012/RESULTS.md)
- Offline N9, triggered refresh and I11 on CIFAR states: [../studies/basis_refresh_snr_20261013/RESULTS.md](../studies/basis_refresh_snr_20261013/RESULTS.md)
- N9 in training (SOAP, top-free): [../studies/n9_training_20261013/RESULTS.md](../studies/n9_training_20261013/RESULTS.md)
- Probe variants (OT probes, other-batch probes): [../studies/probe_variants_20261013/RESULTS.md](../studies/probe_variants_20261013/RESULTS.md)
- Reduced-subspace ladder and N21 control: [../studies/ladder_mechanism_20261013/RESULTS.md](../studies/ladder_mechanism_20261013/RESULTS.md)
- Long run, SOAP vs top-free (and + N9) over 2,000 steps: [../studies/long_run_20261013/RESULTS.md](../studies/long_run_20261013/RESULTS.md)
- Tuned OT-probe top-free vs tuned SOAP, 1,000 steps: [../studies/ot_tuned_1000_20261014/RESULTS.md](../studies/ot_tuned_1000_20261014/RESULTS.md)
- Original source: `references/user_optimizers/parsec_pasted.py` (`pns_eigenadam`)

## 8. Changelog

- **2026-10-08** Created. Banked rules 1–10, the reference tables and the idea status after the gap
  audit, geometry follow-up, SN curvature and PARSEC offline studies. I2 passed, I3 killed, S2 not
  confirmed, N1–N4 added. Spec v1 proposed for stage D.
- **2026-10-08** Added N5–N12 from the written-out "SOAP half-step + shrinkage" algorithm (refresh
  signals ρ, e, c). Started `studies/sn_ggn_20261008` (per-tensor GGN and native SOAP with and without SN,
  250 steps).
- **2026-10-08** SN x GGN study complete: all readings mixed; SN raises GGN's early next-batch transfer
  (19% to 32%) without changing real progress. Section 4 SN answer and I13 updated.
- **2026-10-09** Implemented PARSEC-L as a separate PyTorch optimizer (`studies/parsec_l_20261009`), with
  extract/track/exact refresh, ρ-triggered per-side SOAP refresh, residual trigger and stop rule as
  switches, plus depth branch scaling (N13). 11 tests pass; one-seed smoke run added. N13–N15 added.
- **2026-10-09** PARSEC-L gained `probe_until` (I7), `norm_blend` (N12), batched extraction ('auto':
  CPU per tensor) and an optional ρ signal; `block_jacobian_deviation` added for N13/N14. 15 tests pass;
  one-seed smoke ablations added. N16 (signal/optimism theory) tested on banked states: partly supported.
- **2026-10-09** Optimism follow-up: second-batch step size validated offline (50 of 60, 92% of oracle); PARSEC-L's damping calibration found to remove its most productive directions at trajectory step sizes (N16 follow-up); learning-rate calibration proposed (N17). Norm blend 1 and 1/depth branch scaling made standard settings by owner decision, with their off-settings kept as required ablations.
- **2026-10-09** PARSEC-L gained `calibration='lr'` (N17), `rhs='gradient'`, step length chosen on a selection
  batch with a model-free hand-over to SOAP, and the true-label optimism diagnostic; 21 tests pass. Stage-D
  protocol drafted (`studies/stage_d_20261009/PROTOCOL.md`), awaiting approval; runs deferred by the owner.
- **2026-10-11** Stage D complete. The brake (lr calibration, c = 4) beats tuned SOAP at step 500 in all seeds
  (N17 supported). Early-Newton (selection-batch step) was mixed and unstable; its model-free hand-over never
  triggered. GGN with a selected step was better at step 200 and mixed at 500 (N1 remedy, partial). Spec v2
  proposed.
- **2026-10-11** Brake mechanism checked offline: not edge of stability (full preconditioned sharpness 10–27
  vs limit 78; an earlier edge-of-stability claim was an error and is corrected). Noise is concentrated in the
  braked directions (57–70%), and the brake lowers training loss on identical batches in all seeds. Mechanism:
  selective annealing of underdamped sharp directions.
- **2026-10-11** Noise controls: SOAP + WSD decay or + weight EMA beats the constant brake in all seeds. The
  brake adds nothing consistent on top (mixed). Mechanism N (noise) supported; S, E and T not. Stage-D claim
  qualified. Momentum-free top space (c = 16) is the remaining PARSEC-L candidate.
- **2026-10-11** Top-free confirmation on fresh seeds 4–6: momentum-free top space with c = 64 + EMA beats SOAP +
  EMA and SOAP + WSD + EMA in all seeds (0.121 vs 0.135). In-loop damping adds beyond averaging. Next: larger c,
  per-direction Kalman gain, OU check on the saved checkpoints, longer runs and other models.
- **2026-10-11** Damping sweep and OU check (`studies/damping_sweep_20261011`). c ∈ {256, 1024, 4096, 10⁹} are all
  mixed against c = 64, with seed means worsening towards freezing; c = 64 stays. The OU noise model is not
  adequate for any arm (slow deviations; 16–110× the prediction under top-free; non-stationary). By the
  pre-declared rule, no Kalman version is built. N18 and N19 added.
- **2026-10-11** Transfer-law test (`studies/transfer_law_20261011`, offline, six states).
  - A1–A3 partial: the law ranks transfer (Spearman 0.96) but overstates small-batch harm for curvature steps,
    because the batch curvature along its own step is inflated.
  - B1 fails: the loss-free γ̂/λ̂ step reaches 52% of second-batch selection (96% late, 8% early).
  - B2 not evaluated: the Wiener gate failed at a 2e-8 forward difference on a block with condition number 1e10;
    the full Wiener step was catastrophic as a diagnostic.
  - A scale bug for the SOAP map was found, fixed and rerun.
  - N20 added; rule 8 extended. Repository `.gitignore` extended for the GPU move (separate from the study).
- **2026-10-12** GPU port (`studies/gpu_port_20261012`).
  - The device-ready PARSEC-L copy equals the original bitwise on the CPU; GPU runs repeat bitwise; one GPU step
    agrees with the CPU to 3.3e-10.
  - The top-free result replicates on the GPU machine in float64 and in float32 (seeds 4–6, all better).
  - Single runs differ by 0.002–0.008 across device, precision and machine (rule 11).
  - The probe path's memory limits PARSEC-L to about 2,000 probe rows × 0.2M parameters on 8 GB.
  - The attention key bias has a zero gradient (its probes are rounding noise).
- **2026-10-12** CIFAR-10 scale-up (`studies/cifar_ggn_20261012`), closed early.
  - Top-free c = 64 + EMA beats tuned SOAP + EMA on CIFAR-10 with a depth-6 ViT (0.31M parameters, batch 2048), in
    all of seeds 1–3: 0.541 vs 0.554 (N18).
  - The exact per-tensor GGN ± SN arms stopped at the step-100 float32 action audit (D1). The solved block was
    accurate; the original-coordinate reference was not. The owner kept the gate.
  - Rule 12 added. Reference numbers for CIFAR added. N14 gets its first depth-6 descriptive data.
  - The SN questions remain open.
- **2026-10-12** SOAP ± SN on CIFAR-10 (`studies/cifar_sn_soap_20261012`).
  - SN (c = 1, tuned lr 1e-2) makes SOAP worse in every seed (EMA 0.580 vs 0.554).
  - None of the five SN hypotheses passes. Coupling rises with SN; representation drift falls.
  - Added measures from the owner's theory draft: error split, matched logit-step nonlinearity, ten-step curvature
    persistence, and per-depth linearization and drift.
  - Prediction 3 is not shown (3 of 6 units). N21 (ten-step curvature staleness) and N22 (preservation, drift and
    nonlinearity are distinct) added; I13 and N14 updated.
- **2026-10-13** Offline N9, refresh and I11 tests (`studies/basis_refresh_snr_20261013`), at CIFAR SOAP and top-free
  states.
  - N9 is supported on the per-tensor quadratic (9 of 9), and post hoc on held-out loss at steps 10–50.
  - Refresh staleness grows from the first step on, but no trigger signal beats age (I10, I15a, N8).
  - Top-free brakes its highest-SNR directions (I11 reversed; static noise filtering refuted).
  - Deviations: D1 (loader bug, Part C rerun) and D2 (the transfer grid never bracketed; post-hoc check).
  - I10, I11, I15, N8 and N9 updated.
- **2026-10-13** N9 in training (`studies/n9_training_20261013`, CIFAR, seeds 1–3).
  - Probe-factor bases do not help SOAP: it is worse early at its selected lr 3e-2 and mixed at the end, and the
    own-factor exact-rebuild control is the same.
  - Under top-free PARSEC-L they help modestly: raw better at steps 25 and 200 in every seed, EMA 500 lower in 2 of
    3 (0.531 vs 0.541), accuracy higher in all 3.
  - G1 bitwise; G2 at all 450 rebuilds.
- **2026-10-13** Probe variants (`probe_variants_20261013`).
  - OT-metric probes beat GGN probes in top-free in every seed (EMA 0.521 vs 0.541), with c tuned for OT only.
    Control queued (N23).
  - Probes from an independent batch: mixed (N24).
- **2026-10-13** Ladder and N21 control (`ladder_mechanism_20261013`).
  - 56–80% of the per-tensor GGN gap lies outside a rank-16 top space, so "close the GGN gap" is formally dropped
    (rule 13).
  - N21 is mostly selection, with global sharpening around step 60.
- **2026-10-13** Long run (`long_run_20261013`, CIFAR, 2,000 steps, seeds 1–5, all arms at lr 9e-2 after retuning).
  - Top-free + EMA 0.462 vs SOAP + EMA 0.457, lower in only 1 of 5 seeds (R1 mixed); raw losses tied.
  - Top-free + N9: mixed against both.
  - The run was interrupted by a session end (resumed bitwise-equivalently) and stopped by the owner after
    completion.
  - The OT fairness control was cancelled.
  - The owner stops building on this PARSEC idea. N18 and N23 updated.
- **2026-10-14** Tuned OT-probe top-free vs tuned SOAP, 1,000 steps (`ot_tuned_1000_20261014`).
  - Better in every seed: EMA 0.460 vs 0.483, raw 0.484 vs 0.608.
  - The selected lrs differ (OT 9e-2, SOAP 3e-2). At matched lr on seed 0, SOAP wins at 3e-2 and OT at 9e-2.
  - N23 updated. The superseded `ot_1000_20261013` was closed without results.
- **2026-10-14** GGN vs OT probes at matched tuning (`ggn_ot_fairness_20261014`, 1,000 steps, seeds 1–3).
  - Tuned GGN top-free selected lr 9e-2, c = 256 (interior). OT top-free beats it in every seed by 0.006 in EMA loss
    (F1, *the OT metric matters* by the declared reading; weak evidence). Tuned GGN vs SOAP is mixed (F2).
  - Tuning the brake and lr gives 74% of OT top-free's gain over SOAP. G1 bitwise.
  - Cost corrected: both top-free arms take about 1.65× SOAP's wall time per step with the probe steps
    (`ot_tuned_1000`'s 1.3× was the median step).
  - N23 updated.
- **2026-10-14** Power-law spectra and bulk preconditioning, offline (`bulk_krylov_20261014`, 12 ladder states).
  - The spectrum is a broken power law early. The probes measure its top and trace accurately but cannot predict the
    bulk (P1, P2 not supported).
  - The power-law diagonal reaches only the exact (Jacobi) diagonal's level, a median 6% (K1).
  - Deflated Krylov with 8 products closes 67–76% of the per-tensor gap (K2 m = 8) and matches the exact step held
    out. Chebyshev with the probe interval fails (K3); sub-batch full-GGN CG wins held out in 7 of 9 states (K4).
  - D1: a fit-helper crash before any output; fixed and rerun. N25 and N26 added.
- **2026-10-20** Spectral stability vs rotation along training (`spectral_rotation_20261020`, 6 replayed runs).
  - Late in training the spectrum is stable while its eigenvectors rotate (both optimizers, M1 and M2). Early, both
    change. The power law straightens to α ≈ 1.
  - Rotation exceeds batch noise for the top-16 subspace at every state. G1 bitwise on 24 of 24 checkpoints. N27
    added.
- **2026-10-20** Curvature along the data queue (`queue_curvature_20261020`, top-free replays, 12 cases).
  - The visit phase is visible in losses only (V1 and V2 fail by the declared reading).
  - Queue position does not matter for curvature (S1 negligible; S2, S5 fail; S6 fails as predicted). Staleness
    (S3) and pooling (S4) do.
  - F1 not supported: the trailing pool overtakes a fresh batch earlier than predicted.
  - Probe-row CG doubles the one-step held-out gain. D1: an out-of-memory crash before output; fixed and rerun.
    N28 added.
- **2026-10-21** Created the companion file [batch_noise.md](batch_noise.md) for batch-noise theory and results
  (N1, N16, N20, N24, I11, N27, N28 consolidated; no new study).
- **2026-10-21** Noise scales, break time and fractional Newton (`noise_scale_20261021`, 6 replayed runs, 36
  states).
  - Noise scales grow with training (N2), and the full Newton power stops paying at B = 2,048 from step 25–50 (N3).
  - N1 not supported: K½ < K₀ in two step-10 states.
  - The best power is interior and falls from ¾ to ½ (F1, F2). The tuned-step law is within one grid step (P1), but
    about ¼ too high and too optimistic about the gains.
  - G1 bitwise 18 of 18, G2 worst 1.2e-4, D1 before launch.
  - N29 added, rule 9 extended with the mechanism ([L]), batch_noise.md R8 added.

# Deferred large-model design and dry-run storage estimate

This document is a design, not an implemented framework adapter or readiness claim.
The current mathematical interfaces require pure model evaluation, a stable complete
parameter registry, JVP, VJP, true loss gradient, output-metric application, a frozen
SPD map/inverse, and clone/propose/commit optimizer state semantics. A JAX adapter would
provide PyTree flattening plus jax.jvp/vjp with matching layout and dtype audits;
PyTorch-specific autograd and optimizer classes must not leak into those contracts.

For D parameters and k shared factors, factor storage is D*k*element_bytes; refreshing
requires k VJPs, plus metric/probe work. At D=162,000,000, k=8 and float32 this is
5,184,000,000 bytes (5.184 GB or 4.828 GiB), before model, gradients, SOAP moments,
matrix bases, activations or solver work. Full-factor distributed all-reduction can
communicate that volume per refresh unless partitioning changes the algorithm.
The factors must be restricted to a declared parameter scope, streamed/recomputed,
or represented by another separately validated approximation. They are not free
because backward transport already exists.

A real adapter must declare framework, tensor scopes, refresh cadence and damping
trace source; measure basis/state drift; test microbatch accumulation and global
valid-token normalization; audit distributed reductions, masking, precision and
checkpoint replay; and include an independently reproducible disabled comparison arm.
Rough storage scaling cannot substitute for measured ordinary-attention JVP/VJP cost.
Validate a tiny untied decoder with causal attention, RMSNorm, gated MLP, no dropout
and no KV cache on a network-free token task first. Only then consider fused kernels,
compilation or a documented small licensed text corpus.

No large-model launcher, dataset download or scheduler submission is supplied.

class PnsEigenAdamState(NamedTuple):
    opt_state: Optional[optax.OptState] = None
    adam_state: Optional[optax.OptState] = None
    step: Optional[Array] = None
    eigenvalues: Optional[Array] = None
    eigenvectors: Optional[Array] = None
    rng_key: Optional[Array] = None
    rotation_diff: Optional[Array] = None
    m_top: Optional[Array] = None
    v_top: Optional[Array] = None
    m_perp: Optional[Array] = None
    v_perp: Optional[Array] = None
    lr_perp_eff: Optional[Array] = None
    last_refresh_step: Optional[Array] = None
    active_k: Optional[Array] = None
    innovation_residual: Optional[Array] = None

def apply_eigen_preconditioner(
    grad_flat: Array,
    eigenvalues: Array,
    eigenvectors: Array,
    damping: float = 1e-4,
    saddle_free_neg: bool = False,
) -> Array:
    """Apply a partial Newton-like preconditioner in a global eigenbasis."""
    if eigenvalues.size == 0:
        return grad_flat

    V = eigenvectors
    lambdas = eigenvalues

    proj = V @ grad_flat
    proj_vec = V.T @ proj

    if saddle_free_neg:
        lam_eff = jnp.abs(lambdas)
        scale = 1.0 / (lam_eff + damping)
    else:
        scale = 1.0 / (lambdas + damping)

    # Keep EigenAdam-style sqrt scaling used by the PN-S implementations.
    scale = jnp.sqrt(scale)
    new_subspace = V.T @ (proj * scale)
    g_perp = grad_flat - proj_vec
    return new_subspace + g_perp

def lanczos(
    matvec: Callable[[Array], Array],
    dim: int,
    num_iter: int,
    key: Array,
    eps: float = 1e-6,
    sort_by_abs: bool = False,
    init_v: Optional[Array] = None,
    use_light_ortho: bool = False,
    light_ortho_every: int = 4,
    return_residuals: bool = False,
    max_return_vectors: Optional[int] = None,
    return_tridiag: bool = False,
) -> tuple[Array, ...]:
    v0_rand = jax.random.normal(key, (dim,))
    v0_rand = v0_rand / (jnp.linalg.norm(v0_rand) + eps)

    if init_v is None:
        v0 = v0_rand
    else:
        init_norm = jnp.linalg.norm(init_v)
        init_is_valid = jnp.logical_and(jnp.isfinite(init_norm), init_norm > eps)
        init_dir = init_v / (init_norm + eps)
        v0 = jnp.where(init_is_valid, init_dir, v0_rand)

    def body_fun(carry, i):
        v_basis, alphas, betas = carry
        # Read as fp32 for numerically stable orthogonalization; v_basis stored as bf16.
        v = v_basis[i].astype(jnp.float32)

        w = matvec(v)
        alpha = jnp.vdot(v, w)
        w = w - alpha * v

        def ortho_against_prev(current_w, basis_idx):
            prev_v = v_basis[basis_idx].astype(jnp.float32)
            proj = jnp.vdot(prev_v, current_w)
            return current_w - proj * prev_v

        full_reorth = lambda ww: jax.lax.fori_loop(
            0,
            i,
            lambda basis_idx, current_w: ortho_against_prev(current_w, basis_idx),
            ww,
        )

        def prev_only_reorth(ww):
            return jax.lax.cond(
                i > 0,
                lambda x: ortho_against_prev(x, i - 1),
                lambda x: x,
                ww,
            )

        if use_light_ortho:
            do_full = (i % light_ortho_every) == 0
            w = jax.lax.cond(do_full, full_reorth, prev_only_reorth, w)
        else:
            w = full_reorth(w)

        beta = jnp.linalg.norm(w)
        beta = jnp.where(beta < eps, 0.0, beta)
        next_v = jnp.where(beta > 0, w / (beta + eps), jnp.zeros_like(w))

        # Store back as bf16 to halve the 8 GiB Krylov basis memory.
        v_basis = v_basis.at[i + 1].set(next_v.astype(jnp.bfloat16))
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i].set(beta)
        return (v_basis, alphas, betas), None

    # bf16 storage: (num_iter+1, dim) fp32 would be ~8 GiB for 162M-param models.
    v_basis = jnp.zeros((num_iter + 1, dim), dtype=jnp.bfloat16)
    v_basis = v_basis.at[0].set(v0.astype(jnp.bfloat16))
    alphas = jnp.zeros((num_iter,))
    betas = jnp.zeros((num_iter,))

    (v_basis, alphas, betas), _ = jax.lax.scan(
        body_fun,
        (v_basis, alphas, betas),
        jnp.arange(num_iter),
    )

    tridiag = jnp.diag(alphas)
    if num_iter > 1:
        tridiag = tridiag.at[jnp.arange(num_iter - 1), jnp.arange(1, num_iter)].set(
            betas[: num_iter - 1]
        )
        tridiag = tridiag.at[jnp.arange(1, num_iter), jnp.arange(num_iter - 1)].set(
            betas[: num_iter - 1]
        )

    evals, evecs_t = jnp.linalg.eigh(tridiag)
    if sort_by_abs:
        idx = jnp.argsort(jnp.abs(evals))[::-1]
    else:
        idx = jnp.argsort(evals)[::-1]

    evals = evals[idx]
    evecs_t = evecs_t[:, idx]
    residuals = jnp.abs(betas[num_iter - 1] * evecs_t[-1, :])

    if max_return_vectors is not None:
        keep = min(int(max_return_vectors), num_iter)
        evecs_t = evecs_t[:, :keep]

    v_k = v_basis[:-1]
    # _expand_from_basis promotes bf16 basis × fp32 coeffs → fp32; cast back to bf16.
    eigenvectors = _expand_from_basis(evecs_t.T, v_k).reshape(evecs_t.shape[1], dim).astype(jnp.bfloat16)
    if return_residuals and return_tridiag:
        return evals, eigenvectors, residuals, alphas, betas
    if return_residuals:
        return evals, eigenvectors, residuals
    if return_tridiag:
        return evals, eigenvectors, alphas, betas
    return evals, eigenvectors

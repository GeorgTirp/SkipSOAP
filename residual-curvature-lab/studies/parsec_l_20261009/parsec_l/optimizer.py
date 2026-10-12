"""PARSEC-L: SOAP whose step is shrunk along the top whitened GGN directions of each tensor.

Per tensor, SOAP keeps bases Q (one per preconditioned mode), the rotated momentum m~ and second
moment v~, with d = sqrt(v~) + eps. Its update direction is Q(m~ / d) = F0 F0^T u, where
F0 = Q diag(d^-1/2) and u is the momentum in physical coordinates. PARSEC-L writes the step as two
half-steps around the whitened momentum z = m~ / sqrt(d) = F0^T u and shrinks z:

    z' = z - V (I + V^T V)^-1 V^T z,        step = -lr_t Q (z' / sqrt(d)).

V V^T is a rank-r approximation of the whitened GGN K_C = beta F0^T C F0, with the lab's damping
calibration beta = 1 / (mu scale), mu = max(gamma tr(C) / D, floor) and scale = D / sum_b tr(P_b^-1)
(P = F0 F0^T, D = number of parameters). For V = W diag(sqrt(kappa)) with orthonormal W the
shrinkage is z - W diag(kappa / (1 + kappa)) W^T z. With r >= rank(K_C) the step equals
-lr_t mu scale (C + mu scale P^-1)^-1 u, the damped GGN step in SOAP's metric; r = 0 is SOAP.

Curvature comes from probe rows u_j with sum_j E[u_j u_j^T] = C (probes.py). At a probe step the
top-r directions are taken from the whitened probes Z = sqrt(beta) F0^T U^T:
  'extract'  randomized SVD (oversampling, power steps), as in the PARSEC offline study;
  'exact'    full SVD (tests, small tensors);
  'track'    one warm-started power step from the held subspace plus Rayleigh-Ritz; extraction
             when nothing is held or the invariance residual exceeds `residual_threshold`.
Tensors of equal shape can be extracted together with batched linear algebra (`batched`; 'auto' batches
off the CPU only, since on one CPU thread the per-tensor loop measured 8-14% faster).
Between probe steps the low-rank curvature is held in physical coordinates and re-whitened with the
current F0 and beta; a SOAP basis change transports it exactly.

Standard setting: norm_blend = 1 (the step keeps SOAP's global norm; 0 gives the pure shrinkage step).
Switches (off by default unless stated; signals are logged in `last_info`):
  probe_until       hand-over: from step probe_until + 1 on, no probes and no correction (plain SOAP);
  norm_blend        a in [0, 1], default 1: rescale the whole step (globally, not per tensor) by
                    (|SOAP step| / |PARSEC-L step|)^a; a = 1 restores SOAP's step norm, a = 0 turns it off;
  rho_threshold     per-side SOAP basis refresh when the off-diagonal energy of the probe partial
                    trace in SOAP's basis has risen by this much since that side's last refresh
                    (needs soap_refresh='triggered' or 'both');
  residual_threshold  re-extraction in 'track' mode;
  stop_energy       stop probing and continue as plain SOAP once the mean correction energy
                    c = 1 - |z'|^2 / |z|^2 between probe steps stays below it for `stop_patience`
                    consecutive probe steps;
  signals           compute the rho signal at probe steps (forced on by rho triggers).
  calibration       'damping' (beta = 1/(mu scale), default) or 'lr' (beta = strength * lr * sqrt(1 - beta2^t):
                    SOAP's step is then the proximal step of its metric and kappa_k = strength * lr * lambda_k,
                    the edge-of-stability ratio; gamma then only sets mu for logging);
  rhs               'momentum' (default: shrink SOAP's own update) or 'gradient' (shrink the SOAP map of the
                    current gradient instead; SOAP's statistics still advance as usual);
  top_rhs           'momentum' (default) or 'gradient': inside the held top subspace only, replace SOAP's
                    momentum by the current gradient (bias-corrected to the momentum's scale), then brake.
                    Under EMA momentum the sharp directions are underdamped (they ring at a rate set by beta1);
                    without momentum and with the brake they are overdamped and stable (lr' lambda <= 1/c);
  selection         step(selection=f) chooses the step length on a selection batch (f returns its loss at
                    the current parameters): alpha in {0, 2^k for k in select_grid} plus golden refinement;
                    select_compare_soap also evaluates SOAP's own step and takes the better one, and
                    select_handover = k switches to plain SOAP after k consecutive SOAP wins;
  true_gradients    step(true_gradients=...) logs the optimism ratio r for SOAP's metric and PARSEC-L's.
"""
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
from third_party.soap.soap import SOAP

REFRESH = ('extract', 'exact', 'track')
SOAP_REFRESH = ('periodic', 'triggered', 'both')
NEVER = 2**62


@contextmanager
def default_dtype(dtype):
    """Upstream SOAP allocates its factor matrices in the default dtype (as the lab's soap_propose)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def rotate(x, bases, transpose=True):
    """Apply Q^T (transpose=True) or Q along every preconditioned mode of x with shape (n, *shape)."""
    for axis, q in enumerate(bases):
        if len(q):
            x = torch.tensordot(x, q, dims=[[axis+1], [0 if transpose else 1]]).movedim(-1, axis+1)
    return x


def rotate_batched(u, bases_list):
    """Q_i^T along every preconditioned mode of u[i] for u with shape (b, n, *shape), equal patterns."""
    x = u
    for axis in range(u.ndim-2):
        if len(bases_list[0][axis]) == 0:
            continue
        q = torch.stack([bases[axis] for bases in bases_list])
        x = x.movedim(axis+2, -1)
        x = (x@q.reshape(q.shape[0], *([1]*(x.ndim-3)), *q.shape[1:])).movedim(-1, axis+2)
    return x


def same_basis(a, b):
    if a is b:
        return True
    if a is None or b is None or len(a) != len(b):
        return False
    return all((len(x) == 0 and len(y) == 0) or (len(x) > 0 and len(y) > 0 and (x is y or torch.equal(x, y)))
               for x, y in zip(a, b))


def inverse_trace_weights(bases, like):
    """w with tr(P^-1) = sum(w d) for P^-1 = Q^-T diag(d) Q^-1 (row norms of Q^-1, as the lab's SOAPMetric)."""
    weights = torch.ones_like(like)
    for axis, q in enumerate(bases):
        if len(q):
            qinv = torch.linalg.solve(q, torch.eye(q.shape[0], dtype=q.dtype, device=q.device))
            view = [1]*like.ndim
            view[axis] = q.shape[0]
            weights = weights*qinv.square().sum(1).reshape(view)
    return weights


def transport(wt, old, new):
    """Rotated coordinates (r, *shape) of covectors under bases `old`, re-expressed under `new`."""
    for axis, (a, b) in enumerate(zip(old, new)):
        if len(a) == 0 or a is b:
            continue
        m = b.T@torch.linalg.solve(a.T, torch.eye(a.shape[0], dtype=a.dtype, device=a.device))
        wt = torch.tensordot(wt, m, dims=[[axis+1], [1]]).movedim(-1, axis+1)
    return wt


def _batch(z):
    return (z[None], True) if z.ndim == 2 else (z, False)


def left_singular(m, rank):
    """Top-r left singular vectors and squared singular values of m (b, p, n).

    Falls back to an eigendecomposition of the Gram matrix m m^T when LAPACK's SVD does not converge."""
    try:
        u, s, _ = torch.linalg.svd(m, full_matrices=False)
        return u[..., :rank], s[..., :rank].square()
    except torch.linalg.LinAlgError:
        eig, vectors = torch.linalg.eigh(m@m.transpose(1, 2))
        return vectors.flip(-1)[..., :rank], eig.flip(-1)[..., :rank].clamp_min(0)


def randomized_top(z, rank, generator, oversample, power):
    """Top-r left singular vectors and squared singular values of z, (count, n) or batched (b, count, n)."""
    z, single = _batch(z)
    b, count, n = z.shape
    k = min(rank+oversample, count, n)
    q, _ = torch.linalg.qr(z@torch.randn(b, n, k, generator=generator, dtype=z.dtype, device=z.device))
    for _ in range(power):
        q, _ = torch.linalg.qr(z@(z.transpose(1, 2)@q))
    r = min(rank, count, n)
    u, kappa = left_singular(q.transpose(1, 2)@z, r)
    w = q@u
    return (w[0], kappa[0]) if single else (w, kappa)


def exact_top(z, rank):
    z, single = _batch(z)
    r = min(rank, *z.shape[1:])
    w, kappa = left_singular(z, r)
    return (w[0], kappa[0]) if single else (w, kappa)


def track_step(z, held):
    """One power step from the held subspace, then Rayleigh-Ritz inside the new subspace."""
    z, single = _batch(z)
    held = held[None] if single else held
    q, _ = torch.linalg.qr(held)
    q, _ = torch.linalg.qr(z@(z.transpose(1, 2)@q))
    a = z.transpose(1, 2)@q
    eig, rotation = torch.linalg.eigh(a.transpose(1, 2)@a)
    eig, rotation = eig.flip(-1), rotation.flip(-1)
    w = q@rotation
    return (w[0], eig[0]) if single else (w, eig)


def invariance_residual(z, w):
    """|K W - W (W^T K W)|_F / |K W|_F for K = z z^T and orthonormal W (0 for an exact invariant subspace)."""
    z, single = _batch(z)
    w = w[None] if single else w
    kw = z@(z.transpose(1, 2)@w)
    norm = kw.flatten(1).norm(dim=1)
    res = (kw-w@(w.transpose(1, 2)@kw)).flatten(1).norm(dim=1)
    values = [float(r)/float(m) if float(m) > 0 else 0. for r, m in zip(res, norm)]
    return values[0] if single else values


def offdiagonal_ratio(y, axis):
    """|offdiag(S)|_F / |S|_F, S = sum_j Y_j,(axis) Y_j,(axis)^T: the probe partial trace along one mode.

    y has shape (n, *shape); `offdiagonal_ratios` takes a batch (b, n, *shape)."""
    return offdiagonal_ratios(y[None], axis)[0]


def offdiagonal_ratios(y, axis):
    a = y.movedim(axis+2, 2).reshape(y.shape[0], y.shape[1], y.shape[axis+2], -1)
    s = torch.einsum('bnik,bnjk->bij', a, a)
    total = s.flatten(1).norm(dim=1)
    off = (s-torch.diag_embed(s.diagonal(dim1=1, dim2=2))).flatten(1).norm(dim=1)
    return [float(o)/float(t) if float(t) > 0 else 0. for o, t in zip(off, total)]




CALIBRATION = ('damping', 'lr')
RHS = ('momentum', 'gradient')


class PARSECL(SOAP):
    """SOAP (upstream, unmodified update) plus the PARSEC-L shrinkage of every step.

    Call `step(probes=...)` with a dict {parameter: tensor (n, *parameter.shape)} on the steps where
    `wants_probes()` is true (or whenever new curvature should be used). `trace` overrides the probe
    trace estimate of tr(C) in mu. `selection` (a callable returning the loss on a selection batch at
    the current parameters) chooses the step length on that batch. `true_gradients` (per-example
    true-label gradients of the training batch) logs the optimism ratio r. Diagnostics of the last step
    are in `last_info`.
    """

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=-1, eps=1e-8, weight_decay=0.01,
                 precondition_frequency=10, max_precond_dim=10000, merge_dims=False, precondition_1d=False,
                 normalize_grads=False, data_format='channels_first', correct_bias=True, *,
                 rank=16, gamma=0.01, damping_floor=1e-8, calibration='damping', strength=1., rhs='momentum', top_rhs='momentum',
                 probe_every=10, refresh='extract', oversample=8, power_steps=2, seed=0, batched='auto', signals=True,
                 probe_until=None, norm_blend=1., select_grid=(-4, 8), select_refine=12, select_compare_soap=False,
                 select_handover=None, soap_refresh='periodic', rho_threshold=None, residual_threshold=None,
                 stop_energy=None, stop_patience=3):
        if merge_dims or normalize_grads:
            raise NotImplementedError('PARSEC-L supports neither merge_dims nor normalize_grads')
        if refresh not in REFRESH or soap_refresh not in SOAP_REFRESH or calibration not in CALIBRATION or rhs not in RHS:
            raise ValueError(f'refresh in {REFRESH}, soap_refresh in {SOAP_REFRESH}, calibration in {CALIBRATION}, rhs in {RHS}')
        if soap_refresh != 'periodic' and rho_threshold is None:
            raise ValueError("soap_refresh='triggered'/'both' needs rho_threshold")
        if rank < 0 or probe_every < 1 or gamma <= 0 or strength <= 0 or not 0 <= norm_blend <= 1:
            raise ValueError('rank >= 0, probe_every >= 1, gamma > 0, strength > 0 and 0 <= norm_blend <= 1 required')
        if probe_until is not None and probe_until < 1:
            raise ValueError('probe_until must be a positive step or None')
        if batched not in (True, False, 'auto'):
            raise ValueError("batched must be True, False or 'auto'")
        if top_rhs not in RHS or (top_rhs == 'gradient' and rhs == 'gradient'):
            raise ValueError("top_rhs in ('momentum', 'gradient'), and not together with rhs='gradient'")
        if select_handover is not None and not select_compare_soap:
            raise ValueError('select_handover needs select_compare_soap')
        if select_grid[0] > select_grid[1]:
            raise ValueError('select_grid must be (low, high) exponents of 2')
        super().__init__(params, lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps, weight_decay=weight_decay,
                         precondition_frequency=NEVER if soap_refresh == 'triggered' else precondition_frequency,
                         max_precond_dim=max_precond_dim, merge_dims=False, precondition_1d=precondition_1d,
                         normalize_grads=False, data_format=data_format, correct_bias=correct_bias)
        self.parsec = dict(rank=rank, gamma=gamma, damping_floor=damping_floor, calibration=calibration, strength=strength,
                           rhs=rhs, top_rhs=top_rhs, probe_every=probe_every, refresh=refresh, oversample=oversample, power_steps=power_steps,
                           seed=seed, batched=batched, signals=signals or soap_refresh != 'periodic', probe_until=probe_until,
                           norm_blend=norm_blend, select_grid=tuple(select_grid), select_refine=select_refine,
                           select_compare_soap=select_compare_soap, select_handover=select_handover,
                           soap_refresh=soap_refresh, rho_threshold=rho_threshold, residual_threshold=residual_threshold,
                           stop_energy=stop_energy, stop_patience=stop_patience)
        self.index = {p: i for i, p in enumerate(p for g in self.param_groups for p in g['params'])}
        self._t, self._mu, self._stopped, self._low, self._energies = 0, None, False, 0, []
        self._soap_wins, self._select_off = 0, False
        self._generator, self._recording, self._owners, self._records = None, False, {}, {}
        self.last_info = None

    # ----------------------------------------------------------------------------- schedule
    def wants_probes(self):
        """True if the next step is a scheduled probe step (SOAP's first step only initializes)."""
        if self._stopped or self.parsec['rank'] == 0:
            return False
        upcoming = self._t+1
        until = self.parsec['probe_until']
        return upcoming >= 2 and (upcoming-2) % self.parsec['probe_every'] == 0 and (until is None or upcoming <= until)

    def wants_selection(self):
        """False once the selection hand-over has switched the optimizer to plain SOAP."""
        return not self._select_off

    # ----------------------------------------------------------------------------- step
    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """Upstream calls this first with the update m~/d of a tensor; record it with the step's F0."""
        key = id(state)
        if self._recording and key in self._owners and key not in self._records and state.get('step', 0) > 0:
            p, group = self._owners[key]
            b1, b2 = group['betas']
            size, prox = group['lr'], group['lr']
            if group['correct_bias']:
                size = size*(1-b2**state['step'])**.5/(1-b1**state['step'])
                prox = prox*(1-b2**state['step'])**.5
            denominator = state['exp_avg_sq'].sqrt().add(group['eps'])
            record = {'param': p, 'group': group, 'x': grad, 'bases': state['Q'], 'denominator': denominator,
                      'step_size': size, 'beta_lr': prox,
                      'bc1': (1-b1**state['step']) if group['correct_bias'] else 1.}
            if self.parsec['rhs'] == 'gradient' or self.parsec['top_rhs'] == 'gradient':
                record['xg'] = self.project(p.grad, state)/denominator
            self._records[key] = record
        return super().project_back(grad, state, merge_dims=merge_dims, max_precond_dim=max_precond_dim)

    @torch.no_grad()
    def step(self, closure=None, probes=None, trace=None, selection=None, true_gradients=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = [p for g in self.param_groups for p in g['params'] if p.grad is not None]
        select = selection is not None and not self._select_off
        start = [p.detach().clone() for p in params] if select else None
        self._owners = {id(self.state[p]): (p, g) for g in self.param_groups for p in g['params'] if p.grad is not None}
        self._records, self._recording = {}, True
        try:
            dtype = params[0].dtype if params else torch.get_default_dtype()
            with default_dtype(dtype):
                super().step()
        finally:
            self._recording = False
        self._t += 1
        records = list(self._records.values())
        info = {'step': self._t, 'probe': None, 'correction_energy': None, 'rescale': None, 'handover': False,
                'select': None, 'optimism': None}
        until = self.parsec['probe_until']
        if until is not None and self._t > until and not self._stopped:
            self._handover(records)
            info['handover'] = True
        sides = []
        if records:
            scale, size = self._scale(records)
            info['scale'] = scale
            if probes is not None and self.parsec['rank'] > 0 and not self._stopped:
                info['probe'], sides = self._refresh(records, probes, trace, scale, size)
            info['correction_energy'], info['rescale'] = self._correct(records, scale)
            if self._mu is not None:
                self._energies.append(info['correction_energy'])
            if true_gradients is not None:
                info['optimism'] = self._optimism(records, true_gradients, scale)
            if select:
                info['select'] = self._select(records, params, start, selection)
        for p, axis in sides:
            self._refresh_side(p, axis)
        info['stopped'] = self._stopped
        self.last_info, self._records, self._owners = info, {}, {}
        return loss

    # ----------------------------------------------------------------------------- pieces
    def _pstate(self, p):
        return self.state[p].setdefault('parsec', {})

    def _beta(self, rec, scale):
        """Damping calibration 1/(mu scale), or learning-rate calibration c lr sqrt(1 - beta2^t).

        The latter makes SOAP's bias-corrected step -lr P_hat m_hat the proximal step for
        |dtheta|^2_{P_hat^-1} / (2 lr), so adding the curvature C gives kappa_k = c lr lambda_k(P_hat C)."""
        if self.parsec['calibration'] == 'lr':
            return self.parsec['strength']*rec['beta_lr']
        return 1/(self._mu*scale)

    def _handover(self, records):
        """Stop probing and drop the held curvature: plain SOAP from now on."""
        self._stopped, self._mu = True, None
        for p in self.index:
            self.state[p].get('parsec', {}).pop('wt', None)

    def _scale(self, records):
        """scale = D / sum_b tr(P_b^-1) at this step's F0 (the lab's SOAPMetric normalization)."""
        total, size = 0., 0
        for rec in records:
            ps = self._pstate(rec['param'])
            if not same_basis(ps.get('trace_bases'), rec['bases']):
                ps['trace_bases'], ps['trace_weights'] = rec['bases'], inverse_trace_weights(rec['bases'], rec['denominator'])
            total += float((ps['trace_weights']*rec['denominator']).sum())
            size += rec['x'].numel()
        return size/total, size

    def _held(self, rec):
        """Held low-rank factor in this step's whitened coordinates up to beta: (count, r), or None."""
        ps = self._pstate(rec['param'])
        wt = ps.get('wt')
        if wt is None or len(wt) == 0:
            return None
        if not same_basis(ps['bases'], rec['bases']):
            ps['wt'], ps['bases'] = transport(wt, ps['bases'], rec['bases']), rec['bases']
        return (ps['wt']/rec['denominator'].sqrt()).reshape(len(wt), -1).T

    def _refresh(self, records, probes, trace, scale, size):
        cfg = self.parsec
        if cfg['stop_energy'] is not None and self._energies:
            window = sum(self._energies)/len(self._energies)
            self._low = self._low+1 if window < cfg['stop_energy'] else 0
            if self._low >= cfg['stop_patience']:
                self._handover(records)
                return {'stopped_at_window_energy': window}, []
        self._energies = []
        used = [(rec, probes[rec['param']]) for rec in records if rec['param'] in probes]
        probe_trace = sum(float(u.square().sum()) for _, u in used)
        traced = probe_trace if trace is None else float(trace)
        self._mu = max(cfg['gamma']*traced/size, cfg['damping_floor'])
        if self._generator is None:
            self._generator = torch.Generator(device=records[0]['x'].device).manual_seed(cfg['seed'])
        batched = cfg['batched'] if cfg['batched'] != 'auto' else records[0]['x'].device.type != 'cpu'
        groups = defaultdict(list)
        for rec, u in used:
            key = (tuple(rec['param'].shape), tuple(len(q) > 0 for q in rec['bases']), len(u)) if batched else len(groups)
            groups[key].append((rec, u))
        rows, sides = {}, []
        for members in groups.values():
            group_rows, group_sides = self._extract(members, scale)
            rows.update(group_rows)
            sides += group_sides
        eta_lambda = [row['eta_lambda_max'] for row in rows.values() if 'eta_lambda_max' in row]
        info = {'mu': self._mu, 'beta_damping': 1/(self._mu*scale), 'probe_trace': probe_trace, 'trace_used': traced,
                'tensors': rows, 'groups': len(groups), 'eta_lambda_max': max(eta_lambda) if eta_lambda else 0.,
                'side_refreshes': [(self.index[p], axis) for p, axis in sides]}
        return info, sides

    def _extract(self, members, scale):
        """Top-r whitened curvature for tensors of equal shape, as one batch."""
        cfg = self.parsec
        recs = [rec for rec, _ in members]
        shape = tuple(recs[0]['param'].shape)
        u = torch.stack([v.to(rec['param'].dtype) for rec, v in members])
        y = rotate_batched(u, [rec['bases'] for rec in recs])
        root = torch.stack([rec['denominator'].sqrt() for rec in recs])
        b, n = u.shape[:2]
        betas = torch.tensor([self._beta(rec, scale) for rec in recs], dtype=u.dtype, device=u.device)
        view = (b,)+(1,)*(y.ndim-1)
        z = (betas.sqrt().reshape(view)*y/root[:, None]).reshape(b, n, -1).transpose(1, 2)
        r = min(cfg['rank'], *z.shape[1:])
        held = [self._held(rec) for rec in recs]
        rows = [{} for _ in recs]
        complete = all(h is not None for h in held) and len({h.shape for h in held if h is not None}) == 1
        if complete:
            orth = torch.linalg.qr(torch.stack(held))[0]
            for row, value in zip(rows, invariance_residual(z, orth)):
                row['residual_held'] = value
        mode = cfg['refresh']
        if mode == 'track':
            if complete and held[0].shape[1] == r:
                w, kappa = track_step(z, torch.stack(held))
                tracked = invariance_residual(z, w)
                for row, value in zip(rows, tracked):
                    row['residual_tracked'] = value
                if cfg['residual_threshold'] is not None and max(tracked) > cfg['residual_threshold']:
                    mode = 'extract'
            else:
                mode = 'extract'
        if mode == 'extract':
            w, kappa = randomized_top(z, r, self._generator, cfg['oversample'], cfg['power_steps'])
        elif mode == 'exact':
            w, kappa = exact_top(z, r)
        kappa = kappa.clamp_min(0)
        residuals = invariance_residual(z, w) if r else [0.]*b
        wt = ((w*kappa.sqrt()[:, None, :]).transpose(1, 2).reshape((b, r)+shape)*root[:, None]
              / betas.sqrt().reshape((b,)+(1,)*(len(shape)+1))) if r else None
        sides, out = [], {}
        for i, rec in enumerate(recs):
            p, ps, row = rec['param'], self._pstate(rec['param']), rows[i]
            kmax = float(kappa[i].max()) if r else 0.
            row.update(mode=mode, rank=r, kappa_max=kmax, kappa_min=float(kappa[i].min()) if r else 0., residual=residuals[i],
                       eta_lambda_max=kmax*rec['beta_lr']/float(betas[i]))
            ps['wt'], ps['bases'] = (wt[i] if r else None), rec['bases']
            out[self.index[p]] = row
        if cfg['signals']:
            for axis, q in enumerate(recs[0]['bases']):
                if len(q) < 2:
                    continue
                for i, rho in enumerate(offdiagonal_ratios(y, axis)):
                    rec = recs[i]
                    p, ps, row = rec['param'], self._pstate(rec['param']), rows[i]
                    floors, floor_bases = ps.setdefault('rho_floor', {}), ps.setdefault('rho_bases', {})
                    q_i = rec['bases'][axis]
                    if floor_bases.get(axis) is not q_i:
                        floors[axis], floor_bases[axis] = rho, q_i
                    row[f'rho_{axis}'], row[f'rho_rise_{axis}'] = rho, rho-floors[axis]
                    refreshed_now = self.state[p]['Q'] is not rec['bases']
                    if (cfg['soap_refresh'] != 'periodic' and not refreshed_now
                            and rho-floors[axis] > cfg['rho_threshold']):
                        sides.append((p, axis))
        return out, sides

    def _shrink(self, rec, scale, z):
        """(V (I + V^T V)^-1 V^T z, V, Cholesky factor) for this tensor's held curvature, or None."""
        held = self._held(rec) if self._mu is not None and not self._stopped else None
        if held is None:
            return None
        v = self._beta(rec, scale)**.5*held.T
        chol = torch.linalg.cholesky(torch.eye(len(v), dtype=v.dtype, device=v.device)+v@v.T)
        return v.T@torch.cholesky_solve((v@z)[:, None], chol)[:, 0], v, chol

    def _correct(self, records, scale):
        """Apply the shrinkage, the gradient RHS and the optional global norm rescale to this step's update.

        Returns c = 1 - |z'|^2/|z|^2 over all tensors and the rescale factor (None if norm_blend = 0)."""
        before = after = 0.
        gradient = self.parsec['rhs'] == 'gradient' and not self._stopped
        top_gradient = self.parsec['top_rhs'] == 'gradient' and not self._stopped
        shifts = []
        for rec in records:
            used = rec['xg'] if gradient else rec['x']
            root = rec['denominator'].sqrt()
            z = (used*root).reshape(-1)
            norm = float(z.square().sum())
            before += norm
            shrunk = self._shrink(rec, scale, z)
            if shrunk is None:
                after += norm
                shifts.append(used-rec['x'] if gradient else None)
                continue
            shrink, v, chol = shrunk
            if top_gradient:
                # Replace the momentum's component in span(V) by the bias-corrected current gradient's, then brake.
                basis = torch.linalg.qr(v.T)[0]
                zg = (rec['xg']*rec['bc1']*root).reshape(-1)
                target = z+basis@(basis.T@(zg-z))
                new = target-v.T@torch.cholesky_solve((v@target)[:, None], chol)[:, 0]
                after += float(new.square().sum())
                shifts.append((new-z).reshape(rec['x'].shape)/root)
                continue
            after += float((z-shrink).square().sum())
            dx = -(shrink.reshape(rec['x'].shape)/root)
            shifts.append((used-rec['x'])+dx if gradient else dx)
        energy = 1-after/before if before > 0 else 0.
        blend = self.parsec['norm_blend']
        if all(s is None for s in shifts):
            return energy, (1. if blend else None)
        if blend == 0:
            for rec, dx in zip(records, shifts):
                if dx is not None:
                    self._apply(rec, super().project_back(dx, {'Q': rec['bases']}))
            return energy, None
        native = corrected = 0.
        steps = []
        for rec, dx in zip(records, shifts):
            n = super().project_back(rec['x'], {'Q': rec['bases']})*rec['step_size']
            c = n if dx is None else super().project_back(rec['x']+dx, {'Q': rec['bases']})*rec['step_size']
            native += float(n.square().sum())
            corrected += float(c.square().sum())
            steps.append((n, c))
        factor = (native/corrected)**(blend/2) if corrected > 0 else 1.
        for rec, (n, c) in zip(records, steps):
            self._apply(rec, (factor*c-n)/rec['step_size'])
        return energy, factor

    def _apply(self, rec, direction):
        """Add -step_size * direction to the parameter, consistent with upstream's decoupled weight decay."""
        group = rec['group']
        rec['param'].add_(direction, alpha=-rec['step_size']*(1-group['lr']*group['weight_decay']))

    def _select(self, records, params, start, selection):
        """Choose the step length alpha on a selection batch: theta = start + alpha (theta_step - start).

        Grid: 0 and 2^k for k in select_grid, then golden-section refinement in log2(alpha) around the
        best positive grid point. With select_compare_soap, SOAP's own step from the same start is also
        evaluated and taken if better; select_handover consecutive SOAP wins switch to plain SOAP."""
        cfg = self.parsec
        delta = [p.detach()-s for p, s in zip(params, start)]
        final = [p.detach().clone() for p in params]

        def place(alpha):
            for p, s, d, f in zip(params, start, delta, final):
                p.copy_(f if alpha == 1. else s+alpha*d)

        def evaluate(alpha):
            place(alpha)
            return float(selection())

        low, high = cfg['select_grid']
        grid = [0.]+[2.**k for k in range(low, high+1)]
        values = [evaluate(a) for a in grid]
        best = min(range(len(grid)), key=lambda i: values[i])
        alpha, value, evaluations = grid[best], values[best], len(grid)
        if best > 0 and cfg['select_refine']:
            left = low-1 if best == 1 else low+best-2
            right = low+best if best < len(grid)-1 else high+1
            golden = (5**.5-1)/2
            a, b = float(left), float(right)
            c, d = b-golden*(b-a), a+golden*(b-a)
            fc, fd = evaluate(2.**c), evaluate(2.**d)
            evaluations += 2
            for _ in range(cfg['select_refine']-2):
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b-golden*(b-a)
                    fc = evaluate(2.**c)
                else:
                    a, c, fc = c, d, fd
                    d = a+golden*(b-a)
                    fd = evaluate(2.**d)
                evaluations += 1
            for candidate, f in [(2.**c, fc), (2.**d, fd)]:
                if f < value:
                    alpha, value = candidate, f
        winner, soap_value = 'parsec', None
        if cfg['select_compare_soap']:
            by_param = {rec['param']: rec for rec in records}
            soap = []
            for p, s in zip(params, start):
                t = s.clone()
                rec = by_param.get(p)
                if rec is not None:
                    group = rec['group']
                    t.add_(super().project_back(rec['x'], {'Q': rec['bases']}), alpha=-rec['step_size'])
                    if group['weight_decay'] > 0.0:
                        t.add_(t, alpha=-group['lr']*group['weight_decay'])
                soap.append(t)
            for p, t in zip(params, soap):
                p.copy_(t)
            soap_value = float(selection())
            evaluations += 1
            if soap_value < value:
                winner = 'soap'
        self._soap_wins = self._soap_wins+1 if winner == 'soap' else 0
        if winner == 'parsec':
            place(alpha)
        handed = cfg['select_handover'] is not None and self._soap_wins >= cfg['select_handover']
        if handed:
            self._handover(records)
            self._select_off = True
        return {'alpha': alpha, 'value': value, 'soap_value': soap_value, 'winner': winner,
                'soap_wins': self._soap_wins, 'evaluations': evaluations, 'handover': handed}

    def _optimism(self, records, gradients, scale):
        """Optimism ratio r = tr(A^-1 Sigma)/B / (g1^T A^-1 g2) for SOAP's metric and this step's PARSEC-L operator.

        `gradients` are per-example true-label gradients of the training batch (rows averaging to its
        gradient). Sigma is their sample covariance; the signal is cross-fitted from the two halves."""
        started = time.perf_counter()
        totals = {'soap': [0., 0.], 'parsec_l': [0., 0.]}
        for rec in records:
            p = rec['param']
            if p not in gradients:
                continue
            g = gradients[p].to(p.dtype)
            batch, half = len(g), len(g)//2
            y = (rotate(g, rec['bases'])/rec['denominator'].sqrt()).reshape(batch, -1)
            centered = y-y.mean(0)
            a, b = y[:half].mean(0), y[half:2*half].mean(0)
            optimism, signal = float(centered.square().sum())/(batch-1)/batch, float(a@b)
            totals['soap'][0] += optimism
            totals['soap'][1] += signal
            shrunk = self._shrink(rec, scale, a)
            if shrunk is not None:
                _, v, chol = shrunk
                vc = centered@v.T
                optimism -= float((vc*torch.cholesky_solve(vc.T, chol).T).sum())/(batch-1)/batch
                signal -= float((v@a)@torch.cholesky_solve((v@b)[:, None], chol)[:, 0])
            totals['parsec_l'][0] += optimism
            totals['parsec_l'][1] += signal
        out = {f'r_{name}': (o/s if s > 0 else float('inf')) for name, (o, s) in totals.items()}
        out.update({f'{kind}_{name}': value for name, (o, s) in totals.items() for kind, value in [('optimism', o), ('signal', s)]})
        out['seconds'] = time.perf_counter()-started
        return out

    def _refresh_side(self, p, axis):
        """Upstream's power-iteration QR refresh for one mode only, with the momentum carried over."""
        state = self.state[p]
        m, o = state['GG'][axis], state['Q'][axis]
        float_data = m.dtype == torch.float
        mf, of = m.float(), o.float()
        order = torch.argsort(torch.diag(of.T@mf@of), descending=True)
        state['exp_avg_sq'] = state['exp_avg_sq'].index_select(axis, order)
        q, _ = torch.linalg.qr(mf@of[:, order])
        if not float_data:
            q = q.to(o.device).type(o.dtype)
        momentum = super().project_back(state['exp_avg'], state)
        bases = list(state['Q'])
        bases[axis] = q
        state['Q'] = bases
        state['exp_avg'] = self.project(momentum, state)

    # ----------------------------------------------------------------------------- persistence
    def state_dict(self):
        out = super().state_dict()
        out['parsec'] = {'t': self._t, 'mu': self._mu, 'stopped': self._stopped, 'low': self._low,
                         'energies': list(self._energies), 'soap_wins': self._soap_wins, 'select_off': self._select_off,
                         'generator': None if self._generator is None else self._generator.get_state()}
        return out

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        saved = state_dict.pop('parsec', None)
        super().load_state_dict(state_dict)
        if saved is not None:
            self._t, self._mu, self._stopped, self._low = saved['t'], saved['mu'], saved['stopped'], saved['low']
            self._energies = list(saved['energies'])
            self._soap_wins, self._select_off = saved.get('soap_wins', 0), saved.get('select_off', False)
            if saved['generator'] is not None:
                device = next(iter(self.index)).device
                self._generator = torch.Generator(device=device)
                self._generator.set_state(saved['generator'])

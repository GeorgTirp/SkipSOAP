"""Per-state GGN spectra and anchor-pair metrics (PROTOCOL.md, section 3).

The raw per-tensor GGN block C_b of a fixed batch, in parameter coordinates (`ggn.accumulate` without SOAP contexts),
for the matrix tensors. Per tensor: top eigenpairs, trace, Ritz residual, power-law fits. Against stored anchor
eigenspaces: capture_k = tr(U_k(a)^T C U_k(a)) / sum_{i<=k} lambda_i and overlap_k = ||U_k(a)^T U_k||_F^2 / k.
"""
import torch

import ggn
import spectral

SPECTRUM, RANK, CHUNK, BUDGET = 200, 16, 128, int(2.5*2**30)
KS = [1, 4, 16, 64, 200]


def matrix_specs(registry):
    """Two-dimensional tensors with both sides above 1 (as bulk_krylov_20261014)."""
    return [s for s in registry.specs if len(s.shape) == 2 and min(s.shape) > 1]


def groups(specs, budget_bytes, element_bytes):
    """Consecutive specs whose dense blocks fit the budget together (ggn.groups_for over a subset)."""
    out, current, used = [], [], 0
    for spec in specs:
        size = spec.count**2*element_bytes
        if current and used+size > budget_bytes:
            out.append(current)
            current, used = [], 0
        current.append(spec)
        used += size
    if current:
        out.append(current)
    return out


def capture(anchor_vectors, matrix, values, ks=KS):
    """{k: tr(U_k^T C U_k) / sum_{i<=k} values_i} for the anchor's orthonormal columns U (in rank order)."""
    projected = (anchor_vectors.T@(matrix@anchor_vectors)).double()
    diagonal = projected.diagonal()
    values = values.double()
    out = {}
    for k in ks:
        k_eff = min(k, anchor_vectors.shape[1], len(values))
        denominator = float(values[:k_eff].sum())
        out[k] = float(diagonal[:k_eff].sum())/denominator if denominator > 0 else None
    return out


def overlap(anchor_vectors, vectors, ks=KS):
    """{k: ||U_k(a)^T U_k||_F^2 / k}."""
    out = {}
    for k in ks:
        k_eff = min(k, anchor_vectors.shape[1], vectors.shape[1])
        out[k] = float((anchor_vectors[:, :k_eff].T.double()@vectors[:, :k_eff].double()).square().sum())/k_eff
    return out


def tensor_record(matrix, generator):
    """Spectrum summary of one dense symmetric PSD block; returns (record, eigenvectors, eigenvalues in float64)."""
    values, vectors = spectral.top_eigenpairs(matrix, SPECTRUM, generator, oversample=SPECTRUM)
    last = vectors[:, -1]
    ritz = float((matrix@last-values[-1]*last).norm()/values[-1]) if float(values[-1]) > 0 else None
    values64 = values.double().cpu()
    trace = float(matrix.diagonal().double().sum())
    fits = {}
    for label, (first, last_rank) in {'all': (1, SPECTRUM), 'top': (1, RANK), 'bulk': (RANK+1, SPECTRUM)}.items():
        alpha, r2 = spectral.loglog_fit(values64, first, last_rank)
        fits[label] = {'alpha': alpha, 'r2': r2}
    record = {'values': [float(f'{v:.7g}') for v in values64.tolist()], 'trace': trace, 'ritz_residual_last': ritz,
              'fits': fits, 'top16_trace_share': float(values64[:RANK].sum())/trace if trace > 0 else None}
    return record, vectors, values64


def state_pass(model, registry, flat, x, generator, anchors=(), keep=False, floor_of=None):
    """One dense pass over the matrix tensors on batch x.

    anchors: [(step, {name: U on CPU})] whose capture/overlap against this state are recorded.
    keep: return this state's eigenvectors on the CPU (for an anchor).
    floor_of: {name: U on CPU} of the same weights on the other batch; recorded as 'floor'.
    Returns ({name: record}, {name: U} or None)."""
    specs = matrix_specs(registry)
    records, kept = {}, ({} if keep else None)
    for group in groups(specs, BUDGET, flat.element_size()):
        blocks, _ = ggn.accumulate(model, registry, flat, x, group, CHUNK, all_parameters=False, contexts=None)
        for spec in group:
            matrix = ggn.symmetrize_(blocks.pop(spec.name))
            record, vectors, values = tensor_record(matrix, generator)
            record['pairs'] = {}
            for step, stored in anchors:
                u = stored[spec.name].to(matrix.device)
                record['pairs'][str(step)] = {'capture': capture(u, matrix, values), 'overlap': overlap(u, vectors)}
            if floor_of is not None:
                u = floor_of[spec.name].to(matrix.device)
                record['floor'] = {'capture': capture(u, matrix, values), 'overlap': overlap(u, vectors)}
            if keep:
                kept[spec.name] = vectors.cpu()
            records[spec.name] = record
            del matrix, vectors
        del blocks
    return records, kept

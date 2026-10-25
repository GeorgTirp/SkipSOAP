"""Figures and numerical checks for spectral_frame.tex and workbook.tex (Part I). Synthetic and seeded; reads no lab data.

Usage (from residual-curvature-lab/):
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ideas/spectral_frame/make_figures.py
Writes ideas/spectral_frame/figures/{faces,resolution}.pdf and ideas/spectral_frame/generated/numbers.tex
(LaTeX macros for every number the note and the workbook quote from these computations).
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES, GENERATED = HERE/'figures', HERE/'generated'
SEED = 20261023
ZETA2 = math.pi**2/6

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'figure.dpi': 150})


def law(dim, alpha, c=1.):
    """Pure power law λ_k = c k^{−α}, k = 1..dim (descending)."""
    return c*np.arange(1, dim+1, dtype=float)**-alpha


def harmonic(n):
    return float(np.sum(1/np.arange(1, n+1, dtype=float)))


def goe(dim, rng):
    """Symmetric Gaussian matrix: off-diagonal variance 1, diagonal variance 2."""
    z = rng.standard_normal((dim, dim))
    return (z+z.T)/math.sqrt(2)


def moving_median(x, width=11):
    half = width//2
    padded = np.pad(x, half, mode='edge')
    return np.array([np.median(padded[i:i+width]) for i in range(len(x))])


def figure_faces():
    dim, alphas = 4096, [0.7, 1.0, 2.0]
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
    ranks = np.arange(1, dim+1)
    for alpha in alphas:
        lam = law(dim, alpha)
        label = rf'$\alpha = {alpha:g}$'
        ax[0, 0].loglog(ranks, lam, label=label)
        ax[0, 1].loglog(lam, ranks, label=label)  # N(λ_k) = k for a strictly decreasing spectrum
        logs = np.log10(lam)
        bins = np.arange(logs.min()-1e-9, 1e-9+0.1, 0.1)
        counts, edges = np.histogram(logs, bins=bins)
        ax[1, 0].step(edges[:-1], np.maximum(counts, 0.5), where='post', label=label)
        octave = np.floor(np.log2(lam[0]/lam)+1e-12).astype(int)
        share = np.bincount(octave, weights=lam)/lam.sum()
        full = np.arange(len(share)) < octave.max()  # drop the last, partial octave
        keep = full & (share > 0)  # and the empty octaves at the discrete top (α = 2)
        ax[1, 1].plot(np.arange(len(share))[keep], share[keep], marker='o', ms=2.5, label=label)
    ax[0, 0].set(xlabel='rank $k$', ylabel=r'$\lambda_k$', title=r'(a) rank law $\lambda_k = k^{-\alpha}$')
    ax[0, 1].set(xlabel=r'threshold $t$', ylabel=r'$N(t) = \#\{k:\lambda_k \geq t\}$',
                 title=r'(b) counting function, slope $-1/\alpha$')
    ax[1, 0].set(yscale='log', xlabel=r'$\log_{10}\lambda$', ylabel='eigenvalues per 0.1 decade',
                 title='(c) where the eigen-directions are')
    ax[1, 1].set(yscale='log', xlabel=r'octaves below $\lambda_1$', ylabel='share of the trace',
                 title='(d) where the trace is')
    for a in ax.flat:
        a.legend(frameon=False)
    fig.suptitle(f'Four faces of a pure power law (D = {dim})', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES/'faces.pdf')
    plt.close(fig)


def figure_resolution(numbers):
    dim, alpha, c, eps, draws = 2000, 1.0, 1.0, 1e-5, 3
    rng = np.random.default_rng(SEED)
    lam = law(dim, alpha, c)
    ranks = np.arange(1, dim+1)
    cuts = np.unique(np.round(np.logspace(0, math.log10(dim-1), 40)).astype(int))
    # A = diag(λ) without loss of generality: the perturbation law is rotation invariant.
    rel_val, sines, hard, soft, norms = [], [], [], [], []
    for _ in range(draws):
        e = eps*goe(dim, rng)
        w, v = np.linalg.eigh(np.diag(lam)+e)
        w, v = w[::-1], v[:, ::-1]
        rel_val.append(np.abs(w-lam)/lam)
        sines.append(np.sqrt(np.clip(1-np.diag(v)**2, 0, None)))
        inside = np.diagonal(np.cumsum(np.cumsum(v**2, 0), 1))  # ||Π_k(A) Π_k(B)||_F² for each k
        hard.append(np.sqrt(np.clip(2*(ranks-inside)/ranks, 0, None)))  # ||Π_k(B) − Π_k(A)||_F / ||Π_k(A)||_F
        weights = v**2  # weights[j, i] = (e_jᵀ v_i)²
        row = []
        for k in cuts:  # soft cut f_k(λ) = λ/(λ + λ_k) at the same place
            fa, fb = lam/(lam+lam[k-1]), w/(w+lam[k-1])
            cross = fa@weights@fb  # tr f_k(A) f_k(B)
            row.append(math.sqrt(max(fa@fa+fb@fb-2*cross, 0.))/math.sqrt(fa@fa))
        soft.append(row)
        norms.append((np.linalg.norm(e, 2), np.linalg.norm(e)))
    rel_val, sines, hard, soft = (np.median(np.array(x), 0) for x in (rel_val, sines, hard, soft))
    eta = float(np.median([n[0] for n in norms]))
    frob = float(np.median([n[1] for n in norms]))
    gaps = lam[:, None]-lam[None, :]
    np.fill_diagonal(gaps, np.inf)
    first_order_sine = eps*np.sqrt(np.sum(gaps**-2., 1))  # ε (Σ_{j≠k} (λ_k − λ_j)^{−2})^{1/2}
    weyl = eta/lam
    k_vec = int(ranks[np.argmax(moving_median(sines) >= 0.5)])
    rel_smooth = moving_median(rel_val)
    k_val = int(ranks[np.argmax(rel_smooth >= 0.5)]) if (rel_smooth >= 0.5).any() else None

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax[0].loglog(ranks, rel_val, lw=0.6, label='simulated (median of 3)')
    ax[0].loglog(ranks, np.minimum(weyl, 10), 'k--', lw=0.8, label=r'Weyl bound $\|E\|_{op}/\lambda_k$')
    ax[0].loglog(ranks, math.sqrt(2)*eps/lam, 'k:', lw=0.8, label=r'isolated: $\sqrt{2}\varepsilon/\lambda_k$')
    ax[0].set(xlabel='rank $k$', ylabel=r'$|\Delta\lambda_k|/\lambda_k$', title='(a) eigenvalues')
    ax[1].loglog(ranks, sines, lw=0.6, label='simulated (median of 3)')
    ax[1].loglog(ranks, np.minimum(first_order_sine, 1), 'k:', lw=0.8, label='first-order formula')
    ax[1].set(xlabel='rank $k$', ylabel=r'$\sin\angle(u_k(A), u_k(B))$', title='(b) eigenvectors', ylim=(1e-4, 1.5))
    ax[2].loglog(ranks, np.maximum(hard, 1e-4), lw=0.8, label=r'hard cut $\Pi_k(A)$')
    ax[2].loglog(cuts, np.maximum(soft, 1e-4), marker='o', ms=2, lw=0.8, label=r'soft cut $A(A+\lambda_k I)^{-1}$')
    ax[2].set(xlabel='cut position $k$', ylabel='relative Frobenius error', title='(c) hard vs soft cut',
              ylim=(1e-4, 2))
    for a in ax:
        a.legend(frameon=False, loc='best')
    fig.suptitle(rf'Pure law $\alpha=1$, $D={dim}$, symmetric Gaussian perturbation with entry scale '
                 rf'$\varepsilon=10^{{-5}}$ ($\|E\|_{{op}}\approx {eta:.1e}$)', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES/'resolution.pdf')
    plt.close(fig)

    k_vec_pred = (0.5*alpha*c/(math.sqrt(2*ZETA2)*eps))**(1/(alpha+1))
    at = lambda k: float(soft[int(np.argmin(np.abs(cuts-k)))])
    nearest = lambda k: int(cuts[int(np.argmin(np.abs(cuts-k)))])
    numbers.update({
        'NumSimDim': f'{dim:,}', 'NumSimEtaScaled': f'{eta*1e4:.1f}', 'NumSimFrob': f'{frob:.3f}',
        'NumKvecSim': f'{k_vec}', 'NumKvecPred': f'{k_vec_pred:.0f}',
        'NumKvalSim': f'{k_val:,}' if k_val else f'beyond {dim:,}',
        'NumKvalWeyl': f'{0.5*c/eta:.0f}',
        # Worst-case eigenvector limit from Weyl's bound and the one-vector sin-theta theorem (workbook).
        'NumKvecWorst': f'{(0.5*alpha*c/(2**(alpha+2)*eta))**(1/(alpha+1)):.1f}',
        'NumNeighbourRank': f'{alpha*math.sqrt(dim/(2*ZETA2)):.0f}',
        'NumRelValTopScaled': f'{float(np.max(rel_val[:100]))*1e3:.1f}',
        'NumHardSixteen': f'{float(hard[15]):.3f}', 'NumHardTwoHundred': f'{float(hard[199]):.2f}',
        'NumHardThousand': f'{float(hard[999]):.2f}',
        'NumSoftSixteen': f'{at(16):.4f}', 'NumSoftTwoHundred': f'{at(200):.3f}', 'NumSoftThousand': f'{at(1000):.3f}',
        'NumSoftCuts': f'{nearest(16)}, {nearest(200)} and {nearest(1000)}',
        'NumCrossCut': f'{int(cuts[np.argmax(soft < hard[cuts-1])])}'})


def checks(numbers):
    shares = {d: harmonic(16)/harmonic(d) for d in [640, 3072, 4096, 4160, 16384]}
    numbers.update({'NumShareSixForty': f'{shares[640]:.3f}', 'NumShareFourK': f'{shares[4096]:.3f}',
                    'NumShareSixteenK': f'{shares[16384]:.3f}', 'NumShareThreeK': f'{shares[3072]:.3f}'})
    for d, name in [(4096, 'FourK'), (16384, 'SixteenK')]:
        numbers[f'NumKant{name}'] = f'{harmonic(d)*(d+1)/(2*d):.2f}'
        numbers[f'NumFrameParams{name}'] = f'{d*(d-1)//2:,}'

    # Proposition 5.2: Var of the Rayleigh quotient of a uniform unit vector = 2 Var_mu(lambda)/(D + 2).
    lam = law(4096, 1.)
    rel_sd = math.sqrt(2/(4096+2))*lam.std()/lam.mean()
    numbers['NumHaarRelSd'] = f'{rel_sd:.3f}'
    rng = np.random.default_rng(SEED+1)
    small, draws = law(256, 1.), 200_000
    values = []
    for _ in range(draws//20_000):
        g = rng.standard_normal((20_000, 256))
        values.append((g**2@small)/np.sum(g**2, 1))
    values = np.concatenate(values)
    predicted = 2*small.var()/(256+2)
    numbers.update({'NumHaarMCDraws': f'{draws:,}', 'NumHaarMCVarRatio': f'{values.var()/predicted:.4f}',
                    'NumHaarMCMeanRatio': f'{values.mean()/small.mean():.4f}'})

    # Proposition 4.5: effective dimension under damping.
    k = np.arange(1, 10_000_001, dtype=float)
    exact = float(np.sum(1/(1+1e-6*k**2)))
    numbers.update({'NumDdeltaExact': f'{exact:.1f}', 'NumDdeltaAsym': f'{(math.pi/2)/math.sin(math.pi/2)*1e3:.1f}'})
    d, delta = 4096, 1e-3
    exact_log = float(np.sum(1/(1+delta*np.arange(1, d+1))))
    numbers.update({'NumDdeltaLogExact': f'{exact_log:.1f}', 'NumDdeltaLogAsym': f'{math.log(1+d*delta)/delta:.1f}'})

    # Remark 3.8: Kronecker product of two alpha = 1 laws (64 x 256), apparent exponent on ranks 17-200.
    kron = np.sort(np.outer(law(64, 1.), law(256, 1.)).ravel())[::-1]
    window = np.arange(17, 201)
    slope = np.polyfit(np.log(window), np.log(kron[window-1]), 1)[0]
    x_mid = math.exp(np.mean(np.log(window)))
    x = x_mid/math.log(x_mid)
    numbers.update({'NumKronSlope': f'{-slope:.2f}', 'NumKronPred': f'{1/(1+1/math.log(x)):.2f}'})


def workbook_checks(numbers):
    """Numbers quoted only in the workbook's exercises."""
    # Exercise on reading the exponent: alpha = 1, D = 4,096; directions and trace within a factor 10 and 1,000 of the top.
    d = 4096
    for factor, name in [(10, 'Ten'), (1000, 'Thousand')]:
        numbers[f'NumDirWithin{name}'] = f'{100*factor/d:.2g}'  # percent
        numbers[f'NumTraceWithin{name}'] = f'{harmonic(factor)/harmonic(d):.3f}'
    # Power sums with s = 2: bound and true value of the share beyond rank 16.
    k = np.arange(1, 17, dtype=float)
    numbers['NumTailBoundTwo'] = f'{(1/16)/ZETA2:.3f}'
    numbers['NumTailTrueTwo'] = f'{(ZETA2-float(np.sum(k**-2.)))/ZETA2:.3f}'
    # 2 x 2 example: diag(1, 0.99) plus 0.01 on the off-diagonal.
    gap, delta = 0.01, 0.01
    numbers['NumTwoShift'] = f'{math.sqrt(gap**2/4+delta**2)-gap/2:.4f}'
    numbers['NumTwoAngle'] = f'{math.degrees(0.5*math.atan(2*delta/gap)):.1f}'
    # Kantorovich's worst case for a pure law on D = 4,096 ranks: kappa = D.
    numbers['NumKantWorstFourK'] = f'{(d+1)**2/(4*d):,.0f}'


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    numbers = {}
    figure_faces()
    figure_resolution(numbers)
    checks(numbers)
    workbook_checks(numbers)
    lines = ['% Generated by make_figures.py; do not edit by hand.']
    lines += [rf'\newcommand{{\{name}}}{{{value.replace(",", "{,}")}}}' for name, value in numbers.items()]
    (GENERATED/'numbers.tex').write_text('\n'.join(lines)+'\n')
    print('\n'.join(f'{k} = {v}' for k, v in numbers.items()))


if __name__ == '__main__':
    main()

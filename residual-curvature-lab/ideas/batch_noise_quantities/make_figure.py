"""Figure and numerical checks for three_quantities.tex. Synthetic and seeded; reads no lab data.

Usage (from residual-curvature-lab/):
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ideas/batch_noise_quantities/make_figure.py
Writes figures/survival_cost.pdf and generated/numbers.tex (LaTeX macros for every synthetic number the note quotes).
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 20261023
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 7})


def kappa(n, dim):
    """Fraction of the Newton gain kept under curvature noise from n Gaussian samples (exact, n > dim + 3)."""
    n = np.asarray(n, dtype=float)
    out = (n-dim)*(n-dim-3)/((n-dim-1)*(n-1))
    return np.where(n > dim+3, out, np.nan)


def check_wishart(numbers, rng):
    dim, n, draws = 20, 60, 20_000
    m1 = m2 = 0.
    for _ in range(draws):
        z = rng.standard_normal((n, dim))
        inv = np.linalg.inv(z.T@z/n)
        m1 += np.trace(inv)/dim/draws
        m2 += np.trace(inv@inv)/dim/draws
    numbers.update({'NumWishDraws': f'{draws:,}', 'NumWishMOne': f'{m1:.4f}', 'NumWishMOnePred': f'{n/(n-dim-1):.4f}',
                    'NumWishMTwo': f'{m2:.4f}',
                    'NumWishMTwoPred': f'{n*n*(n-1)/((n-dim)*(n-dim-1)*(n-dim-3)):.4f}'})


def check_product(numbers, rng):
    """Newton step with curvature from n samples and an independent gradient with noise ratio r (whitened)."""
    dim, n, r, draws = 50, 120, 1.0, 20_000
    m1 = n/(n-dim-1)
    m2 = n*n*(n-1)/((n-dim)*(n-dim-1)*(n-dim-3))
    alpha = m1/(m2*(1+r))
    u = np.zeros(dim)
    u[0] = 1.
    removed = 0.
    for _ in range(draws):
        z = rng.standard_normal((n, dim))
        noisy = u+rng.standard_normal(dim)*math.sqrt(r/dim)
        step = np.linalg.solve(z.T@z/n, noisy)
        removed += (1-np.sum((u-alpha*step)**2))/draws
    numbers.update({'NumProdSim': f'{removed:.4f}', 'NumProdPred': f'{float(kappa(n, dim))/(1+r):.4f}'})


def check_share(numbers, rng):
    """Same-batch step size: share 2 u.g_B/|g_B|^2 - 1 against (1 - r)/(1 + r), whitened Newton metric."""
    dim, draws = 200, 20_000
    for r, name in [(0.5, 'Half'), (2.0, 'Two')]:
        u = np.zeros(dim)
        u[0] = 1.
        noise = rng.standard_normal((draws, dim))*math.sqrt(r/dim)
        g = u+noise
        share = 2*(g@u)/np.sum(g*g, 1)-1
        numbers.update({f'NumShareSim{name}': f'{share.mean():.3f}', f'NumSharePred{name}': f'{(1-r)/(1+r):.3f}'})


def figure(numbers):
    dim, k = 256, 2000.
    b_ov = dim/3
    batch = np.logspace(math.log10(dim+4), 5, 600)
    grad = 1/(1+k/batch)
    curv = kappa(batch, dim)
    both = grad*curv
    share = np.clip((1-k/batch)/(1+k/batch), -1, 1)
    cost = dim**3/3+dim**2*batch
    eff_grad = grad/cost
    eff_both = both/cost
    b_grad = math.sqrt(k*b_ov)
    b_both = float(batch[np.nanargmax(eff_both)])
    fine = np.logspace(0, 6, 200_001)
    b_grad_numeric = float(fine[np.argmax((1/(1+k/fine))/(dim**3/3+dim**2*fine))])

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax[0].semilogx(batch, grad, label=r'gradient noise: $1/(1+r)$')
    ax[0].semilogx(batch, curv, label=r'curvature noise: $\kappa(n{=}B, D)$')
    ax[0].semilogx(batch, both, 'k', lw=1.2, label='both (independent batches)')
    ax[0].semilogx(batch, share, '--', color='C3', label=r'same-batch step size: $(1-r)/(1+r)$')
    for x, text in [(k, r'$B=K_1$'), (dim, r'$B=D$')]:
        ax[0].axvline(x, color='grey', lw=0.6, ls=':')
        ax[0].text(x*1.08, -0.92, text, fontsize=7, color='grey')
    ax[0].set(xlabel='batch size $B$', ylabel='share of the Newton gain kept', ylim=(-1, 1.02),
              title='(a) what survives of the quadratic model')
    ax[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    ax[1].semilogx(batch, eff_grad/np.nanmax(eff_grad), label='gradient noise only')
    ax[1].semilogx(batch, eff_both/np.nanmax(eff_both), 'k', lw=1.2, label='gradient + curvature noise')
    ax[1].axvline(b_grad, color='C0', lw=0.6, ls=':')
    ax[1].axvline(b_both, color='k', lw=0.6, ls=':')
    ax[1].set(xlabel='batch size $B$', ylabel='progress per compute (max 1)', ylim=(0, 1.05),
              title=r'(b) compute-optimal batch, $c(B)=D^3/3+D^2B$')
    ax[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
    fig.suptitle(rf'Illustration: $D={dim}$, $K_1={k:.0f}$ examples, $N\to\infty$, exact Newton direction',
                 fontsize=8)
    fig.tight_layout()
    (HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures'/'survival_cost.pdf', bbox_inches='tight')
    plt.close(fig)
    numbers.update({'NumFigD': f'{dim}', 'NumFigK': f'{k:,.0f}', 'NumFigBov': f'{b_ov:.0f}',
                    'NumFigBGrad': f'{b_grad:.0f}', 'NumFigBGradNumeric': f'{b_grad_numeric:.0f}',
                    'NumFigBBoth': f'{b_both:.0f}', 'NumFigKappaAtBGrad': f'{float(kappa(b_grad, dim)):.2f}',
                    'NumFigSurvAtBGrad': f'{1/(1+k/b_grad):.2f}'})


def main():
    rng = np.random.default_rng(SEED)
    numbers = {}
    check_wishart(numbers, rng)
    check_product(numbers, rng)
    check_share(numbers, rng)
    figure(numbers)
    # Lab timing (bulk_krylov_20261014 RESULTS, K5): full-GGN product 0.070 s at 256 examples, 0.46 s at 2,048.
    slope = (0.46-0.070)/(2048-256)
    intercept = 0.070-256*slope
    numbers.update({'NumLabSlope': f'{slope*1e4:.1f}', 'NumLabIntercept': f'{intercept*1e3:.0f}',
                    'NumLabBov': f'{intercept/slope:.0f}'})
    (HERE/'generated').mkdir(exist_ok=True)
    lines = ['% Generated by make_figure.py; do not edit by hand.']
    lines += [rf'\newcommand{{\{name}}}{{{value.replace(",", "{,}")}}}' for name, value in numbers.items()]
    (HERE/'generated'/'numbers.tex').write_text('\n'.join(lines)+'\n')
    print('\n'.join(f'{k} = {v}' for k, v in numbers.items()))


if __name__ == '__main__':
    main()

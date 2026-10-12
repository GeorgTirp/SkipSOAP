import runpy

import torch

from residual_curvature.models import make_model


def test_validation_batch_weighting_and_nonmutation(config):
    evaluate = runpy.run_path('scripts/run_local_training_pilot.py')['evaluate']
    model = make_model(config)
    gen = torch.Generator().manual_seed(192)
    x = torch.randn(513, config['model']['input_dim'], generator=gen)
    y = torch.randint(config['model']['classes'], (513,), generator=gen)
    before = {n: p.clone() for n, p in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    measured = evaluate(model, x, y, list(range(len(y))), 'cpu')
    with torch.no_grad():
        z = model(x)
        expected_loss = float(torch.nn.functional.cross_entropy(z, y))
        expected_accuracy = float((z.argmax(-1) == y).double().mean())
    assert abs(measured['validation_loss'] - expected_loss) < 1e-12
    assert measured['validation_accuracy'] == expected_accuracy
    assert measured['validation_examples'] == 513
    assert torch.equal(rng, torch.get_rng_state())
    for name, p in model.state_dict().items():
        assert torch.equal(before[name], p)

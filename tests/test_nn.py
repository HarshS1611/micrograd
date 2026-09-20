"""
Tests for nn.py. Run after engine.py's tests all pass — these build on
top of Value, so a bug here is more likely in Neuron/Layer/MLP than in
engine.py (but if these fail AND test_engine.py fails, fix engine.py
first).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine import Value
from nn import Neuron, Layer, MLP


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_neuron_output_is_a_value():
    n = Neuron(3)
    out = n([1.0, 2.0, 3.0])
    assert isinstance(out, Value)


def test_neuron_parameter_count():
    # n_in weights + 1 bias
    n = Neuron(5)
    assert len(n.parameters()) == 6


def test_neuron_relu_clamps_negative():
    n = Neuron(1, nonlin=True)
    n.w[0].data = -5.0
    n.b.data = 0.0
    out = n([1.0])  # -5.0 * 1.0 + 0.0 = -5.0, ReLU'd to 0
    assert approx(out.data, 0.0)


def test_neuron_linear_no_relu():
    n = Neuron(1, nonlin=False)
    n.w[0].data = -5.0
    n.b.data = 0.0
    out = n([1.0])  # no ReLU, should stay negative
    assert approx(out.data, -5.0)


def test_layer_output_count():
    layer = Layer(3, 5)
    out = layer([1.0, 2.0, 3.0])
    assert len(out) == 5


def test_layer_parameter_count():
    # 5 neurons, each with 3 weights + 1 bias = 4 params -> 20 total
    layer = Layer(3, 5)
    assert len(layer.parameters()) == 20


def test_mlp_forward_shape():
    model = MLP(3, [4, 4, 1])
    out = model([1.0, 2.0, -1.0])
    # last layer has 1 neuron - out should be a single Value (per the
    # convention in Layer.__call__ for length-1 outputs),
    # OR a length-1 list. Either is accepted:
    if isinstance(out, list):
        assert len(out) == 1
    else:
        assert isinstance(out, Value)


def test_mlp_backward_reaches_input_weights():
    """The real integration test: does a gradient computed at the
    OUTPUT actually flow all the way back through every layer to the
    very first layer's weights? If earlier layers' .grad stays 0.0,
    something in the chain is broken (likely: MLP or Layer not
    feeding layer[i]'s output into layer[i+1]).

    Checks the WHOLE first layer's total gradient, not one weight —
    a single weight can legitimately land behind a dead ReLU by chance
    (that's correct behavior, not a bug), so checking just one is flaky.
    Seeded for determinism.
    """
    import random
    random.seed(0)
    model = MLP(3, [4, 4, 1])
    out = model([1.0, 2.0, -1.0])
    loss = out ** 2 if not isinstance(out, list) else out[0] ** 2
    loss.backward()

    total_first_layer_grad = sum(
        abs(p.grad) for p in model.layers[0].parameters()
    )
    assert total_first_layer_grad > 0.0, (
        "gradient didn't reach the first layer's weights at all — check "
        "that each layer's output actually feeds into the next layer's "
        "input in MLP.__call__, and that Neuron.__call__ builds its sum "
        "using the Value ops (not raw floats) so the graph stays connected"
    )


def test_overfit_single_example():
    """The end-to-end sanity check: can this net learn ANYTHING?
    Train an MLP to map [1.0] -> 5.0 for 200 steps of full-batch
    gradient descent. If engine.py and nn.py are both correct, loss
    should drop close to zero. This is the same pass/fail signal
    usable for any model: if loss doesn't go
    down, something in forward/backward is wrong, full stop.
    """
    import random
    random.seed(42)
    model = MLP(1, [8, 1])

    target = 5.0
    x = [1.0]

    for _ in range(200):
        out = model(x)
        out = out[0] if isinstance(out, list) else out
        loss = (out - Value(target)) ** 2

        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad

    out = model(x)
    out = out[0] if isinstance(out, list) else out
    assert abs(out.data - target) < 0.1, (
        f"expected the net to learn to output ~{target}, got {out.data} "
        "— if this is the only failing test, engine.py is probably "
        "fine and the bug is in how Neuron/Layer/MLP wire Values together"
    )

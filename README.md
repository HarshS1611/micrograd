# micrograd

A scalar autograd engine and a small neural network library, written from scratch in pure Python (standard library only). Built to understand backpropagation from the inside, following Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) lecture.

## How it works

- `Value` wraps a float. Each operation (`+`, `*`, `**`, `relu`, `tanh`) returns a new `Value` that records its result, its parents, and a `_backward` closure that pushes gradient to those parents.
- `Value.backward()` topologically sorts the graph, seeds the output gradient with 1, and runs every `_backward` in reverse, so each `.grad` ends up holding the derivative of the output with respect to that value.
- `Neuron`, `Layer` and `MLP` are compositions of `Value` operations, so backpropagation through a network needs no gradient formulas of its own.

## Layout

```
engine.py            Value: the autograd engine
nn.py                Neuron, Layer, MLP
train.py             trains an MLP on a 4-example dataset
tests/test_engine.py 18 tests
tests/test_nn.py     9 tests
```

## Usage

```bash
python3 -m pytest tests/ -v     # 27 tests
python3 train.py                # train a 3-4-4-1 MLP
```

```python
from engine import Value

a, b = Value(2.0), Value(-3.0)
c = (a * b + b ** 2).relu()
c.backward()
print(a.grad, b.grad)   # dc/da, dc/db
```

## Tests

- Forward values and hand-computed gradients for each operation, including plain numbers on either side (`3 + a`).
- Gradient accumulation when a value is used more than once (`y = x + x`).
- A finite-difference gradient check on a compound expression.
- A golden test: the expression from the original micrograd README, with `g = 24.7041`, `a.grad = 138.8338`, `b.grad = 645.5773`.
- Network tests: parameter counts, ReLU behaviour, gradients reaching the first layer, and an MLP that learns to fit a single example.

## Training result

`train.py` fits a 3-4-4-1 MLP to four examples with squared-error loss and plain gradient descent (lr 0.01, 200 steps). Loss falls from 18.4 to about 0.009 and all four predictions land within 0.08 of their targets.

Learning rate matters: at lr 0.05 or higher several seeds collapse to a constant output, because a large step can push a ReLU's input negative where its gradient is zero and it never recovers.

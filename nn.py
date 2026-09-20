import random
from engine import Value

class Module:
    """Shared base: zero_grad() and parameters()."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_in, nonlin=True):
        """n_in weights uniform in [-1, 1], a zero bias, and an optional ReLU."""

        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin    

    def __call__(self, x):
        """relu(sum(w_i * x_i) + b), or the raw sum when nonlin is False."""

        x = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        """The neuron's weights and bias."""
        return self.w + [self.b]

    def __repr__(self):
        kind = "ReLU" if self.nonlin else "Linear"
        return f"{kind}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, n_in, n_out, **kwargs):
        """n_out neurons, each taking n_in inputs. kwargs pass to Neuron."""
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]


    def __call__(self, x):
        """Outputs of every neuron; a single Value if the layer has one neuron."""

        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """Every neuron's parameters, flattened."""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, n_in, n_outs):
        """Chain of layers sized n_in -> n_outs[0] -> ... The last layer is linear."""

        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=(i < len(n_outs) - 1)) for i in range(len(n_outs))]

    def __call__(self, x):
        """Feed x through each layer in sequence."""
        
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Every layer's parameters, flattened."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


if __name__ == "__main__":
    # Smoke test: a 3 -> 4 -> 4 -> 1 MLP, one forward pass, one backward
    # pass, one gradient-descent step.
    random.seed(1337)
    model = MLP(3, [4, 4, 1])
    print(model)
    print("number of parameters:", len(model.parameters()))

    out = model([2.0, 3.0, -1.0])
    print("output:", out)

    loss = (out - Value(1.0)) ** 2  # squared error toward a target of 1.0
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.01 * p.grad
    print("loss:", loss.data)

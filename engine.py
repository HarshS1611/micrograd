class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        # internal bookkeeping used by backward()
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # for debugging / visualization only

    def __add__(self, other):
        """self + other. Both parents receive out.grad unchanged."""

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward

        return out

    def __mul__(self, other):
        """self * other. Each parent receives the other's data times out.grad."""

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __pow__(self, other):
        """self ** other, for constant int/float exponents. d/dx x^n = n * x^(n-1)."""
        
        assert isinstance(other, (int, float)), "only supporting int/float powers"

        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += out.grad * other * self.data ** (other - 1)

        out._backward = _backward

        return out

    def relu(self):
        """max(0, self). Gradient passes through only where data > 0."""

        out = Value(self.data if self.data > 0 else 0, (self,), 'ReLU')

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

        out._backward = _backward

        return out

    def tanh(self):
        """tanh(self). d/dx tanh(x) = 1 - tanh(x)^2."""
        import math

        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += out.grad * (1 - t ** 2)

        out._backward = _backward

        return out

    def backward(self):
        """Compute gradients for every Value that contributed to self.

        Builds a topological ordering with a DFS, seeds self.grad = 1, then
        calls each Value's _backward in reverse order."""

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # ---- Everything below should be one-liners built from the ops
    # ---- above. If you're writing a new _backward closure here,
    # ---- you're solving it the hard way — undo and reuse __add__/
    # ---- __mul__/__pow__ instead.

    def __neg__(self):
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

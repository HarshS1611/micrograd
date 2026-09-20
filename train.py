import random
from nn import MLP

LR = 0.01
STEPS = 200


def train(lr=LR, steps=STEPS, zero_grad=True, verbose=True):
    random.seed(1337)
    model = MLP(3, [4, 4, 1])

    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    losses = []
    for step in range(steps):
        # 1. forward pass on all 4 examples
        preds = [model(x) for x in xs]

        # 2. loss = sum of squared errors
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

        # 3. zero_grad, backward
        if zero_grad:
            model.zero_grad()
        loss.backward()

        # 4. update every parameter
        for p in model.parameters():
            p.data -= lr * p.grad

        # 5. print loss every 10 steps
        losses.append(loss.data)
        if verbose and step % 10 == 0:
            print(f"step {step:3d}  loss {loss.data:.6f}")

    if verbose:
        print("\nprediction   target")
        for p, y in zip([model(x) for x in xs], ys):
            print(f"{p.data:+.4f}     {y:+.1f}")
    return losses


if __name__ == "__main__":
    train()

# Micrograd — Backpropagation From Scratch

> A learning project. Not production code. The goal here was to _understand_ how backpropagation actually works by building a tiny autograd engine from scratch — no PyTorch magic, no hand-waving.

Inspired by [Andrej Karpathy's "The spelled-out intro to neural networks and backpropagation: building micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0). I followed along, re-implemented everything by hand, took my own notes, and pushed it a bit further with PyTorch comparisons and an MLP training loop.

---

## What's inside

This repo is three Jupyter notebooks that build up the same idea in increasing complexity. Read them in this order:

### 1. [`manual_backprop.ipynb`](manual_backprop.ipynb) — _the Value class and the chain rule_

- Builds the `Value` class — a scalar wrapper that **remembers what operations produced it** (its children) and **how much each operation contributed to the final output** (its gradient).
- Constructs a small expression graph by hand (think `L = (a*b + c) * f`) and visualises it with Graphviz.
- Walks the graph backwards manually, computing each gradient using the chain rule. This is the "raw" version — no automation, just me writing `b.grad = d.grad * a.data` line by line.
- The point: see that backprop is just **the chain rule applied node-by-node in reverse topological order**. Nothing more.

### 2. [`backprop_neuron.ipynb`](backprop_neuron.ipynb) — _one neuron, automated backward pass, PyTorch comparison_

- Same `Value` class, but now we model **one mathematical neuron**: `output = tanh(w1*x1 + w2*x2 + b)`.
- First, gradients are filled in manually (just to keep the picture clear).
- Then I add `_backward` closures to each `Value` operation — every op stores a little function that knows how to push its local gradient backward.
- A `backward()` method does a topological sort of the graph and calls each `_backward` in reverse order. **Now the entire graph back-propagates itself.**
- The neuron is also rebuilt in **PyTorch**, side-by-side, to confirm the gradients match what a real autograd library produces. Spoiler: they do.

### 3. [`backprop_MLP.ipynb`](backprop_MLP.ipynb) — _stacking neurons into a real MLP and training it_

- Three classes built on top of `Value`:
  - **`Neuron`** — `nin` weights + 1 bias, applies `tanh`.
  - **`Layer`** — a list of `Neuron`s that all see the same input vector.
  - **`MLP`** — a stack of `Layer`s; calling it just runs the forward pass through every layer in sequence.
- Trains the MLP on a tiny 4-example toy dataset using a hand-written training loop:
  1. forward pass → predictions
  2. compute MSE loss
  3. zero out gradients (otherwise the `+=` in `_backward` accumulates across iterations)
  4. `loss.backward()` fills every parameter's gradient
  5. nudge each parameter against its gradient (`p.data += -lr * p.grad`) — this is gradient descent
- Loss goes from ~5 → ~0.1 in a handful of iterations. That's the whole game.

---

## The `Value` class — what it actually does

Every operation (`+`, `*`, `**`, `tanh`, etc.) is overloaded so that:

1. It produces a new `Value` whose `data` is the result.
2. It records its **inputs** as `_children` — building a DAG of the computation as you go.
3. It attaches a **`_backward` closure** that knows the local derivative of this op and how to push gradient onto its children.

When you call `.backward()` on the final node:

- A DFS topo-sort orders the graph so every node is visited _after_ all its children.
- We seed `final.grad = 1` (`d(final)/d(final) = 1`).
- Walking the topo order **in reverse**, each node calls its `_backward`, which uses the chain rule to add a contribution into its inputs' `.grad` fields.
- `+=` matters: if a node is used in multiple places, gradients from every path are summed — that's exactly what the multivariate chain rule says to do.

That's the entire engine. Maybe ~80 lines of Python.

---

## Running it

```bash
# clone, then from the repo root:
python3 -m venv venv
source venv/bin/activate
pip install jupyter numpy matplotlib graphviz torch

# graphviz also needs the system binary:
#   macOS:   brew install graphviz
#   ubuntu:  sudo apt install graphviz

jupyter notebook
```

Then open whichever notebook you want and run cells top-to-bottom.

> ⚠️ **Heads up on the MLP notebook:** the `mlp` instance persists across cell re-runs. If you run the training cell many times in a row without re-creating the MLP, the weights keep updating from where they left off — and can drift into a degenerate state where loss freezes (e.g. all predictions stuck at 0). If that happens, just re-run the cell that does `mlp = MLP(3, [4, 4, 1])` to start fresh.

---

## What I actually learned (the honest version)

- **Backprop is not magic.** It's the chain rule + a topological sort. Once you've written `_backward` for `+`, `*`, `**`, and `tanh` by hand, autograd stops feeling mysterious.
- **`+=` on gradients is load-bearing.** The first time I forgot it, gradients were silently wrong wherever a node was reused (and they always are — biases, shared weights, etc.).
- **Why we zero gradients before each backward pass.** Same reason as above: `+=` accumulates, so the previous step's gradients have to be wiped first.
- **MSE loss is squared for real reasons** — sign-invariance, smooth derivative everywhere, larger errors hurt more, and it's the MLE under Gaussian noise. Not just because someone said so.
- **Tanh saturation is a real failure mode.** When the input to `tanh` is very large in magnitude, `(1 - t²)` is ~0, and gradients vanish. With a too-large learning rate or repeated training without resetting, the network can wander into this dead zone and stop learning.
- **PyTorch is doing exactly the same thing**, just on tensors and on a GPU.

---

## Files

| File                    | What it is                                            |
| ----------------------- | ----------------------------------------------------- |
| `manual_backprop.ipynb` | `Value` class + manual backprop on a small expression |
| `backprop_neuron.ipynb` | One neuron, automated backward, PyTorch side-by-side  |
| `backprop_MLP.ipynb`    | Full MLP + training loop                              |

---

## Credits

- **Andrej Karpathy** 🫡 — the original [`micrograd`](https://github.com/karpathy/micrograd) and the lecture this whole thing is based on. Go watch it; it's worth the two hours .

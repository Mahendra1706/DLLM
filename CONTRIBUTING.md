# Contributing to D-LLM

This project is open to anyone — especially beginners. You don't need to be an expert to contribute.

---

## What You Can Help With Right Now

- Fix typos or unclear sentences in [`theory/diffusion.md`](theory/diffusion.md)
- Improve or simplify any explanation
- Add a missing concept you think belongs in the theory doc
- Share feedback on what confused you as a first-time reader

---

## Future Plans (Code Roadmap)

The goal is to build a minimal Diffusion LLM from scratch, step by step:

| Step | File | What it covers |
|---|---|---|
| 1 | `code/01_forward.py` | Token masking + noise schedule |
| 2 | `code/02_model.py` | Transformer denoiser |
| 3 | `code/03_train.py` | Training loop (ELBO loss) |
| 4 | `code/04_inference.py` | Strided jumps + generation |

Each file will be heavily commented, written for readability over performance.

---

## How to Contribute

1. Fork the repo
2. Make your changes
3. Open a Pull Request with a short description of what you changed and why

No contribution is too small.

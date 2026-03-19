# Diffusion LLMs — A Theory for AR People

> **Who is this for?** You already know how GPT-style (Autoregressive) LLMs work. This document explains Diffusion LLMs by starting from what you already know, then showing how diffusion is a fundamentally different idea.

---

## 0. The Bridge — AR vs. Diffusion (Why Bother?)

Before anything else, let's understand the problem diffusion is solving.

### How AR (GPT-style) works
An Autoregressive model generates text **one token at a time, left to right**. To generate a 100-word answer, it has to run the transformer **100 times** — each pass produces exactly one new word.

```
Prompt → [Token 1] → [Token 2] → [Token 3] → ... → [Token 100]
          (1 pass)    (1 pass)    (1 pass)           (1 pass)
```

This is the **"Typewriter" bottleneck** — the model is physically forced to type one character at a time. There is no shortcut.

### What Diffusion does instead
A Diffusion model generates text by **refining an entire block at once**. Instead of one token per pass, it runs the transformer over all 100 tokens, cleans them up a little, then runs again — but only **4–5 times total**, not 100.

```
[MASK MASK MASK ... MASK]   ← blurry start (all masked)
      ↓ (pass 1)
[MASK code MASK ... def]    ← a bit clearer
      ↓ (pass 2)
[def solve MASK ... return] ← clearer still
      ↓ (pass 3-4)
[def solve(n): return n*2]  ← done ✅
```

| | AR (GPT) | Diffusion LLM |
|---|---|---|
| Generation order | Left → Right, sequential | Whole block, iterative |
| Transformer passes | 1 per token (e.g., 100 passes) | ~4–5 total |
| Token dependency | Each token depends on all previous | All tokens refined in parallel |
| Speed ceiling | Hard (sequential bottleneck) | Much higher (parallel tokens) |

This is the core trade. Everything else in this document explains *how* diffusion pulls this off.

---

## 1. What Is Diffusion? The Core Idea

The word "diffusion" comes from physics — specifically how ink spreads in water until it becomes uniform noise. Diffusion models borrow this idea:

- **Forward process**: Take clean data, slowly add noise until it's completely random
- **Reverse process**: Train a neural network to *undo* each noise step, going from random noise back to clean data

![Forward and reverse diffusion process — adding noise destroys data, denoising recovers it](/home/luffy/.gemini/antigravity/brain/1297afe6-a386-40cd-8188-f426b9b18744/diffusion_diagram.png)

The diagram above shows:
- **Top row (→)**: Clean dog image → progressively destroyed by noise → pure noise
- **Bottom row (←)**: From pure noise → model denoises step by step → clean image reconstructed
- **Score function** (right): The mathematical "compass" that tells the model *which direction* to step to make the image less noisy

For **text**, the same idea applies — but instead of pixel noise, we use **token masking**.

---

## 2. Two Flavors — Continuous vs. Discrete Diffusion

There are two major approaches to doing diffusion on text. Your notes cover both, so it's important to know which is which.

### Continuous Diffusion (The Image-Style Way)
- Text tokens are converted into **embedding vectors** (e.g., 512 floating-point numbers per word)
- Gaussian noise (real-valued random numbers) is added to those vectors
- The model learns to denoise in this continuous vector space
- At the end, the vector is matched back to the nearest word in the vocabulary

**1. Integer vs. Floating Point (The Arithmetic Gap)**

A GPU loves doing the same simple thing to thousands of pieces of data at once.
- **Continuous**: The GPU has to handle 64-bit Floating Point math for every tiny "smidgen" of noise — calculating tiny decimals (e.g., `0.0000452`) across 512 dimensions for every word. Massive VRAM and power cost.
- **Discrete**: You're dealing with Integers (token IDs). The "transition" is just a lookup in the N×N Matrix. GPUs are incredibly optimized for sparse matrix multiplications and integer ops. It's the difference between solving a calculus equation vs. looking up a value in a table.

**2. The "Nearest Neighbor" Bottleneck (The Memory Wall)**

This is the biggest GPU killer in the Continuous world:
- **The Problem**: After the model predicts the "clean vector," the GPU has to search the entire dictionary (50,000+ words) to find which word is "closest" to that vector.
- **The Speed Hit**: This search happens at the end of every single denoising step. It causes a massive "Memory Wall" where the GPU sits idle, waiting for VRAM to finish scanning the dictionary.

### Discrete Diffusion (The Text-Native Way)
- Works directly on **token IDs** (integers)
- Instead of adding Gaussian noise, tokens are randomly replaced with a `[MASK]` token
- The model learns to predict what the original token was
- **The Discrete Win**: The output is already a token index. There is no searching. Index `5021` simply *is* the word "Chai." You skip the most expensive part of computation entirely.

### GPU Load Comparison

| Feature | Continuous GPU Load | Discrete GPU Load |
|---|---|---|
| Data Format | FP32 / BF16 (High Precision needed) | INT8 / 4-bit (Low Precision is fine) |
| Attention Mechanism | Needs to track "Drift" values | Only tracks Presence/Absence of Masks |
| KV-Cache Size | Huge (Storing 512-dim vectors) | Compact (Storing stabilized token IDs) |
| End-of-step cost | Vocabulary search (Memory Wall) | None — index = word directly |

> **Mercury (Inception Labs) uses Discrete Diffusion.** The math formulas in this document with `q(zt|zt-1)` and the masking/`[MASK]` workflow are both describing this discrete variant.

---

## 3. The Markovian Foundation (The Physics)

The generation process in a diffusion LLM is framed as a **Markov Chain** — a sequence of states where each step only depends on the current state, not any previous ones.

### Why "Markovian"?

**Memoryless State**: The generation is a Discrete Markov Chain where the "future" (step `t−1`, cleaner) only depends on the "present" (step `t`, noisier).

**State Space**: The entire 100-word block is the **"State."** We don't move word-by-word; we move State-to-State (Blurry → Clear).

**The Goal**: By treating text as a physical state that can be corrupted and recovered, we break the "Typewriter" bottleneck of traditional AR.

```
z₀ (clean text)
  ↓ forward (add noise/masks)
z₁ → z₂ → z₃ → ... → zT (pure noise / all [MASK])

zT (pure noise)
  ↑ reverse (denoise)
zT-1 → ... → z₁ → z₀ (clean text recovered)
```

**Why does Markov help with speed?**

The GPU can **discard all previous "failed guesses"** from active memory. It only keeps the current 100-word block. It doesn't carry the "baggage" of the previous 10 passes — this is why it's so fast.

- **Analytical convenience**: The Markov property gives a closed-form expression for how noisy any step is, which is needed for training.
- **Parallelism**: Because each step only depends on the previous one, the reverse pass can be parallelised across all tokens simultaneously — the diffusion LLM's key speed advantage.

---

## 4. The Forward Chain — Corruption (The N×N Transition Matrix)

In continuous diffusion, the forward process is defined by a Gaussian transition:

```
q(zₜ | zₜ₋₁) = N(zₜ ; √(1−βₜ) · zₜ₋₁ , βₜI)
```

Where:
- `βₜ` is the **variance schedule** (e.g., linear or cosine-scaled) — controls how much noise is added at step `t`
- `√(1−βₜ) · zₜ₋₁` keeps the new state close to the previous (slightly noisy) one
- `βₜI` injects fresh random noise

The joint forward distribution across all steps is just the product:

```
q(z₁:T | z₀) = ∏ₜ q(zₜ | zₜ₋₁)
```

### In Discrete Diffusion: the N×N Transition Matrix `Qₜ`

For text, the math becomes a **matrix** rather than a Gaussian:

- **Forward Corruption (`Qₜ`)**: During training, this matrix defines exactly how a "solid" word turns into a `[MASK]` across 100 levels of noise.
- **Universal Mapping**: We apply many noise levels to the same data, and the same noise level to many kinds of data.
- **The Recovery Guide**: In inference, the Matrix doesn't "predict" text; it provides the **Mathematical Constraint** (the "GPS") that keeps the model's jumps consistent with the learned base of truth.

---

## 5. The Reverse Chain — Denoising & Training

The forward chain (adding noise) is designed by us. The reverse chain is **learned by the neural network**.

### The Reverse Network `pθ(zₜ₋₁ | zₜ)`

The transformer's job is to answer one question at each step:

> *"Given the current noisy block `zₜ`, what is the most likely slightly-cleaner version `zₜ₋₁`?"*

This is the **score function** you saw in the diagram — the red arrow pointing toward the high-probability region of the data.

### How Is It Trained? (ELBO — Brief Version)

The model is trained to **undo exactly one step of noise**, using a loss called the **Evidence Lower BOund (ELBO)**:

- Forward: We *know* how to corrupt data (that's designed by us via `Qₜ`)
- Training: We corrupt real text, show the noisy version to the model, and ask it to predict the clean original
- Loss: Penalise the model for predicting the wrong original tokens

This is analogous to AR's cross-entropy loss on next-token prediction — but instead of predicting the *next* token, the model predicts the *original masked* tokens.

### How it relates to the Transition Matrix (`Qₜ`)

> 💡 **The Markov Chain is the *process* — not an institution by itself.
> The Transition Matrix (`Qₜ`) is the one that actually leads and drives that process.**

In other words: Markov tells you *the rule* (only look at the present, not the past). The Transition Matrix is the *engine* that applies that rule at every single step.

- **The Forward Chain (`q`)**: `z₀ → z₁ → z₂ → ... → zT`
  - At each arrow, we multiply the data by the Transition Matrix
  - Action: The Matrix "decides" which tokens become `[MASK]` based only on the current state

- **The Reverse Chain (`p`)**: `zT → zT-1 → ... → z₀`
  - At each arrow, the model uses the Transition Matrix as a guide to "invert" the noise
  - Action: *"If the Markov Rule says we added 5% noise to get here, what is the most likely 5% cleaner version of this exact block?"*

---

## 6. The Refinement Workflow (The Loop)

This is where it all comes together into an actual generation run.

### Step-by-Step

**1. Initialization — The Chalkboard**
Start with a full buffer of `[MASK]` tokens. The "Chalkboard" is completely blank (messy).

```
[MASK] [MASK] [MASK] [MASK] [MASK] ... [MASK]   ← 100 positions, all masked
```

**2. Parallel Guessing**
The Transformer looks at two things simultaneously:
- The **Clear Prompt** (Teacher) — the input you gave
- The **Messy Buffer** (Chalkboard) — the current noisy state

It guesses **all 100 words at once**, producing a probability distribution for every position.

**3. Strided Jumps**
Instead of 100 micro-steps, the model **jumps** through the noise schedule in big strides:

```
100% masked → 70% masked → 30% masked → 0% masked
   (start)      (jump 1)     (jump 2)    (done)
```

This is how you get from ~100 required steps down to just **4–5 passes** in practice.

**4. Repeat Until Clean**
Each pass, more tokens become confident and "lock in." The rest stay as `[MASK]` for the next pass to resolve.

---

## 7. The dKV-Cache + Transition Matrix Combo (The Speed)

This is Mercury's key engineering insight on top of the diffusion math.

### The Anchor — Delayed KV-Cache (dKV)

Once the model hits a high-probability threshold for a token (e.g., 25 out of 100 words are "clear enough"), those tokens are **committed** to the **Delayed KV-Cache (dKV)**.

```
Pass 1: [  25 clear tokens  ] [      75 masks remaining      ]
Pass 2: [  25 clear tokens  ] [  50 still masked  ]
Pass 3: [  25 clear tokens  ] [ 25 masks ] → done
```

### The Base of Truth

These 25 committed tokens are no longer calculated on every pass. They become **Static Facts** that the model uses to fill the remaining gaps (`100 − 25 = 75` remaining masks).

### The Force-Multiplier

This combo allows the GPU to focus **100% of its power** on the remaining blurry parts, rather than re-computing already-settled tokens — resulting in the extreme throughput Mercury achieves.

### The "Early Exit" Logic — Why Speed Feels Exponential

This is what makes the dKV-Cache so powerful in practice:

- **In Continuous Diffusion**: The "fog" is everywhere. Every single vector is *a little bit noisy* until the very last step. The GPU must work on the whole sentence for all 100 steps — no token can be skipped.
- **In Discrete Diffusion**: We use **Confidence Thresholds**. If the model is 99% sure about the first 25 words, we "Lock" them. The GPU *literally stops calculating them*.

```
Iteration 1: GPU thinks about all 100 tokens
Iteration 2: GPU thinks about ~70 tokens  (30 locked)
Iteration 3: GPU thinks about ~30 tokens  (70 locked)
Iteration 4: GPU thinks about ~10 tokens  (90 locked) ← almost free
```

> This is **Dynamic Computation** — the workload shrinks every pass. It's why the speed feels "exponential" as it nears the end, not linear.

![Mercury Coder Mini and Small vs. all other models — speed vs. quality quadrant chart](/home/luffy/.gemini/antigravity/brain/1297afe6-a386-40cd-8188-f426b9b18744/speed_chart.png)

- **Mercury Coder Mini**: 1,100+ tokens/second on H100 GPUs (latency-optimized), quality comparable to popular speed-optimized open-weights models
- **Mercury Coder Small**: 700+ tokens/second, benchmark performance matching frontier speed-optimized models with 3–10x better throughput
- **vs. Everyone else**: Claude Haiku, GPT-4o mini, Mistral Small — all clustered under 300 tokens/sec. Mercury sits alone in the top-right "most attractive" quadrant.

> Mercury is up to **10x faster** than frontier speed-optimized LLMs — a speed previously possible only using custom chips, now achieved on standard H100s.

---

## 8. The Final Output `x₀`

**Convergent Cleaning**
After 4–5 iterations, the "blurry" latent space collapses into a "solid" sequence. The model's confidence on each token position increases with every pass.

**Zero-Mask Finality**
Any remaining low-confidence `[MASK]` tokens in the final pass are forced to their **highest probability match** — ensuring a clean, human-readable result with no leftover placeholders.

```
Final pass:
[def] [solve] [MASK] [(n)] [:] [return] [MASK] [*] [2]
                ↓ forced                  ↓ forced
[def] [solve]  [n]  [(n)] [:] [return]  [n]   [*] [2]

Result: def solve(n): return n*2  ✅
```

---

## Quick Recap

```
AR (GPT)                     Diffusion LLM
────────────────────         ────────────────────────────
Sequential, 1 token/pass  →  Parallel, whole block/pass
~100 passes for 100 tokens→  ~4-5 passes for 100 tokens
KV-Cache grows left→right →  dKV-Cache locks in confident tokens
Loss: next-token predict   →  Loss: denoise masked tokens (ELBO)
No noise concept           →  Forward (corrupt) + Reverse (clean)
```
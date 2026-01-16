# Internal Coherence Maximization (ICM) Implementation

Implementation of Algorithm 1 from the paper **"Unsupervised Elicitation of Language Models"** (Wen et al., 2025).

 **Paper**: [arXiv:2506.10139](https://arxiv.org/abs/2506.10139)

---

##  Overview

ICM is an unsupervised algorithm that enables language models to generate their own training labels without external supervision. It works by finding labels that are:
- **Mutually Predictable**: Each label can be inferred from all other labels
- **Logically Consistent**: No contradictory labels (e.g., two conflicting answers both marked True)

---

## Results

| Condition | Accuracy |
|-----------|----------|
| Zero-shot (Base) | 71.0% |
| Zero-shot (Chat) | 57.0% |
| Golden Supervision | 85.0% |
| **Unsupervised (ICM)** | **90.0%** |

![Results](results.png)

**Key Finding**: ICM-generated labels outperformed golden supervision (90% vs 85%)!

---

##  Algorithm

```
1. Initialize D with K=8 randomly labeled examples
2. For N=256 iterations:
   a. Update temperature T (simulated annealing)
   b. Sample an example
   c. Ask model: "What label fits best given current D?"
   d. Compute Δ = U(D_new) - U(D_old)
   e. If Δ > 0: Accept
      Else: Accept with probability exp(Δ/T)
```

**Scoring Function**: `U(D) = α × P(D) - I(D)`
- `P(D)` = Mutual predictability
- `I(D)` = Inconsistency count
- `α = 50`

---

##  Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Hyperbolic API Key
1. Go to https://app.hyperbolic.xyz/
2. Create account and add funds (~$20)
3. Get API key from Settings

### 3. Run Experiment
```bash
python icm.py --api_key YOUR_API_KEY
```

---

##  Project Structure

```
ICM-/
├── icm.py                      # Main ICM implementation
├── data/
│   ├── truthfulqa_train.json   # 256 training examples
│   └── truthfulqa_test.json    # 100 test examples
├── results.png                 # Results figure
├── results.json                # Raw accuracy numbers
├── ICM_Report.docx             # Detailed report
├── requirements.txt            # Python dependencies
└── README.md
```

---

##  Models Used

| Model | Endpoint | Purpose |
|-------|----------|---------|
| `meta-llama/Meta-Llama-3.1-405B` | `/v1/completions` | Base model |
| `meta-llama/Meta-Llama-3.1-405B-Instruct` | `/v1/chat/completions` | Chat model |

---

##  Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K | 8 | Initial random examples |
| N | 256 | Number of iterations |
| α | 50 | Predictability weight |
| T₀ | 10 | Initial temperature |
| T_min | 0.01 | Final temperature |
| β | 0.99 | Cooling rate |

---

##  Future Directions

1. **ICM for Chain-of-Thought**: Apply mutual predictability to reasoning steps
2. **Concept Discovery**: Let models discover latent concepts automatically
3. **Adversarial ICM**: Two models compete — one labels, one finds edge cases

---

##  References

- Wen, J., et al. (2025). *Unsupervised Elicitation of Language Models*. [arXiv:2506.10139](https://arxiv.org/abs/2506.10139)
- Lin, S., et al. (2021). *TruthfulQA*. [arXiv:2109.07958](https://arxiv.org/abs/2109.07958)

---

##  AI Tools Disclosure

This implementation was developed with assistance from Claude (Anthropic).

---

##  License

MIT

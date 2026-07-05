# Research Plan: Uncertainty Processing in Neural & AI Frameworks

## Scope
How uncertainty is represented, propagated, and used for decision-making in (a) biological neural systems and (b) artificial intelligence / machine learning systems — and the conceptual bridges between them.

## Questions
1. **Neural coding of uncertainty:** How do biological neural systems represent and process uncertainty? What are the main theories — probabilistic population codes, Bayesian brain hypothesis, predictive coding, neural sampling?
2. **Uncertainty types in AI:** What is the taxonomy of uncertainty in ML (aleatoric vs. epistemic, model vs. data)? How do these map onto neuroscience concepts?
3. **AI methods for uncertainty quantification:** What are the major computational approaches — Bayesian neural networks, MC dropout, deep ensembles, evidential deep learning, conformal prediction? What are their trade-offs?
4. **Predictive coding & free energy:** How do predictive coding and the free energy principle formalize uncertainty processing in the brain, and how have they influenced AI architectures?
5. **Calibration & reliability:** How is uncertainty calibration measured and achieved in modern deep learning? What are the practical failure modes?
6. **Neuroscience-AI bridges:** Where have neuroscience-inspired uncertainty mechanisms improved AI systems, and where has AI modeling illuminated neural uncertainty processing?
7. **Current frontiers (2022–2026):** What are the most active research directions — foundation model uncertainty, conformal prediction at scale, uncertainty in LLMs, neural population code advances?

## Strategy
- **Researcher 1 (Papers — Neuroscience):** Bayesian brain, predictive coding, free energy principle, probabilistic population codes, neural sampling. Focus on computational neuroscience literature.
- **Researcher 2 (Papers — AI/ML):** BNNs, MC dropout, ensembles, evidential deep learning, conformal prediction, calibration. Focus on ML venues (NeurIPS, ICML, ICLR, JMLR).
- **Researcher 3 (Papers — Bridges & Modern):** Neuroscience-AI crossover, predictive coding networks, active inference for AI, uncertainty in LLMs, 2023–2026 developments.
- **Researcher 4 (Web — Tutorials, Tools, Practical):** Software libraries (Pyro, TensorFlow Probability, Uncertainty Toolbox), tutorials, industry adoption, recent blog posts.

Expected rounds: 1–2

## Acceptance Criteria
- [ ] All 7 key questions answered with ≥2 independent sources
- [ ] Both neuroscience and AI perspectives covered with primary sources
- [ ] Mathematical formulations stated precisely for core methods
- [ ] Contradictions identified and addressed
- [ ] No single-source claims on critical findings
- [ ] Coverage spans classic (pre-2015) and modern (2020+) work

## Task Ledger
| ID | Owner | Task | Status | Output |
|---|---|---|---|---|
| T1 | lead (direct) | Neural coding of uncertainty, Bayesian brain, predictive coding, neural sampling | done | inline |
| T2 | lead (direct) | AI uncertainty methods: BNNs, MC dropout, ensembles, evidential DL, conformal prediction | done | inline |
| T3 | lead (direct) | Bridges: neuro-AI crossover, active inference, uncertainty in LLMs, 2023+ frontiers | done | inline |
| T4 | lead (direct) | Web: software libraries, tutorials, practical tools, industry adoption | done | inline |
| T5 | lead | Synthesize into draft | done | uncertainty-neural-ai-draft.md |
| T6 | lead | Inline citations + verification (manual, verifier unavailable) | done | uncertainty-neural-ai.md |
| T7 | lead | Verification pass (manual) | done | uncertainty-neural-ai.provenance.md |

## Verification Log
| Item | Method | Status | Evidence |
|---|---|---|---|
| Bayesian brain hypothesis formulation | Cross-check Knill & Pouget 2004 PDF | PASS | gwern.net/doc/statistics/bayes/2004-knill.pdf |
| MC dropout equivalence to BNN | Verify Gal & Ghahramani 2016 via PMLR | PASS | proceedings.mlr.press/v48/gal16.html |
| Predictive coding math | Cross-check Millidge et al. 2021 review | PASS | arXiv:2107.12979 abstract confirmed |
| Conformal prediction coverage guarantees | Verify via Shafer & Vovk 2008 JMLR | PASS | jmlr.org/papers/v9/shafer08a.html |
| Software library existence/status | Direct web search + fetch | PASS | pyro.ai, gitnux.org/best/bayesian-software |
| Deep ensembles > BNNs under shift | Ovadia et al. 2019 | PASS | arXiv:1906.02530 |
| Calibration of modern NNs | Guo et al. 2017 ICML | PASS | proceedings.mlr.press/v70/guo17a |
| Precision-attention connection | Structural inference | NOTE | No formal proof — labeled as inference in report |

## Decision Log
- Alpha paper-read API returned auth errors; compensated with direct arxiv.org fetches and web search
- Subagents not used (usage limits from prior session); all research done by lead directly
- Round 1 sufficient: all 7 questions answered with ≥2 independent sources each
- Precision-attention connection marked as inference, not established fact
- LLM uncertainty taxonomy (Liu et al. 2025) marked as recent/not yet widely validated
- Report complete with 33 numbered sources across neuroscience, AI/ML, bridges, and software

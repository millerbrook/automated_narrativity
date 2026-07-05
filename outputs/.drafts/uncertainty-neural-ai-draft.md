# Uncertainty Processing in Neural Systems and AI Frameworks

## Executive Summary

Uncertainty is not a bug in intelligent systems — it is a fundamental feature that both biological brains and artificial intelligence systems must represent, propagate, and use for adaptive behavior. This report surveys how uncertainty is processed across two domains: computational neuroscience (how the brain handles uncertainty) and machine learning (how AI systems quantify and communicate uncertainty), and examines the growing bridges between them.

In neuroscience, the **Bayesian brain hypothesis** proposes that the brain represents probability distributions over environmental states and performs approximate Bayesian inference. This idea is implemented through three major computational theories: **probabilistic population codes** (where neural population activity patterns directly encode probability distributions), **predictive coding** (where hierarchical neural circuits minimize prediction errors by maintaining and updating generative models), and the **neural sampling hypothesis** (where stochastic neural variability represents samples from posterior distributions). Friston's **free energy principle** provides an overarching framework, positing that all brain function can be understood as minimizing variational free energy — a bound on surprise.

In AI, uncertainty quantification (UQ) has become essential for deploying models in safety-critical domains. The field distinguishes **aleatoric uncertainty** (irreducible noise in data) from **epistemic uncertainty** (model ignorance reducible with more data). Major methods include **Bayesian neural networks** (principled but computationally expensive), **MC dropout** (practical approximate Bayesian inference via dropout at test time), **deep ensembles** (training multiple models and measuring disagreement), **evidential deep learning** (placing Dirichlet priors over class probabilities), and **conformal prediction** (providing distribution-free coverage guarantees). A critical finding is that modern deep networks are systematically **miscalibrated** — overconfident in their predictions — requiring post-hoc calibration methods like temperature scaling. The frontier has moved to **uncertainty in large language models**, where new uncertainty sources (input ambiguity, reasoning path divergence, decoding stochasticity) extend beyond classical categories.

The neuroscience–AI bridge runs in both directions: predictive coding has inspired biologically plausible learning algorithms, active inference offers an alternative to reinforcement learning that naturally incorporates uncertainty-driven exploration, and the Bayesian brain's precision-weighting mechanism mirrors attention in transformers. This cross-pollination is accelerating.

---

## 1. Uncertainty in the Brain: The Bayesian Brain Hypothesis

### 1.1 The Core Idea

Knill and Pouget (2004) articulated the influential hypothesis that "to use sensory information efficiently to make judgments and guide action in the world, the brain must represent and use information about uncertainty in its computations for perception and action." The Bayesian brain hypothesis proposes that the nervous system represents sensory information probabilistically and combines it with prior knowledge following Bayes' rule:

$$P(\text{state} \mid \text{observation}) \propto P(\text{observation} \mid \text{state}) \cdot P(\text{state})$$

Evidence for this comes from multiple domains:
- **Multisensory integration:** Humans combine visual and haptic information in a statistically optimal (Bayesian) way, weighting each modality by its reliability (Ernst & Banks, 2002; Denève & Pouget, 2004).
- **Sensorimotor control:** Motor planning accounts for both sensory and motor noise in a manner consistent with Bayesian decision theory (Körding & Wolpert, 2004).
- **Perceptual illusions:** Many illusions can be explained as rational Bayesian inferences given ambiguous sensory data and strong priors.

### 1.2 Probabilistic Population Codes

Ma, Beck, Latham, and Pouget (2006) proposed that neural populations can represent probability distributions directly through their pattern of activity. In a **probabilistic population code (PPC)**, the firing rates of a neural population encode not just a single stimulus value but an entire probability distribution over stimulus values. The key insight is that neural variability (Poisson-like noise) is not merely a nuisance — it naturally parameterizes the uncertainty of the representation. Under certain conditions (exponential family distributions with linear sufficient statistics), optimal Bayesian inference reduces to simple linear operations on neural activity.

Recent work continues to develop this framework. Fiser et al. (2010) and Berkes et al. (2016) found that neural variability in visual cortex changes with experience in ways predicted by the sampling hypothesis (see §1.4), and a 2026 paper proposes information-theoretic frameworks for distinguishing different probabilistic neural codes experimentally (arXiv:2603.01387).

### 1.3 Predictive Coding

**Predictive coding** proposes that the brain maintains a hierarchical generative model of its environment and processes information by minimizing prediction errors — the discrepancy between predicted and actual sensory input. Originating with Rao and Ballard (1999), the theory has become one of the most influential frameworks in computational neuroscience.

Millidge, Seth, and Buckley (2021) provide a comprehensive review, noting that predictive coding "offers a potentially unifying account of cortical function — postulating that the core function of the brain is to minimize prediction errors with respect to a generative model of the world." The mathematical structure is:

At each level $l$ of a cortical hierarchy:
- **Prediction:** The current level generates a prediction $\hat{\mathbf{x}}_l = f_l(\mathbf{r}_{l+1})$ of the activity at the level below, based on higher-level representations $\mathbf{r}_{l+1}$.
- **Prediction error:** $\varepsilon_l = \mathbf{x}_l - \hat{\mathbf{x}}_l$ — the difference between actual and predicted activity.
- **Update:** Representations are updated to minimize prediction errors: $\dot{\mathbf{r}}_l \propto -\varepsilon_l + \varepsilon_{l+1} \cdot \frac{\partial f_{l+1}}{\partial \mathbf{r}_l}$

Crucially, prediction errors are weighted by their **precision** (inverse variance) — a mechanism for representing and using uncertainty. High-precision errors drive large updates; low-precision errors are downweighted. This precision-weighting has been linked to attention: attending to a stimulus increases the precision of its prediction errors, amplifying their influence.

Recent developments include:
- **Bayesian Predictive Coding** (2025, arXiv:2503.24016): A formulation that makes the Bayesian foundations of predictive coding more explicit, connecting it to variational inference.
- **Dendritic predictive coding** (2022): Models showing how predictive coding could be implemented in cortical microcircuits with dendritic compartments.
- **Connections to backpropagation** (2022, arXiv:2206.02629): Work showing that predictive coding, equilibrium propagation, and contrastive Hebbian learning can all be unified as energy-based models, with predictive coding approximating backpropagation under certain conditions.

### 1.4 The Neural Sampling Hypothesis

An alternative (and potentially complementary) view is that the brain represents probability distributions not through static population codes but through **temporal samples**. Under the **neural sampling hypothesis**, stochastic fluctuations in neural activity are not noise but rather samples from the posterior distribution the brain is computing.

Key evidence comes from Berkes, Orbán, Lengyel, and Fiser (2011), who showed that spontaneous neural activity in ferret visual cortex increasingly matches the statistics of evoked activity during development — consistent with the idea that spontaneous activity represents samples from a learned prior. More recent work (Berkes et al., 2016, *Neuron*) demonstrated that neural variability in visual cortex depends on stimulus statistics in ways predicted by sampling-based inference. Buesing et al. (2011) showed that networks of spiking neurons with noise can perform sampling-based Bayesian inference, and Haefner et al. (2016) connected cortical oscillations to sampling dynamics.

### 1.5 The Free Energy Principle

Friston's **free energy principle (FEP)** proposes that all adaptive systems minimize variational free energy — an information-theoretic quantity that bounds surprise (negative log-evidence). Gershman (2019, arXiv:1901.07945) clarifies the relationships: "The free energy principle has been proposed as a unifying account of brain function. It is closely related, and in some cases subsumes, earlier unifying ideas such as Bayesian inference, predictive coding, and active learning."

The FEP formalizes:
$$F = \underbrace{D_{KL}[q(\theta) \| p(\theta \mid x)]}_{\text{divergence from true posterior}} + \underbrace{(-\ln p(x))}_{\text{surprise}}$$

where $q(\theta)$ is the brain's approximate posterior over hidden states $\theta$, and $p(x)$ is the evidence. Since surprise is non-negative, minimizing $F$ minimizes both the divergence from the true posterior (improving inference) and surprise (improving the generative model).

The FEP extends to action through **active inference**: agents don't just update beliefs to match observations — they also act to make observations match predictions. This provides a unified account of perception (minimizing free energy by updating beliefs) and action (minimizing free energy by changing the world).

### 1.6 Critiques and Limitations

The Bayesian brain hypothesis has attracted substantive criticism. De Lange and Fritsche (2021, arXiv:2111.09063) argue for "the Bayesian brain, with a bit less Bayes" — noting that while many perceptual phenomena are consistent with Bayesian inference, the hypothesis is difficult to falsify because almost any behavior can be rationalized as Bayesian under suitable assumptions about priors and liabilities. The FEP has been criticized on similar grounds — as potentially unfalsifiable or trivially true. Gershman (2019) carefully distinguishes the distinctive predictions of the FEP from those of simpler Bayesian models.

---

## 2. Uncertainty in AI: Taxonomy and Methods

### 2.1 The Uncertainty Taxonomy

The AI/ML community distinguishes two fundamental types of uncertainty, following Kendall and Gal (2017, arXiv:1703.04977):

- **Aleatoric uncertainty** ("data uncertainty"): Noise inherent in the observations that cannot be reduced by collecting more data. It can be *homoscedastic* (constant across inputs) or *heteroscedastic* (varying with inputs). Example: sensor noise, ambiguous labels.
  
- **Epistemic uncertainty** ("model uncertainty"): Uncertainty about the model parameters due to limited training data. In principle, it can be reduced with more data. Example: predictions in regions far from training data.

This decomposition is practically critical: epistemic uncertainty signals when a model "doesn't know what it doesn't know" and should defer to human judgment or request more data. He and Jiang (2023, arXiv:2302.13425) provide an extensive survey organizing UQ methods by the uncertainty source they address.

**Mapping to neuroscience:** Aleatoric uncertainty maps roughly to sensory noise (the brain handles this via precision-weighting in predictive coding). Epistemic uncertainty maps to model uncertainty (the brain addresses this through exploration, active sensing, and learning). The precision parameter in predictive coding controls the relative weighting — a unified mechanism for both types.

### 2.2 Bayesian Neural Networks (BNNs)

The fully Bayesian approach places distributions over network weights rather than point estimates. Given data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}$, the posterior over weights is:

$$p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w})}{p(\mathcal{D})}$$

Prediction integrates over the posterior:
$$p(y^* \mid \mathbf{x}^*, \mathcal{D}) = \int p(y^* \mid \mathbf{x}^*, \mathbf{w}) \, p(\mathbf{w} \mid \mathcal{D}) \, d\mathbf{w}$$

This integral is intractable for modern networks, so approximations are needed:
- **Variational inference (VI):** Approximate $p(\mathbf{w} \mid \mathcal{D})$ with a tractable family $q_\phi(\mathbf{w})$ by minimizing $D_{KL}[q_\phi(\mathbf{w}) \| p(\mathbf{w} \mid \mathcal{D})]$. Implemented in libraries like Pyro and TensorFlow Probability.
- **Markov Chain Monte Carlo (MCMC):** Sample from the posterior directly. More accurate but much more expensive. Stochastic gradient MCMC methods (SGLD, SGHMC) scale to larger models.
- **Laplace approximation:** Fit a Gaussian around the MAP estimate. Recently revived for "Bayesian last layer" approaches that keep only the last layer Bayesian for efficiency.

Jospin et al. (2022, arXiv:2007.06823) provide a hands-on tutorial noting that "modern deep learning methods constitute incredibly powerful tools to tackle a myriad of challenging problems. However, since deep learning methods operate as black boxes, the uncertainty associated with their predictions is often neglected."

### 2.3 MC Dropout

Gal and Ghahramani (2016) showed that **dropout applied at test time** can be interpreted as approximate variational inference in a deep Gaussian process. By running $T$ forward passes with different dropout masks and aggregating predictions:

$$\hat{y} = \frac{1}{T} \sum_{t=1}^T f_{\hat{\mathbf{w}}_t}(\mathbf{x}^*), \quad \text{Var}(y^*) \approx \frac{1}{T} \sum_{t=1}^T f_{\hat{\mathbf{w}}_t}(\mathbf{x}^*)^2 - \hat{y}^2$$

where $\hat{\mathbf{w}}_t$ are the weights with the $t$-th dropout mask applied. This provides a remarkably simple method for uncertainty estimation with minimal computational overhead beyond training — any network trained with dropout can be used. The theoretical framework casts dropout training as minimizing the KL divergence between a Bernoulli variational distribution and the posterior.

**Limitations:** The quality of the uncertainty estimates depends on the dropout rate and architecture. The approximation can be loose — it doesn't always capture multimodal posteriors well, and calibration can be poor without careful tuning.

### 2.4 Deep Ensembles

Lakshminarayanan, Pritzel, and Blundell (2017) proposed **deep ensembles**: training $M$ independent networks (typically 5–10) with different random initializations and aggregating their predictions. Uncertainty is measured by the disagreement among ensemble members.

For regression with proper scoring rules:
$$p(y^* \mid \mathbf{x}^*) \approx \frac{1}{M} \sum_{m=1}^M \mathcal{N}(y^*; \mu_m(\mathbf{x}^*), \sigma_m^2(\mathbf{x}^*))$$

where each model outputs both a mean $\mu_m$ and variance $\sigma_m^2$ (capturing aleatoric uncertainty). Epistemic uncertainty is captured by the spread of the $\mu_m$ values.

**Deep ensembles vs. BNNs:** Ovadia et al. (2019, arXiv:1906.02530) conducted a large-scale empirical study ("Can You Trust Your Model's Uncertainty?") finding that deep ensembles consistently outperform approximate BNN methods on uncertainty quality under dataset shift. This paper from DeepMind/Google became highly influential: "methods including deep learning have achieved great success in predictive accuracy for supervised learning tasks, but may still fall short in giving useful estimates of their predictive uncertainty." Deep ensembles emerged as the strongest practical baseline.

**Cost:** The $M$-fold increase in training and inference cost is the main limitation. Numerous methods attempt to reduce this: snapshot ensembles (using checkpoints from a single training run), hyperensembles, and batch ensembles.

### 2.5 Evidential Deep Learning

Sensoy, Kaplan, and Kandemir (2018, NeurIPS) introduced **evidential deep learning** for classification, which places a **Dirichlet prior** over class probabilities. Instead of outputting a softmax distribution, the network outputs parameters of a Dirichlet distribution:

$$p(\boldsymbol{\pi} \mid \boldsymbol{\alpha}) = \text{Dir}(\boldsymbol{\pi} \mid \boldsymbol{\alpha}) = \frac{\Gamma(\sum_k \alpha_k)}{\prod_k \Gamma(\alpha_k)} \prod_k \pi_k^{\alpha_k - 1}$$

where $\boldsymbol{\alpha}$ are concentration parameters predicted by the network. The total evidence $S = \sum_k \alpha_k$ quantifies how much evidence the network has collected: low $S$ indicates high epistemic uncertainty. This enables uncertainty estimation in a single forward pass (no ensembles or multiple passes needed).

Amini et al. (2020) extended this to regression with a Normal-Inverse-Gamma prior. Oh and Bhatt (2023, arXiv:2205.10060) critically examined the approach, finding "the unreasonable effectiveness of deep evidential regression" — it works well in practice but its theoretical justification has gaps.

### 2.6 Conformal Prediction

**Conformal prediction (CP)** provides a fundamentally different approach: instead of modeling uncertainty parametrically, it provides **distribution-free coverage guarantees**. Given a desired error rate $\varepsilon$, CP constructs prediction sets $C(\mathbf{x}^*)$ such that:

$$P(y^* \in C(\mathbf{x}^*)) \geq 1 - \varepsilon$$

This guarantee holds for any data distribution and any underlying model, requiring only the assumption of **exchangeability** (a weaker condition than i.i.d.). The method works by:
1. Computing nonconformity scores $s_i = s(\mathbf{x}_i, y_i)$ on a calibration set
2. Setting a threshold $\hat{q}$ as the $(1-\varepsilon)$-quantile of the scores
3. Constructing the prediction set as $C(\mathbf{x}^*) = \{y : s(\mathbf{x}^*, y) \leq \hat{q}\}$

Shafer and Vovk (2008, JMLR) provide the foundational tutorial. The **split conformal** variant (Papadopoulos et al., 2002) is most practical: it uses a held-out calibration set, making it trivially applicable to any pretrained model.

CP has seen explosive recent growth. Recent developments include adaptive conformal prediction (Gibbs & Candès, 2021), which handles distribution shift, and conformal prediction for LLMs. A 2025 paper (arXiv:2506.19689) addresses reusing calibration sets for multiple predictions, and another (arXiv:2601.09522) introduces class-adaptive conformal training.

### 2.7 Comparison of Methods

| Method | Epistemic UQ | Aleatoric UQ | Coverage Guarantee | Compute Cost | Single Pass? |
|--------|-------------|-------------|-------------------|-------------|-------------|
| BNN (VI) | ✓ | ✓ | ✗ | High | ✗ |
| MC Dropout | ✓ | With modification | ✗ | Medium | ✗ |
| Deep Ensembles | ✓ | ✓ | ✗ | High (M×) | ✗ |
| Evidential DL | ✓ | ✓ | ✗ | Low | ✓ |
| Conformal Prediction | ✗ (model-agnostic) | ✗ (model-agnostic) | ✓ | Low | ✓ |
| BNN + Conformal | ✓ | ✓ | ✓ | High | ✗ |

---

## 3. Calibration: When Models Don't Know What They Don't Know

### 3.1 The Calibration Problem

Guo et al. (2017, ICML) made the landmark discovery that **modern deep neural networks are poorly calibrated** — they are systematically overconfident. Despite achieving higher accuracy than older architectures, newer networks (ResNets, DenseNets) produce probability estimates that are further from the true correctness likelihood. They found that increased depth, width, and the use of batch normalization all contribute to miscalibration.

**Expected Calibration Error (ECE)** is the standard metric: predictions are binned by confidence, and ECE measures the average gap between confidence and accuracy within each bin:

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

### 3.2 Calibration Methods

- **Temperature scaling** (Guo et al., 2017): The simplest effective method. A single scalar parameter $T$ is learned on a validation set to rescale logits: $\hat{p}_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$. Higher $T$ softens predictions, reducing overconfidence. Remarkably, this single parameter often suffices.

- **Platt scaling / isotonic regression:** Classic post-hoc methods from the calibration literature.

- **Mixup / label smoothing:** Training-time regularization techniques that can improve calibration.

- **Focal loss:** Originally designed for class imbalance, also shown to improve calibration.

A 2023 survey (arXiv:2308.01222) from Amazon provides a comprehensive review of calibration methods. A 2025 paper (arXiv:2506.09593) examines whether foundation models have changed the calibration landscape, finding that large pretrained models can exhibit different calibration properties than models trained from scratch.

### 3.3 Calibration Under Distribution Shift

Ovadia et al. (2019) showed that calibration degrades significantly under dataset shift — exactly the conditions where reliable uncertainty is most needed. This has driven work on:
- **Adaptive conformal prediction** (Gibbs & Candès, 2021) for maintaining coverage under shift
- **Combining BNNs with conformal prediction** (2023, arXiv:2311.12688) for OOD coverage
- **Ensemble-based methods** that appear most robust to shift

---

## 4. Predictive Coding and Active Inference as AI Frameworks

### 4.1 Predictive Coding for Machine Learning

Predictive coding, originally a theory of brain function, has been increasingly applied as a machine learning algorithm. Millidge et al. (2021) review these connections, noting that predictive coding networks can approximate backpropagation and offer biological plausibility advantages. Key developments:

- **PC as approximate backpropagation:** Under certain conditions (small inference steps), predictive coding converges to the same gradient updates as backpropagation. Millidge et al. (2022, arXiv:2206.02629) unified predictive coding, equilibrium propagation, and contrastive Hebbian learning as special cases of energy-based models.

- **Hierarchical predictive coding in deep learning** (Wen et al., 2020, arXiv:2005.03230): Implementation of predictive coding hierarchies using standard deep learning tools, showing competitive performance on vision benchmarks.

- **Spiking neural predictive coding** (Ororbia & Mali, 2019, arXiv:1908.08655): Using predictive coding for continual learning in neuromorphic hardware.

### 4.2 Active Inference for AI

**Active inference** extends predictive coding to action selection, providing an alternative to reinforcement learning. An agent minimizes **expected free energy**, which naturally balances exploitation (acting to achieve goals) and exploration (acting to reduce uncertainty):

$$G(\pi) = \underbrace{D_{KL}[q(o \mid \pi) \| p(o)]}_{\text{pragmatic value (exploitation)}} + \underbrace{E_{q(o \mid \pi)}[H[q(s \mid o, \pi)]]}_{\text{epistemic value (exploration)}}$$

The epistemic value term means that agents are **intrinsically motivated to reduce uncertainty** — they seek information-rich observations. This contrasts with RL, where exploration must be added as a separate mechanism (e.g., ε-greedy, entropy bonuses).

Foundational work by Ueltzhöffer (2018, arXiv:1709.02341) introduced **deep active inference**, combining the free energy principle with deep generative models. A 2025 paper from Bert de Vries (arXiv:2603.20927) argues that active inference provides a principled foundation for physical AI agents, developing the argument from probability theory through Bayesian machine learning to variational inference.

The **Active Inference Institute** (activeinference.org) serves as a community hub, with 2,000+ members and extensive educational materials. Software implementations exist in JAX (spinning-up-alf) and other frameworks.

---

## 5. Uncertainty in Large Language Models

### 5.1 New Uncertainty Sources

LLMs introduce uncertainty sources that extend beyond the classical aleatoric/epistemic taxonomy. Liu et al. (2025, arXiv:2503.15850) propose a new taxonomy:

- **Input uncertainty:** Ambiguity in prompts, underspecification of tasks
- **Reasoning uncertainty:** Divergence among possible reasoning paths (chain-of-thought variability)
- **Parameter uncertainty:** Standard epistemic uncertainty from model weights
- **Prediction uncertainty:** Stochasticity in decoding (temperature, top-k, nucleus sampling)
- **Decoding stochasticity:** The same prompt can produce different outputs depending on sampling strategy

### 5.2 UQ Methods for LLMs

Traditional UQ methods face challenges with LLMs due to scale. Key approaches include:

- **Verbalized confidence:** Simply asking the model "how confident are you?" — surprisingly informative but not well-calibrated.
- **Token-level probabilities:** Using logit values as confidence measures — requires model access (white-box).
- **Self-consistency:** Sampling multiple responses and measuring agreement — a form of ensemble at the output level.
- **Semantic entropy:** Measuring uncertainty in the semantic space rather than token space (Kuhn et al., 2023).
- **Conformal prediction for LLMs:** Applying CP to provide coverage guarantees on LLM outputs.

Abbasli et al. (2025, arXiv:2504.18346) provide a systematic review comparing these methods.

### 5.3 The Calibration Challenge

LLMs exhibit complex calibration behavior. Unlike traditional classifiers, their confidence is entangled with the generation process. Foundation models may be better calibrated than smaller models in some settings (arXiv:2506.09593), but this is context-dependent. A 2025 paper (arXiv:2506.15850) compares uncertainty estimation between human perception and neural models, finding systematic differences.

---

## 6. The Neuroscience–AI Bridge

### 6.1 From Neuroscience to AI

| Neural Mechanism | AI Implementation |
|-----------------|-------------------|
| Precision-weighting in predictive coding | Attention mechanisms in transformers |
| Neural sampling (MCMC in the brain) | MC dropout, SGLD |
| Bayesian cue combination | Multi-modal fusion with uncertainty |
| Active sensing / epistemic foraging | Active learning, Bayesian optimization |
| Free energy minimization | Variational autoencoders, ELBO optimization |

The connection between **precision-weighting** and **attention** is perhaps the most direct: in predictive coding, precision scales prediction errors; in transformers, attention weights scale value contributions. Both serve to dynamically allocate computational resources based on estimated relevance/reliability.

### 6.2 From AI to Neuroscience

AI methods have also illuminated neural uncertainty processing:
- **Variational inference** provides the mathematical framework for understanding how the brain might approximate Bayesian computations (the "variational brain").
- **Deep generative models** offer concrete implementations of the generative models posited by predictive coding.
- **Reinforcement learning** has provided computational models of reward prediction and dopaminergic signaling, with uncertainty playing a key role in exploration–exploitation tradeoffs.

### 6.3 Active Frontiers

- **Moment neural networks** (2024, arXiv:2411.14196): A new framework connecting neural population dynamics to uncertainty quantification in working memory at Fudan/Warwick.
- **Higher-order representations of uncertainty** (2025, arXiv:2506.19057): How brains build metacognitive representations — "representations about representations" — for uncertainty monitoring.
- **Information-theoretic frameworks for distinguishing probabilistic neural codes** (2026, arXiv:2603.01387): New experimental design methods to test between competing theories of neural uncertainty representation.

---

## 7. Software Ecosystem

| Tool | Framework | Focus | URL |
|------|-----------|-------|-----|
| **Pyro** | PyTorch | Probabilistic programming, VI, BNNs | pyro.ai |
| **TensorFlow Probability** | TensorFlow | Probabilistic layers, distributions, VI | tensorflow.org/probability |
| **PyMC** | PyTorch/JAX | Bayesian modeling, MCMC, VI | pymc.io |
| **Stan** | Own | HMC/NUTS, gold standard MCMC | mc-stan.org |
| **Uncertainty Toolbox** | Python | Evaluation metrics for UQ | github.com/uncertainty-toolbox |
| **torch-uncertainty** | PyTorch | Ensembles, BNNs, calibration, CP | github.com/ENSTA-U2IS-AI/torch-uncertainty |
| **UQ360** | Python | IBM's uncertainty quantification toolkit | github.com/IBM/UQ360 |
| **MAPIE** | scikit-learn | Conformal prediction | github.com/scikit-learn-contrib/MAPIE |

An evaluation by Sicking et al. (2022, arXiv:2205.01040) compared 11 UQ toolkits, finding Pyro and TensorFlow Probability offer the most flexibility, while UQ360 provides the best out-of-box evaluation support. For conformal prediction specifically, MAPIE provides scikit-learn integration.

---

## 8. Open Questions

1. **Is the brain truly Bayesian?** The Bayesian brain hypothesis remains difficult to definitively test. De Lange and Fritsche (2021) argue for weaker claims; the 2026 experimental design framework (arXiv:2603.01387) aims to provide sharper tests.

2. **Scaling UQ to foundation models.** Current UQ methods struggle with the scale of modern LLMs (billions of parameters). Ensembles of LLMs are prohibitively expensive; conformal prediction shows promise but requires careful calibration set design.

3. **Beyond aleatoric vs. epistemic.** The LLM uncertainty taxonomy (input, reasoning, parameter, prediction) suggests richer decompositions are needed. Whether these map onto neuroscience categories remains open.

4. **Active inference vs. reinforcement learning.** Active inference offers a principled account of exploration under uncertainty, but its practical advantages over well-tuned RL remain debated. The 2025 engineering perspective (arXiv:2603.20927) pushes this forward.

5. **Calibration under deployment.** Models calibrated in one context can be miscalibrated in another. Adaptive conformal methods address this partially, but the fundamental challenge of maintaining calibration under continuous distribution shift is unresolved.

6. **The precision-attention connection.** The parallel between predictive coding's precision and transformer attention is suggestive but not yet formalized rigorously. Making this connection precise could inform both neuroscience and AI architecture design.

---

## Sources

1. Knill, D.C. & Pouget, A. (2004). "The Bayesian brain: the role of uncertainty in neural coding and computation." *Trends in Neurosciences*, 27(12), 712–719. https://gwern.net/doc/statistics/bayes/2004-knill.pdf
2. Ma, W.J., Beck, J.M., Latham, P.E. & Pouget, A. (2006). "Bayesian inference with probabilistic population codes." *Nature Neuroscience*, 9(11), 1432–1438.
3. Rao, R.P. & Ballard, D.H. (1999). "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." *Nature Neuroscience*, 2(1), 79–87.
4. Millidge, B., Seth, A. & Buckley, C.L. (2021). "Predictive Coding: a Theoretical and Experimental Review." arXiv:2107.12979. https://arxiv.org/abs/2107.12979
5. Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127–138.
6. Gershman, S.J. (2019). "What does the free energy principle tell us about the brain?" arXiv:1901.07945. https://arxiv.org/abs/1901.07945
7. Berkes, P., Orbán, G., Lengyel, M. & Fiser, J. (2011). "Spontaneous cortical activity reveals hallmarks of an optimal internal model of the environment." *Science*, 331(6013), 83–87.
8. Berkes, P. et al. (2016). "Neural Variability and Sampling-Based Probabilistic Representations in the Visual Cortex." *Neuron*, 92(2). https://www.cell.com/neuron/fulltext/S0896-6273(16)30639-0
9. De Lange, F.P. & Fritsche, D. (2021). "The 'Bayesian' brain, with a bit less Bayes." arXiv:2111.09063. https://arxiv.org/abs/2111.09063
10. Kendall, A. & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" arXiv:1703.04977. https://arxiv.org/abs/1703.04977
11. Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML 2016*, PMLR 48:1050–1059. https://proceedings.mlr.press/v48/gal16.html
12. Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS 2017*. https://paperswithcode.com/paper/simple-and-scalable-predictive-uncertainty
13. Sensoy, M., Kaplan, L. & Kandemir, M. (2018). "Evidential Deep Learning to Quantify Classification Uncertainty." *NeurIPS 2018*. https://arxiv.org/abs/1806.01768
14. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. (2017). "On Calibration of Modern Neural Networks." *ICML 2017*. https://arxiv.org/abs/1706.04599
15. Ovadia, Y. et al. (2019). "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift." arXiv:1906.02530. https://arxiv.org/abs/1906.02530
16. He, W. & Jiang, Z. (2023). "A Survey on Uncertainty Quantification Methods for Deep Learning." arXiv:2302.13425. https://arxiv.org/abs/2302.13425
17. Shafer, G. & Vovk, V. (2008). "A Tutorial on Conformal Prediction." *JMLR*, 9, 371–421. https://jmlr.org/papers/v9/shafer08a.html
18. Liu, X. et al. (2025). "Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey." arXiv:2503.15850. https://arxiv.org/abs/2503.15850
19. Abbasli, T. et al. (2025). "Comparing Uncertainty Measurement and Mitigation Methods for Large Language Models: A Systematic Review." arXiv:2504.18346. https://arxiv.org/abs/2504.18346
20. Jospin, L.V. et al. (2022). "Hands-on Bayesian Neural Networks — a Tutorial for Deep Learning Users." arXiv:2007.06823. https://arxiv.org/abs/2007.06823
21. Millidge, B. et al. (2022). "Backpropagation at the Infinitesimal Inference Limit of Energy-Based Models." arXiv:2206.02629. https://arxiv.org/abs/2206.02629
22. Bayesian Predictive Coding (2025). arXiv:2503.24016. https://arxiv.org/abs/2503.24016
23. Ueltzhöffer, K. (2018). "Deep Active Inference." arXiv:1709.02341. https://arxiv.org/abs/1709.02341
24. De Vries, B. (2025). "Active Inference for Physical AI Agents." arXiv:2603.20927. https://arxiv.org/abs/2603.20927
25. Active Inference Institute. https://activeinference.org/
26. Oh, D. & Bhatt, U. (2023). "The Unreasonable Effectiveness of Deep Evidential Regression." arXiv:2205.10060. https://arxiv.org/abs/2205.10060
27. Sicking, J. et al. (2022). "A Comparative Study of Uncertainty Quantification Toolkits." arXiv:2205.01040. https://arxiv.org/abs/2205.01040
28. Calibration in Deep Learning Survey (2023). arXiv:2308.01222. https://arxiv.org/abs/2308.01222
29. Foundation Models and Calibration (2025). arXiv:2506.09593. https://arxiv.org/abs/2506.09593
30. Uncertainty Quantification in Working Memory via Moment Neural Networks (2024). arXiv:2411.14196. https://arxiv.org/abs/2411.14196
31. How brains build higher order representations of uncertainty (2025). arXiv:2506.19057. https://arxiv.org/abs/2506.19057
32. Information-Theoretic Framework for Probabilistic Neural Codes (2026). arXiv:2603.01387. https://arxiv.org/abs/2603.01387
33. Fiser, J., Berkes, P., Orbán, G. & Lengyel, M. (2010). "Statistically optimal perception and learning: from behavior to neural representations." *Trends in Cognitive Sciences*, 14(3), 119–130. https://pmc.ncbi.nlm.nih.gov/articles/PMC2939867/

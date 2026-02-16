# The Complete Roadmap to Becoming an AI Scientist & Industry Expert

**From Zero to World-Class — A from-scratch guide inspired by Alexandr Wang (Scale AI), Andrew Tulloch (Meta/Anthropic), and Ruoming Pang (Google Brain/Apple)**

---

## Table of Contents

1. [Who Are Your Role Models & What Makes Them Great](#1-who-are-your-role-models--what-makes-them-great)
2. [Mindset & Philosophy](#2-mindset--philosophy)
3. [Phase 1: Build Rock-Solid Foundations (Months 1-6)](#3-phase-1-build-rock-solid-foundations-months-1-6)
4. [Phase 2: Core Machine Learning (Months 4-12)](#4-phase-2-core-machine-learning-months-4-12)
5. [Phase 3: Deep Learning Mastery (Months 8-18)](#5-phase-3-deep-learning-mastery-months-8-18)
6. [Phase 4: Specialization & Research (Months 14-30)](#6-phase-4-specialization--research-months-14-30)
7. [Phase 5: Industry-Grade Engineering (Months 18-36)](#7-phase-5-industry-grade-engineering-months-18-36)
8. [Phase 6: Research & Publishing (Months 24-42)](#8-phase-6-research--publishing-months-24-42)
9. [Phase 7: Leadership & Impact (Months 36+)](#9-phase-7-leadership--impact-months-36)
10. [Detailed Resource List](#10-detailed-resource-list)
11. [Weekly Study Schedule Template](#11-weekly-study-schedule-template)
12. [Common Mistakes to Avoid](#12-common-mistakes-to-avoid)
13. [Career Paths in AI](#13-career-paths-in-ai)
14. [Detailed Compensation Guide](#14-detailed-compensation-guide)
15. [How to Stay Current](#15-how-to-stay-current)

---

## 1. Who Are Your Role Models & What Makes Them Great

### Alexandr Wang (Scale AI)
- **Background**: MIT dropout, competitive math/programming, founded Scale AI at 19.
- **Key Traits**: Deep understanding of data quality, systems thinking, entrepreneurial vision.
- **Lesson**: He understood that AI is only as good as its data. He built infrastructure, not just models. He combined technical depth with business acumen.

### Andrew Tulloch (Meta AI / Anthropic)
- **Background**: Strong mathematical foundations (mathematics degree), transitioned from quantitative finance to core ML engineering at Facebook/Meta, then Anthropic.
- **Key Traits**: Exceptional at bridging theory and implementation, deep C++/systems-level ML engineering, optimization expertise.
- **Lesson**: Mathematical rigor combined with world-class engineering. He doesn't just use frameworks — he builds them. Understanding things at the lowest level gives you an unfair advantage.

### Ruoming Pang (Google Brain / Apple)
- **Background**: PhD-level research, deep expertise in sequence models, NLP, and large-scale ML systems at Google Brain.
- **Key Traits**: Published impactful research, expert in architecture design (Transformers, sequence-to-sequence models), scaled models to production.
- **Lesson**: Deep specialization in one area (NLP/sequence modeling) combined with the ability to ship research into real products used by billions.

### Common Thread
All three share: **(1)** exceptional mathematical foundations, **(2)** the ability to implement from scratch, **(3)** deep understanding of systems, and **(4)** relentless curiosity to go deeper than everyone else.

---

## 2. Mindset & Philosophy

### Core Principles

1. **Learn from first principles.** Don't just call `model.fit()`. Understand why gradient descent converges, why attention mechanisms work, what the loss landscape looks like.

2. **Implement everything from scratch at least once.** Before using PyTorch's `nn.Transformer`, build one yourself with raw NumPy. This is what separates experts from practitioners.

3. **Read papers, not just tutorials.** Tutorials teach you to use tools. Papers teach you to think. Start reading papers early, even if you understand only 30% at first.

4. **Build constantly.** Theory without practice is philosophy. Practice without theory is guesswork. You need both.

5. **Embrace struggle.** If you're not confused, you're not learning deeply enough. Spend time with problems that make you uncomfortable.

6. **Teach what you learn.** Writing blog posts, making videos, or explaining concepts to others forces you to truly understand them.

7. **Think in decades, not months.** Wang, Tulloch, and Pang didn't become experts overnight. Commit to years of deliberate practice.

---

## 3. Phase 1: Build Rock-Solid Foundations (Months 1-6)

This is the most important phase. Skip it, and everything built on top will be fragile.

### 3.1 Mathematics — The Language of AI

#### Linear Algebra (Critical Priority)
- Vectors, matrices, and tensor operations
- Matrix decompositions (SVD, eigendecomposition, QR, LU, Cholesky)
- Vector spaces, basis, rank, null space
- Norms (L1, L2, Frobenius) and their geometric meaning
- Positive definite matrices and their role in optimization
- Matrix calculus (Jacobians, Hessians)

**Resources:**
- **Book**: *Linear Algebra Done Right* by Sheldon Axler (theoretical elegance)
- **Book**: *Introduction to Linear Algebra* by Gilbert Strang (practical intuition)
- **Course**: MIT 18.06 (Gilbert Strang's lectures on MIT OCW) — watch every lecture
- **Course**: 3Blue1Brown's *Essence of Linear Algebra* (visual intuition — watch this FIRST)
- **Practice**: Implement matrix operations from scratch in Python/NumPy

#### Calculus & Multivariable Calculus
- Single-variable derivatives and integrals
- Partial derivatives, gradients, directional derivatives
- Chain rule (the backbone of backpropagation)
- Taylor series approximations
- Multivariable optimization (critical points, Hessians)
- Vector calculus basics (divergence, curl — less critical but useful)

**Resources:**
- **Course**: MIT 18.01 and 18.02 on OCW
- **Course**: 3Blue1Brown's *Essence of Calculus*
- **Book**: *Calculus* by James Stewart (for reference)
- **Key exercise**: Derive backpropagation for a 3-layer neural network by hand

#### Probability & Statistics
- Probability axioms, conditional probability, Bayes' theorem
- Random variables (discrete and continuous)
- Common distributions (Gaussian, Bernoulli, Binomial, Poisson, Exponential, Beta, Dirichlet)
- Expectation, variance, covariance, correlation
- Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP)
- Hypothesis testing and confidence intervals
- Information theory basics (entropy, KL divergence, mutual information, cross-entropy)
- Concentration inequalities (Markov, Chebyshev, Hoeffding — useful later)

**Resources:**
- **Book**: *Probability and Statistics for Engineers and Scientists* by Walpole, Myers, Myers
- **Book**: *All of Statistics* by Larry Wasserman (more advanced, excellent for ML)
- **Course**: MIT 6.041 Probabilistic Systems Analysis
- **Course**: Harvard Stat 110 (Joe Blitzstein — excellent intuition builder)

#### Optimization Theory
- Convex sets and convex functions
- Gradient descent and its variants
- Lagrange multipliers and constrained optimization
- Convex optimization basics (linear programming, quadratic programming)
- Duality theory (basics)
- Stochastic optimization

**Resources:**
- **Book**: *Convex Optimization* by Stephen Boyd and Lieven Vandenberghe (free PDF online)
- **Course**: Stanford EE364a (Boyd's lectures on YouTube)
- **Practice**: Implement gradient descent, Newton's method, and SGD from scratch

### 3.2 Programming Fundamentals

#### Python Mastery
- Data structures: lists, dicts, sets, tuples, deques
- Object-oriented programming (classes, inheritance, metaclasses)
- Functional programming (map, filter, reduce, lambda, closures, decorators)
- Generators, iterators, context managers
- Type hints and modern Python (3.10+)
- Profiling and performance optimization
- NumPy: vectorized operations, broadcasting, advanced indexing, memory layout
- Debugging with pdb/ipdb

**Resources:**
- **Book**: *Fluent Python* by Luciano Ramalho (go deep on Python internals)
- **Practice**: Solve 100+ problems on LeetCode/HackerRank in Python
- **Project**: Build a NumPy-only neural network library

#### Data Structures & Algorithms
- Arrays, linked lists, trees, graphs, hash tables, heaps
- Sorting algorithms (quicksort, mergesort — understand the analysis)
- Graph algorithms (BFS, DFS, Dijkstra, topological sort)
- Dynamic programming
- Complexity analysis (Big-O, amortized analysis)
- This matters because efficient ML code requires algorithmic thinking

**Resources:**
- **Book**: *Introduction to Algorithms* (CLRS) — the bible
- **Course**: MIT 6.006 Introduction to Algorithms
- **Practice**: LeetCode medium/hard problems, 2-3 per week minimum

#### Systems Fundamentals
- How computers work: CPU, memory hierarchy, cache, disk
- Operating system basics: processes, threads, virtual memory
- Networking basics: TCP/IP, HTTP, sockets
- Version control: Git (branching, rebasing, cherry-picking)
- Linux command line proficiency
- Basics of parallel and distributed computing

**Resources:**
- **Book**: *Computer Systems: A Programmer's Perspective* (CS:APP) by Bryant and O'Hallaron
- **Course**: MIT 6.004 Computation Structures
- **Practice**: Use Linux as your daily driver, automate everything with shell scripts

### 3.3 Foundation Projects

Build these to cement your understanding:

1. **Matrix Library**: Implement matrix multiplication, transpose, determinant, inverse, eigenvalues from scratch in Python (no NumPy)
2. **Statistical Analysis Tool**: Build a tool that computes distributions, performs hypothesis tests, and visualizes results
3. **Gradient Descent Visualizer**: Implement GD, SGD, momentum, Adam from scratch and visualize their paths on loss surfaces

---

## 4. Phase 2: Core Machine Learning (Months 4-12)

### 4.1 Classical Machine Learning (Understand Deeply, Don't Just Use)

#### Supervised Learning
- **Linear Regression**: Derive the closed-form solution. Implement it. Understand the probabilistic interpretation (MLE with Gaussian noise). Understand regularization (Ridge = L2 = Gaussian prior, Lasso = L1 = Laplace prior).
- **Logistic Regression**: Derive the gradient of cross-entropy loss. Implement from scratch. Understand the connection to GLMs.
- **Support Vector Machines**: Understand the primal and dual formulations. The kernel trick. Why maximizing margin works (VC theory intuition). Implement an SVM solver.
- **Decision Trees**: Information gain, Gini impurity. How splits are chosen. Pruning. Implement ID3/CART from scratch.
- **Ensemble Methods**: Bagging (Random Forests), Boosting (AdaBoost, Gradient Boosting, XGBoost). Why ensembles reduce variance. Implement a Random Forest.
- **k-Nearest Neighbors**: Distance metrics. Curse of dimensionality. KD-trees for efficient search.
- **Naive Bayes**: The independence assumption. Why it works despite being "wrong." Text classification with NB.

#### Unsupervised Learning
- **K-Means Clustering**: Implement it. Understand convergence guarantees. K-means++ initialization.
- **Gaussian Mixture Models (GMMs)**: EM algorithm — derive the E-step and M-step. Implement from scratch.
- **Principal Component Analysis (PCA)**: Connection to eigendecomposition and SVD. Implement it. Understand variance explained.
- **t-SNE and UMAP**: How they work, when to use them, their limitations.
- **Autoencoders**: Linear autoencoders recover PCA. Build the bridge to deep learning.

#### Learning Theory (What Separates Scientists from Practitioners)
- Bias-variance tradeoff (deeply, not just the diagram)
- PAC learning framework
- VC dimension
- Regularization as a form of Occam's razor
- Cross-validation (k-fold, leave-one-out, stratified)
- Overfitting, underfitting, and the double descent phenomenon

### 4.2 Key Resources for This Phase

- **Book**: *Pattern Recognition and Machine Learning* by Christopher Bishop (THE reference)
- **Book**: *The Elements of Statistical Learning* by Hastie, Tibshirani, Friedman (free PDF — mathematical depth)
- **Book**: *Machine Learning: A Probabilistic Perspective* by Kevin Murphy (comprehensive)
- **Course**: Stanford CS229 (Andrew Ng — the mathematical version, not Coursera)
- **Course**: Caltech CS156 (Yaser Abu-Mostafa — *Learning from Data*, phenomenal for theory)
- **Course**: Coursera ML Specialization by Andrew Ng (good starting point, but go beyond it)

### 4.3 Phase 2 Projects

1. **ML Library from Scratch**: Build a scikit-learn-like library with linear regression, logistic regression, decision trees, k-means, PCA. No external ML libraries.
2. **Kaggle Competitions**: Enter 2-3 tabular data competitions. Focus on feature engineering and understanding why things work.
3. **Paper Implementation**: Implement the original AdaBoost paper (Freund & Schapire, 1997) from scratch.

---

## 5. Phase 3: Deep Learning Mastery (Months 8-18)

### 5.1 Neural Network Foundations

#### Core Concepts (Implement All from Scratch First)
- Perceptrons and multilayer perceptrons
- Activation functions: sigmoid, tanh, ReLU, GELU, Swish — why each was invented
- Loss functions: MSE, cross-entropy, hinge loss — derive their gradients
- Backpropagation: Implement it manually. Understand computational graphs.
- Weight initialization: Xavier/Glorot, He, LSUV — why initialization matters
- Batch normalization, layer normalization, group normalization — implement each
- Dropout: Why it works (ensemble interpretation, Bayesian interpretation)
- Optimizers: SGD, momentum, RMSProp, Adam, AdamW, LAMB — implement each, understand their update rules

#### Regularization & Training Techniques
- L1/L2 regularization in the context of neural networks
- Data augmentation strategies
- Learning rate schedules: step decay, cosine annealing, warmup, cyclical LR
- Gradient clipping
- Mixed-precision training
- Early stopping

### 5.2 Convolutional Neural Networks (CNNs)

- Convolution operation (implement from scratch with loops, then with matrix multiplication)
- Pooling layers, stride, padding
- Classic architectures (study the papers, don't just use them):
  - LeNet-5 (1998) — where it all started
  - AlexNet (2012) — the deep learning revolution
  - VGGNet (2014) — depth matters
  - GoogLeNet/Inception (2014) — multi-scale features
  - ResNet (2015) — **read this paper carefully**, skip connections changed everything
  - DenseNet (2017) — feature reuse
  - EfficientNet (2019) — compound scaling
- Object detection: YOLO, Faster R-CNN, SSD (understand the architectures)
- Semantic segmentation: U-Net, DeepLab

### 5.3 Recurrent Neural Networks (RNNs) & Sequence Models

- Vanilla RNNs: Implement from scratch. Understand vanishing/exploding gradients.
- LSTMs: Understand each gate (forget, input, output). Implement from scratch.
- GRUs: Simplified LSTM. When to use which.
- Bidirectional RNNs
- Sequence-to-sequence models with attention (Bahdanau attention, Luong attention)
- Beam search decoding

### 5.4 The Transformer Architecture (CRITICAL)

This is the most important architecture in modern AI. Study it exhaustively.

- **The original paper**: "Attention Is All You Need" (Vaswani et al., 2017) — read it 5+ times
- Self-attention mechanism: Q, K, V — derive it from first principles
- Multi-head attention: Why multiple heads? What do different heads learn?
- Positional encoding: Sinusoidal, learned, rotary (RoPE), ALiBi
- Layer normalization: Pre-norm vs post-norm
- Feed-forward layers and their role
- **Implement a Transformer from scratch in NumPy/PyTorch** — this is non-negotiable
- Understand the computational complexity: O(n^2) attention, KV-cache for inference
- Variants: sparse attention, linear attention, flash attention

### 5.5 Generative Models

- **Variational Autoencoders (VAEs)**: ELBO derivation, reparameterization trick
- **Generative Adversarial Networks (GANs)**: Original GAN, DCGAN, StyleGAN, training instability
- **Normalizing Flows**: Change of variables, coupling layers
- **Diffusion Models**: Forward process, reverse process, score matching, DDPM, DDIM
- **Autoregressive Models**: PixelCNN, WaveNet, GPT-style generation

### 5.6 Framework Mastery

#### PyTorch (Primary — industry standard for research)
- Tensors, autograd, computational graphs
- `nn.Module`, custom layers, custom loss functions
- Data loading: `Dataset`, `DataLoader`, custom collate functions
- Distributed training: `DataParallel`, `DistributedDataParallel`
- TorchScript, ONNX export
- Profiling with PyTorch Profiler
- Custom CUDA kernels with PyTorch C++ extensions

#### JAX (Secondary — growing in research)
- Functional transformations: `jit`, `grad`, `vmap`, `pmap`
- Flax/Haiku for neural network modules
- XLA compilation
- Useful at Google-style research labs

### 5.7 Key Resources for This Phase

- **Book**: *Deep Learning* by Goodfellow, Bengio, Courville (the deep learning bible, free online)
- **Book**: *Dive into Deep Learning* (d2l.ai) — interactive, with code
- **Course**: Stanford CS231n (CNNs for Visual Recognition)
- **Course**: Stanford CS224n (NLP with Deep Learning)
- **Course**: NYU Deep Learning (Yann LeCun & Alfredo Canziani — excellent theoretical depth)
- **Course**: Fast.ai (practical deep learning — great complement to theoretical study)
- **Course**: MIT 6.S191 Introduction to Deep Learning

### 5.8 Phase 3 Projects

1. **Build a Transformer from scratch**: Implement the full encoder-decoder Transformer in PyTorch with only `torch.Tensor` operations (no `nn.Transformer`). Train it on a translation task.
2. **Implement ResNet from scratch**: Train on CIFAR-10, reproduce the paper's results.
3. **Build a GPT**: Follow Andrej Karpathy's "Let's Build GPT" but extend it. Add RoPE, GQA, SwiGLU.
4. **Train a Diffusion Model**: Implement DDPM from the paper, train on a dataset like CIFAR-10 or CelebA.
5. **Reproduce a paper**: Pick any influential deep learning paper and reproduce its results end-to-end.

---

## 6. Phase 4: Specialization & Research (Months 14-30)

Now you choose your area(s) of depth. Pick 1-2 to go very deep.

### 6.1 Natural Language Processing (NLP) / Large Language Models (LLMs)

#### Core Topics
- Word embeddings: Word2Vec, GloVe, FastText — understand the math
- Language modeling: n-gram models to neural LMs
- Pre-training paradigms: BERT (masked LM), GPT (autoregressive), T5 (text-to-text)
- Tokenization: BPE, WordPiece, SentencePiece, Unigram
- Scaling laws (Kaplan et al., Chinchilla paper) — understand how loss scales with compute, data, parameters
- Fine-tuning: Full fine-tuning, LoRA, QLoRA, prefix tuning, prompt tuning
- Reinforcement Learning from Human Feedback (RLHF): PPO, DPO, reward modeling
- Instruction tuning and alignment
- In-context learning, chain-of-thought reasoning, emergent abilities
- Retrieval-Augmented Generation (RAG)
- Long context: RoPE scaling, ring attention, landmark attention
- Evaluation: Perplexity, BLEU, ROUGE, human eval, benchmarks (MMLU, HumanEval, etc.)

#### Key Papers (Read All)
- "Attention Is All You Need" (2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- "Language Models are Few-Shot Learners" (GPT-3, 2020)
- "Training language models to follow instructions with human feedback" (InstructGPT, 2022)
- "LLaMA: Open and Efficient Foundation Language Models" (2023)
- "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
- "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
- "Constitutional AI" (Anthropic, 2022)
- "Direct Preference Optimization" (DPO, 2023)

### 6.2 Computer Vision

#### Core Topics
- Image classification, object detection, semantic/instance/panoptic segmentation
- Vision Transformers (ViT, DeiT, Swin Transformer)
- Self-supervised learning for vision: SimCLR, MoCo, DINO, MAE
- Multi-modal models: CLIP, DALL-E, Stable Diffusion
- 3D vision: NeRF, 3D Gaussian Splatting
- Video understanding: SlowFast, TimeSformer, VideoMAE
- Efficient architectures for edge deployment

#### Key Papers
- "An Image is Worth 16x16 Words" (ViT, 2020)
- "Masked Autoencoders Are Scalable Vision Learners" (MAE, 2021)
- "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021)
- "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)

### 6.3 Reinforcement Learning

#### Core Topics
- Markov Decision Processes (MDPs)
- Value functions, Bellman equations
- Dynamic programming: policy iteration, value iteration
- Monte Carlo methods
- Temporal Difference learning: TD(0), TD(lambda), SARSA, Q-learning
- Deep RL: DQN, Double DQN, Dueling DQN
- Policy gradient: REINFORCE, Actor-Critic, A2C, A3C
- PPO (Proximal Policy Optimization) — used extensively in RLHF
- SAC (Soft Actor-Critic)
- Model-based RL: Dreamer, MuZero
- Multi-agent RL
- Offline RL
- RL for LLM alignment (RLHF, RLAIF)

#### Key Resources
- **Book**: *Reinforcement Learning: An Introduction* by Sutton and Barto (the bible, free online)
- **Course**: David Silver's RL course (DeepMind/UCL)
- **Course**: UC Berkeley CS285 (Deep RL)

### 6.4 ML Systems & Infrastructure

This is what makes you industry-ready (think Andrew Tulloch's expertise).

#### Core Topics
- Distributed training: Data parallelism, model parallelism, pipeline parallelism, tensor parallelism
- Training frameworks: DeepSpeed, Megatron-LM, FSDP, ColossalAI
- GPU programming: CUDA basics, memory management, kernel optimization
- Quantization: INT8, INT4, GPTQ, AWQ, bitsandbytes
- Inference optimization: KV-cache, speculative decoding, continuous batching, vLLM
- Model serving: TensorRT, Triton Inference Server, ONNX Runtime
- MLOps: experiment tracking (W&B, MLflow), data versioning (DVC), model registries
- Compiler stack: XLA, TVM, Triton (OpenAI), torch.compile

#### Key Resources
- **Course**: Stanford CS149 (Parallel Computing)
- **Course**: CMU 10-414/714 (Deep Learning Systems — Tianqi Chen)
- **Paper**: "Megatron-LM" papers (NVIDIA)
- **Paper**: "FlashAttention" (Tri Dao)
- **Practice**: Write custom CUDA kernels, optimize model inference

### 6.5 Multimodal AI & Foundation Models

- Vision-Language Models: CLIP, LLaVA, GPT-4V, Gemini
- Text-to-Image: DALL-E, Stable Diffusion, Midjourney architectures
- Text-to-Video: Sora-style models, temporal consistency
- Audio/Speech: Whisper, AudioLM, MusicLM
- Robotics foundation models
- World models

---

## 7. Phase 5: Industry-Grade Engineering (Months 18-36)

### 7.1 Production ML Engineering

- Writing clean, tested, production-quality code
- Code review practices in ML teams
- Reproducibility: random seeds, deterministic operations, config management
- Experiment management and ablation studies
- Data pipelines: ETL, data validation, feature stores
- CI/CD for ML models
- Monitoring: data drift, model drift, performance degradation
- A/B testing and online evaluation

### 7.2 System Design for ML

- Designing ML systems end-to-end (feature engineering, training, serving, monitoring)
- Trade-offs: latency vs throughput, accuracy vs speed, cost vs performance
- Batch vs real-time inference
- Feature stores and feature engineering at scale
- Handling data at petabyte scale
- Multi-model architectures and model composition

### 7.3 C++ for ML (Andrew Tulloch's Edge)
- C++ fundamentals: memory management, RAII, smart pointers
- Template metaprogramming (used heavily in ML frameworks)
- Understanding PyTorch's C++ backend (ATen, c10)
- Writing custom operators
- SIMD vectorization
- Understanding BLAS/LAPACK

**Resources:**
- **Book**: *Effective Modern C++* by Scott Meyers
- **Practice**: Contribute to PyTorch's C++ codebase, write custom CUDA operators

### 7.4 Cloud & Infrastructure
- AWS/GCP/Azure ML services
- GPU cluster management
- Kubernetes for ML workloads
- Cost optimization for training runs
- Multi-GPU and multi-node training setups

---

## 8. Phase 6: Research & Publishing (Months 24-42)

### 8.1 How to Read Papers Effectively

1. **First pass** (10 min): Read title, abstract, introduction, conclusion, scan figures
2. **Second pass** (1 hour): Read the whole paper, skip proofs/details, understand the main idea
3. **Third pass** (4-5 hours): Reproduce the paper, verify every claim, implement it

### 8.2 How to Find Research Problems

- Read 3-5 papers per week in your area
- Maintain a research notebook of ideas
- Look for: limitations mentioned in papers, combinations of ideas from different fields, scaling existing approaches
- Attend reading groups (online or in-person)
- Follow top researchers on Twitter/X and read their threads

### 8.3 How to Write Papers

- Study the structure of top conference papers (NeurIPS, ICML, ICLR, ACL, CVPR)
- Lead with the result, not the method
- Ablation studies are critical
- Clear figures and tables matter enormously
- Get feedback early and often

### 8.4 Conference Ecosystem

- **Top ML Venues**: NeurIPS, ICML, ICLR
- **NLP**: ACL, EMNLP, NAACL
- **Computer Vision**: CVPR, ICCV, ECCV
- **AI General**: AAAI, IJCAI
- **Systems**: MLSys, OSDI, SOSP

### 8.5 Building Research Profile

- Publish at workshops first (lower bar, good feedback)
- Collaborate with established researchers
- Release code with papers (this massively increases impact)
- Write clear, well-documented code
- Engage with the community (present at meetups, tweet about your work)

---

## 9. Phase 7: Leadership & Impact (Months 36+)

### 9.1 Technical Leadership

- Mentoring junior researchers and engineers
- Setting research direction for a team
- Balancing exploration vs. exploitation in research bets
- Communicating technical vision to non-technical stakeholders
- Building and scaling ML teams

### 9.2 Industry Impact

- Translating research into products
- Understanding business metrics and how ML moves them
- Working with product managers, designers, data scientists
- Making principled trade-offs between perfect and shipped

### 9.3 Entrepreneurship Path (Alexandr Wang's Path)

- Identify a real problem that AI can solve at scale
- Understand the data moat
- Build an MVP quickly
- Fundraising and communicating a technical vision
- Hiring and building a world-class team

### 9.4 Research Scientist Path (Ruoming Pang's Path)

- PhD or equivalent research experience
- Deep expertise in a specific area
- Track record of publications at top venues
- Ability to lead research projects end-to-end
- Industry research labs: Google DeepMind, Meta FAIR, Anthropic, OpenAI, Microsoft Research

### 9.5 ML Engineering Leadership Path (Andrew Tulloch's Path)

- Deep systems expertise combined with ML knowledge
- Ability to optimize and scale ML systems
- Understanding of hardware (GPUs, TPUs) and compilers
- Building infrastructure that enables research
- Senior/Staff/Principal engineer roles at top companies

---

## 10. Detailed Resource List

### Must-Read Books (In Order)

| # | Book | Why |
|---|------|-----|
| 1 | *Linear Algebra Done Right* — Axler | Mathematical maturity for ML |
| 2 | *All of Statistics* — Wasserman | Probability and stats for ML |
| 3 | *Convex Optimization* — Boyd | Optimization foundations |
| 4 | *Pattern Recognition and Machine Learning* — Bishop | Classical ML bible |
| 5 | *Deep Learning* — Goodfellow et al. | Deep learning theory |
| 6 | *Reinforcement Learning* — Sutton & Barto | RL foundations |
| 7 | *Information Theory, Inference, and Learning Algorithms* — MacKay | Bayesian perspective |
| 8 | *Probabilistic Machine Learning (Advanced)* — Kevin Murphy | Modern comprehensive ML |
| 9 | *Designing Data-Intensive Applications* — Kleppmann | Systems for data/ML |
| 10 | *The Art of Doing Science and Engineering* — Hamming | Research mindset |

### Must-Take Courses (Online, Free)

| Course | Platform | Focus |
|--------|----------|-------|
| MIT 18.06 Linear Algebra | MIT OCW | Math foundations |
| Harvard Stat 110 | YouTube | Probability |
| Stanford CS229 | YouTube | ML theory |
| Caltech CS156 Learning from Data | YouTube | Learning theory |
| Stanford CS231n | YouTube | Computer vision |
| Stanford CS224n | YouTube | NLP |
| CMU 10-414 Deep Learning Systems | dlsyscourse.org | ML systems |
| UC Berkeley CS285 | YouTube | Deep RL |
| Stanford CS25 Transformers United | YouTube | Transformers |
| Fast.ai Practical Deep Learning | fast.ai | Practical skills |

### Essential YouTube Channels / Content Creators

- **3Blue1Brown**: Visual math intuition
- **Andrej Karpathy**: "Neural Networks: Zero to Hero" series — MANDATORY
- **Yannic Kilcher**: Paper explanations
- **Mu Li (d2l.ai)**: Deep learning textbook lectures
- **Two Minute Papers**: Stay current with research
- **StatQuest (Josh Starmer)**: Statistics made simple

### Essential Blogs & Websites

- **Distill.pub**: Beautiful interactive ML explanations (archived but still gold)
- **Lil'Log (Lilian Weng)**: Exceptional survey posts on ML topics
- **Jay Alammar**: Visual explanations of Transformers, BERT, GPT
- **Chris Olah's Blog (colah.github.io)**: Deep intuition for neural networks
- **The Gradient**: Long-form ML articles
- **Papers With Code**: Find SOTA and implementations

### ArXiv & Paper Reading

- **arxiv.org**: Daily paper submissions
- **arxiv-sanity-lite**: Curated paper feed
- **Semantic Scholar**: Paper search with citation graphs
- **Connected Papers**: Visual paper exploration

---

## 11. Weekly Study Schedule Template

### Beginner Phase (Phase 1-2): ~25-35 hours/week

| Day | Morning (2-3h) | Evening (2-3h) |
|-----|----------------|-----------------|
| Mon | Math (Linear Algebra) | Programming (Python/Algorithms) |
| Tue | Math (Probability) | ML Course (CS229/CS156) |
| Wed | Math (Calculus/Optimization) | Implementation (code from scratch) |
| Thu | ML Theory (read textbook) | Programming (projects) |
| Fri | ML Course (lectures) | Implementation (reproduce results) |
| Sat | Paper reading (1-2 papers) | Project work (build something) |
| Sun | Review & consolidate notes | Light reading / rest |

### Intermediate Phase (Phase 3-4): ~30-40 hours/week

| Day | Morning (2-3h) | Evening (2-3h) |
|-----|----------------|-----------------|
| Mon | Deep learning theory | Framework practice (PyTorch) |
| Tue | Paper reading (2-3 papers) | Paper implementation |
| Wed | Specialization study | Experiments / training runs |
| Thu | Systems / engineering | Large project work |
| Fri | Research / new ideas | Coding / debugging |
| Sat | Deep dive on one topic | Open source contribution |
| Sun | Write blog post / teach | Review & plan next week |

### Advanced Phase (Phase 5+): ~40-50+ hours/week

- Research / implementation: 25-30 hours
- Paper reading: 5-8 hours
- Writing / communication: 5 hours
- Systems / engineering: 5-10 hours
- Teaching / mentoring: 2-3 hours

---

## 12. Common Mistakes to Avoid

### Mistake 1: Tutorial Hell
**Problem**: Watching tutorial after tutorial without building anything.
**Fix**: For every hour of learning, spend at least an hour implementing.

### Mistake 2: Skipping Math
**Problem**: Jumping straight to deep learning without math foundations.
**Fix**: Invest 3-6 months in math before touching neural networks. It pays off 100x.

### Mistake 3: Only Using High-Level APIs
**Problem**: Only ever calling `model.fit()` in Keras/scikit-learn.
**Fix**: Implement everything from scratch at least once before using libraries.

### Mistake 4: Chasing the Latest Hype
**Problem**: Jumping to every new model/technique without depth in fundamentals.
**Fix**: Go deep on fundamentals. The latest GPT variant uses the same math from 2017's Transformer.

### Mistake 5: Not Reading Papers
**Problem**: Only reading blog posts and tutorials.
**Fix**: Read 2-3 papers per week. It gets easier with practice.

### Mistake 6: Working Alone
**Problem**: Not collaborating or getting feedback.
**Fix**: Join ML communities (Discord, Reddit r/MachineLearning, Twitter/X ML community), attend meetups, find study partners.

### Mistake 7: Ignoring Software Engineering
**Problem**: Writing messy, unreproducible code.
**Fix**: Treat ML code like production software. Write tests, use version control, document your work.

### Mistake 8: Not Building a Portfolio
**Problem**: Having knowledge but nothing to show for it.
**Fix**: Maintain a GitHub with well-documented projects, write blog posts, contribute to open source.

### Mistake 9: Perfectionism Before Shipping
**Problem**: Waiting until everything is perfect before showing your work.
**Fix**: Ship early, iterate. A mediocre project you finish teaches more than a perfect one you never start.

### Mistake 10: Neglecting Communication Skills
**Problem**: Being technically brilliant but unable to explain your work.
**Fix**: Write blog posts, give talks, explain concepts to non-experts. This is how you build influence and get hired.

---

## 13. Career Paths in AI

### Path 1: Research Scientist (Academia or Industry Lab)
- **Requirements**: PhD (usually), strong publication record, deep theoretical knowledge
- **Companies**: Google DeepMind, Meta FAIR, Anthropic, OpenAI, Microsoft Research, Apple ML Research
- **Day-to-day**: Reading papers, designing experiments, running training, writing papers
- **Salary range**: $200K-$600K+ (total compensation at top labs)

### Path 2: ML Engineer
- **Requirements**: Strong engineering skills, BS/MS in CS or equivalent, practical ML experience
- **Companies**: Any tech company, startups, scale-ups
- **Day-to-day**: Building ML systems, training/deploying models, writing production code
- **Salary range**: $150K-$500K+ (total compensation)

### Path 3: AI Infrastructure Engineer
- **Requirements**: Systems programming (C++, CUDA), distributed systems, compiler knowledge
- **Companies**: NVIDIA, Google (TensorFlow/JAX team), Meta (PyTorch team), startups (Modular, Anyscale)
- **Day-to-day**: Optimizing ML frameworks, writing kernels, building distributed training systems
- **Salary range**: $180K-$550K+ (total compensation)

### Path 4: Applied AI Scientist
- **Requirements**: MS/PhD preferred, publications helpful, strong engineering skills
- **Companies**: Tech companies, AI-first startups
- **Day-to-day**: Solving business problems with ML, prototyping and shipping models
- **Salary range**: $170K-$500K+ (total compensation)

### Path 5: AI Entrepreneur / Founder
- **Requirements**: Technical depth, business acumen, leadership, risk tolerance
- **Examples**: Alexandr Wang (Scale AI), Dario Amodei (Anthropic), Demis Hassabis (DeepMind)
- **Day-to-day**: Building product, fundraising, hiring, strategy, some technical work
- **Potential**: Unbounded

### Getting Hired at Top Companies

1. **Build a strong GitHub portfolio** with from-scratch implementations
2. **Publish research** (even at workshops)
3. **Contribute to open source** (PyTorch, HuggingFace, LangChain, etc.)
4. **Network**: Attend NeurIPS/ICML, engage on Twitter/X, reach out to researchers
5. **Prepare for interviews**: ML theory, coding, system design, paper discussions
6. **Internships**: If you're a student, intern at top AI labs (the #1 way to get hired)

---

## 14. Detailed Compensation Guide

AI is one of the highest-paying fields in technology. Below is a comprehensive breakdown of what you can expect at each career stage, by role, company tier, and geography. All figures are approximate **annual total compensation (TC)** in USD, including base salary, stock/equity, and bonuses (as of 2025-2026).

### 14.1 Compensation by Role & Experience Level

#### Research Scientist

| Level | Years of Exp | Top-Tier Lab (OpenAI, DeepMind, Anthropic) | Big Tech (Google, Meta, Apple) | Mid-Tier / Startup |
|-------|-------------|---------------------------------------------|-------------------------------|---------------------|
| Junior / New Grad (PhD) | 0-2 | $300K - $450K | $250K - $400K | $150K - $300K |
| Mid-Level | 3-5 | $400K - $700K | $350K - $600K | $200K - $400K |
| Senior | 5-10 | $600K - $1M+ | $500K - $900K | $300K - $600K |
| Staff / Principal | 10+ | $800K - $1.5M+ | $700K - $1.2M+ | $400K - $800K + significant equity |
| Distinguished / VP | 15+ | $1M - $3M+ | $1M - $2.5M+ | Founder-level equity |

**Breakdown for a typical Senior Research Scientist at a top lab:**
- Base salary: $250K - $350K
- Annual stock/RSU vesting: $200K - $500K
- Annual bonus: $50K - $150K
- Signing bonus (one-time): $50K - $200K

#### ML Engineer

| Level | Years of Exp | Top-Tier (FAANG+) | High-Growth Startup | Mid-Market Company |
|-------|-------------|--------------------|-----------------------|---------------------|
| Junior / New Grad | 0-2 | $180K - $280K | $120K - $220K + equity | $100K - $160K |
| Mid-Level (L4/E4) | 2-5 | $280K - $450K | $200K - $350K + equity | $140K - $220K |
| Senior (L5/E5) | 5-8 | $400K - $650K | $300K - $500K + equity | $180K - $300K |
| Staff (L6/E6) | 8-12 | $550K - $900K | $400K - $700K + equity | $250K - $400K |
| Principal (L7/E7) | 12+ | $800K - $1.3M+ | $500K - $1M + equity | $350K - $550K |

**Breakdown for a typical Senior ML Engineer at FAANG:**
- Base salary: $200K - $270K
- Annual stock/RSU vesting: $150K - $300K
- Annual bonus: $30K - $80K
- Signing bonus (one-time): $30K - $100K

#### AI Infrastructure / Systems Engineer

| Level | Years of Exp | NVIDIA / Top AI Infra | Big Tech | Startup |
|-------|-------------|------------------------|----------|---------|
| Junior | 0-2 | $180K - $300K | $170K - $260K | $120K - $200K |
| Mid-Level | 3-5 | $300K - $500K | $280K - $420K | $180K - $320K |
| Senior | 5-10 | $450K - $750K | $400K - $650K | $250K - $500K |
| Staff / Principal | 10+ | $700K - $1.2M+ | $600K - $1M+ | $350K - $700K + equity |

**Why it pays well:** CUDA/GPU expertise, distributed systems, and compiler knowledge are extremely scarce. If you can write custom kernels and optimize training infrastructure, you are in massive demand.

#### Applied AI / Data Scientist (ML-focused)

| Level | Years of Exp | Big Tech | Mid-Market | Non-Tech / Traditional |
|-------|-------------|----------|------------|------------------------|
| Junior | 0-2 | $150K - $230K | $100K - $150K | $80K - $120K |
| Mid-Level | 2-5 | $230K - $380K | $130K - $220K | $100K - $160K |
| Senior | 5-8 | $350K - $550K | $200K - $320K | $140K - $220K |
| Staff / Principal | 8+ | $500K - $800K | $280K - $450K | $200K - $320K |

#### AI Product / Technical Program Manager

| Level | Years of Exp | Big Tech | Growth-Stage Startup |
|-------|-------------|----------|----------------------|
| Mid-Level | 3-5 | $220K - $350K | $150K - $280K |
| Senior | 5-10 | $350K - $550K | $250K - $450K |
| Director | 10+ | $500K - $900K | $350K - $600K + equity |

### 14.2 Compensation by Company (2025-2026 Estimates)

#### Tier 1: AI-Native Labs (Highest Paying)

| Company | Senior IC TC | Staff+ IC TC | Notes |
|---------|-------------|--------------|-------|
| OpenAI | $500K - $900K | $800K - $1.5M+ | Profit participation units (PPUs), high equity upside |
| Anthropic | $450K - $800K | $700K - $1.3M+ | Significant equity grants, mission-driven |
| DeepMind | $400K - $750K | $650K - $1.2M+ | Google-level RSUs + DeepMind premiums |
| Scale AI | $350K - $600K | $500K - $900K | Strong equity packages |
| Cohere | $300K - $550K | $450K - $800K | Canadian company, competitive globally |
| Mistral AI | $250K - $500K | $400K - $700K+ | Paris-based, strong equity |

#### Tier 2: Big Tech AI Teams

| Company | Senior IC TC | Staff+ IC TC | Notes |
|---------|-------------|--------------|-------|
| Google (Brain/DeepMind) | $400K - $700K | $650K - $1.2M+ | L5-L7 levels, strong RSUs |
| Meta (FAIR / GenAI) | $400K - $700K | $600K - $1.1M+ | E5-E7, generous RSU refreshers |
| Apple (ML Research) | $350K - $600K | $550K - $1M+ | ICT4-ICT6, RSUs growing |
| Microsoft (Research / AI) | $350K - $600K | $550K - $950K | L63-L67, strong bonus structure |
| Amazon (AGI / AWS AI) | $300K - $500K | $450K - $800K | L6-L8, heavy back-loaded RSU vesting |
| NVIDIA | $350K - $650K | $550K - $1M+ | Massive GPU demand = strong stock performance |

#### Tier 3: AI Startups (High Risk, High Reward)

| Stage | Cash Salary | Equity Value (if successful) | Risk Level |
|-------|-------------|-------------------------------|------------|
| Pre-Seed / Seed | $100K - $180K | $500K - $5M+ (on paper) | Very High |
| Series A | $150K - $250K | $300K - $3M+ (on paper) | High |
| Series B-C | $200K - $350K | $200K - $2M+ (on paper) | Medium-High |
| Series D+ / Late Stage | $250K - $400K | $150K - $1M+ (more liquid) | Medium |

**Key insight:** Startup equity is a bet. Early employees at Scale AI, Anthropic, or OpenAI have equity worth millions. But most startups fail. If you join a startup, make sure you believe in the mission and team.

### 14.3 Compensation by Geography

| Location | Multiplier (vs. SF Bay Area) | Notes |
|----------|------------------------------|-------|
| San Francisco / Bay Area | 1.0x (baseline) | Highest absolute pay, highest cost of living |
| New York City | 0.90 - 1.0x | Competitive with SF for AI roles |
| Seattle / Bellevue | 0.90 - 0.95x | No state income tax = higher take-home |
| Los Angeles | 0.80 - 0.90x | Growing AI scene |
| Austin / Denver / Miami | 0.75 - 0.85x | Lower cost of living, growing tech hubs |
| Boston / Cambridge | 0.85 - 0.95x | Strong research ecosystem (MIT, Harvard) |
| London | 0.55 - 0.70x | Top research hub but lower pay than US |
| Toronto / Montreal | 0.50 - 0.65x | Strong AI research (Hinton, Bengio legacy), growing salaries |
| Zurich | 0.70 - 0.85x | Google DeepMind, ETH Zurich, high local salaries |
| Paris | 0.45 - 0.60x | Mistral, Meta FAIR, growing ecosystem |
| Berlin / Amsterdam | 0.40 - 0.55x | Growing but still below US levels |
| Bangalore / Hyderabad | 0.20 - 0.35x | Rapidly growing AI industry, lower cost of living |
| Beijing / Shanghai | 0.30 - 0.50x | Strong AI industry (ByteDance, Baidu, Alibaba) |
| Singapore | 0.50 - 0.65x | Growing AI hub, low taxes |
| Remote (US-based company) | 0.75 - 0.95x | Increasingly common, varies by company policy |

### 14.4 What Drives the Highest Compensation

The people earning $500K-$1M+ share these traits:

| Factor | Impact on Compensation |
|--------|----------------------|
| **Scarce technical skills** (CUDA, distributed training, LLM pretraining) | +30-60% premium |
| **Top-tier publications** (NeurIPS, ICML oral/spotlight) | +20-40% premium, opens top labs |
| **PhD from top program** (Stanford, MIT, CMU, Berkeley, etc.) | +15-30% at entry level |
| **Prior experience at a top lab** (OpenAI, DeepMind, FAIR) | +20-50% at next role |
| **Open-source impact** (major contributor to PyTorch, HuggingFace, etc.) | +15-30%, plus reputation |
| **Competing offers** | +20-50% (negotiation leverage is real) |
| **Willingness to relocate to SF/NYC** | +10-30% vs. remote/other cities |
| **Specialization in hot area** (LLMs, agents, robotics, inference optimization) | +20-40% premium |

### 14.5 Compensation Growth Over a Career

Here is a realistic trajectory for someone following this guide diligently:

| Career Stage | Timeline | Likely Role | Estimated TC (SF Bay Area) |
|-------------|----------|-------------|---------------------------|
| Learning / Student | Year 0-2 | Student / Self-study | $0 (investing in yourself) |
| First Job / Internship | Year 2-3 | ML Intern / Junior MLE | $80K - $180K |
| Early Career | Year 3-5 | ML Engineer / Junior Research Scientist | $180K - $350K |
| Mid Career | Year 5-8 | Senior MLE / Research Scientist | $350K - $650K |
| Established Expert | Year 8-12 | Staff Engineer / Senior Research Scientist | $550K - $1M+ |
| Industry Leader | Year 12+ | Principal / Distinguished / Director | $800K - $1.5M+ |
| Founder / Executive | Varies | CTO / CEO / VP of AI | Equity-driven, potentially $10M+ |

### 14.6 Beyond Cash: Non-Monetary Compensation

Top AI roles often come with significant non-cash benefits:

- **GPU/Compute access**: Some labs provide personal compute budgets ($50K-$500K/year in cloud credits)
- **Conference travel**: NeurIPS, ICML, ICLR attendance paid for ($3K-$10K/year)
- **Education budgets**: $5K-$20K/year for courses, books, conferences
- **Publication bonuses**: Some companies pay $5K-$20K per accepted top-venue paper
- **Sabbaticals**: Some research labs offer 1-3 month research sabbaticals
- **Immigration support**: H-1B, O-1A (extraordinary ability), green card sponsorship
- **Flexible work**: Many AI roles offer remote/hybrid options
- **Hardware**: High-end laptops, multiple monitors, home office stipends ($2K-$5K)
- **Impact**: Working on problems that affect billions of people

### 14.7 Negotiation Tips

1. **Always negotiate.** The first offer is almost never the best offer. Most companies expect negotiation.
2. **Get competing offers.** This is the single most powerful negotiation lever. Apply broadly.
3. **Know your market rate.** Use levels.fyi, Glassdoor, Blind, and Teamblind for real data.
4. **Negotiate equity, not just base.** At senior levels, equity is the majority of TC. Push for more RSUs/options.
5. **Ask for signing bonus.** This is often the easiest component for companies to increase.
6. **Consider the equity upside.** $300K at a Series A startup could be worth $3M if the company succeeds.
7. **Don't reveal your current salary.** In many US states, it's illegal for employers to ask.
8. **Negotiate refresh grants.** Annual equity refreshers vary enormously (ask about the refresh policy upfront).
9. **Factor in taxes.** $500K in Seattle (no state tax) > $550K in San Francisco (13.3% state tax).
10. **Think total lifetime earnings.** A role with better learning opportunities at slightly lower pay can yield far more over 10 years.

### 14.8 Key Compensation Data Sources

| Source | What It Provides | URL |
|--------|-----------------|-----|
| **levels.fyi** | Verified TC data by company, level, role | levels.fyi |
| **Glassdoor** | Salary ranges, reviews | glassdoor.com |
| **Blind / Teamblind** | Anonymous tech worker discussions, TC sharing | teamblind.com |
| **compensation.fyi** | Startup compensation data | compensation.fyi |
| **AI jobs boards** | ai-jobs.net, ML-focused job boards | Various |
| **H1B Salary Database** | Public H-1B wage data (base salary only) | h1bdata.info |
| **LinkedIn Salary** | Salary insights by role and location | linkedin.com |

---

## 15. How to Stay Current

### Daily (15-30 min)
- Skim arxiv-sanity or Papers With Code for new papers
- Check Twitter/X for discussions from top researchers
- Read 1-2 ML news items

### Weekly (3-5 hours)
- Read 2-3 papers in depth
- Watch 1-2 technical talks or lectures
- Work on ongoing projects

### Monthly
- Deep dive into a new topic or technique
- Write a blog post about something you learned
- Review and update your learning roadmap
- Attend a virtual meetup or reading group

### Quarterly
- Complete a significant project
- Submit to a workshop or conference (if in research)
- Evaluate your progress against your goals
- Update your resume/portfolio

---

## Final Words

The path to becoming an AI scientist at the level of Wang, Tulloch, or Pang is long and demanding. There are no shortcuts. But if you:

1. **Master the fundamentals** (math, programming, systems)
2. **Implement everything from scratch** (then use libraries)
3. **Read papers relentlessly** (start early, never stop)
4. **Build real things** (projects, products, research)
5. **Stay curious and humble** (the field moves fast)
6. **Contribute to the community** (open source, writing, teaching)

...you will get there.

The difference between a good ML practitioner and a great AI scientist is **depth**. Go deeper than everyone else. Understand *why* things work, not just *how* to use them. That's the real secret.

**Start today. Start with math. Implement everything. Read papers. Build things. Never stop.**

---

*"The best time to plant a tree was 20 years ago. The second best time is now."*

*Last updated: February 2026*

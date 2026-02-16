# The Complete Roadmap to Becoming an AI Scientist & Industry Expert

> **From Zero to AI Mastery** -- A deeply structured, multi-year guide inspired by the trajectories of Alexandr Wang (Scale AI), Andrew Tulloch (Meta AI / Anthropic), and Ruoming Pang (Google Brain / Apple ML).

---

## Table of Contents

1. [Understanding the Destination](#1-understanding-the-destination)
2. [Role Models & What Makes Them Exceptional](#2-role-models--what-makes-them-exceptional)
3. [Phase 0 -- Mindset & Learning Philosophy (Week 1-2)](#3-phase-0----mindset--learning-philosophy-week-1-2)
4. [Phase 1 -- Mathematical Foundations (Months 1-4)](#4-phase-1----mathematical-foundations-months-1-4)
5. [Phase 2 -- Programming & Computer Science Core (Months 2-5)](#5-phase-2----programming--computer-science-core-months-2-5)
6. [Phase 3 -- Classical Machine Learning (Months 4-7)](#6-phase-3----classical-machine-learning-months-4-7)
7. [Phase 4 -- Deep Learning Mastery (Months 6-12)](#7-phase-4----deep-learning-mastery-months-6-12)
8. [Phase 5 -- Specialization Tracks (Months 10-18)](#8-phase-5----specialization-tracks-months-10-18)
9. [Phase 6 -- Research & Paper Reading (Ongoing from Month 6)](#9-phase-6----research--paper-reading-ongoing-from-month-6)
10. [Phase 7 -- Building Real Systems at Scale (Months 12-24)](#10-phase-7----building-real-systems-at-scale-months-12-24)
11. [Phase 8 -- Contributing to the Field (Months 18-36)](#11-phase-8----contributing-to-the-field-months-18-36)
12. [Phase 9 -- Industry Expertise & Leadership (Months 24+)](#12-phase-9----industry-expertise--leadership-months-24)
13. [Daily Routine & Study Schedule](#13-daily-routine--study-schedule)
14. [Milestone Projects](#14-milestone-projects)
15. [Recommended Hardware & Tools Setup](#15-recommended-hardware--tools-setup)
16. [Key Conferences, Journals & Communities](#16-key-conferences-journals--communities)
17. [Common Pitfalls to Avoid](#17-common-pitfalls-to-avoid)
18. [Final Words](#18-final-words)

---

## 1. Understanding the Destination

An **AI Scientist / Industry Expert** is someone who:

- Has deep mathematical and theoretical understanding of learning algorithms
- Can read, reproduce, and extend cutting-edge research papers
- Builds production-grade AI systems that serve millions of users
- Understands the full stack: data pipelines, model architecture, training infrastructure, deployment, monitoring
- Contributes original ideas to the field through papers, open-source projects, or products
- Can lead teams and translate business problems into ML solutions

This is a **multi-year journey** (typically 3-5 years of intense, focused effort). There are no shortcuts, but the path below is the most efficient route.

---

## 2. Role Models & What Makes Them Exceptional

### Alexandr Wang -- Founder & CEO, Scale AI
- **Background**: Dropped out of MIT at 19; had been doing competitive math and programming since childhood
- **Key Traits**: Systems thinking, understanding that AI is only as good as its data, entrepreneurial vision
- **What to Learn**: The importance of **data infrastructure**, building tools that accelerate the entire AI ecosystem, thinking about AI from a product and business perspective
- **His Edge**: He saw that the bottleneck in AI was not algorithms but high-quality labeled data at scale

### Andrew Tulloch -- Research Engineer (Meta AI, Anthropic)
- **Background**: Mathematics degree; deep expertise in both theory and implementation
- **Key Traits**: Bridges the gap between mathematical rigor and high-performance engineering
- **What to Learn**: The power of combining **strong math fundamentals with systems-level engineering**; contributing to core ML frameworks (Caffe2, PyTorch)
- **His Edge**: Ability to optimize ML systems at the lowest levels (CUDA kernels, quantization, compiler optimizations) while understanding the theory

### Ruoming Pang -- Research Scientist (Google Brain, Apple ML)
- **Background**: PhD-level researcher; deep work in computer vision, object detection, neural architecture search
- **Key Traits**: Rigorous research methodology, systematic experimentation, prolific paper author
- **What to Learn**: How to conduct **world-class ML research**, design experiments, write papers, and push state-of-the-art
- **His Edge**: Combining architectural innovation (EfficientDet, NAS-FPN) with practical large-scale deployment

### Common Thread
All three share: **(1) rock-solid fundamentals**, **(2) relentless curiosity**, **(3) building real things**, **(4) deep technical depth in at least one area**, and **(5) ability to operate at the intersection of theory and practice**.

---

## 3. Phase 0 -- Mindset & Learning Philosophy (Week 1-2)

### Core Principles

1. **Learn by Doing**: For every concept you study, implement it from scratch
2. **Depth Over Breadth**: Understand *why* things work, not just *how* to use them
3. **First Principles Thinking**: Derive results yourself before looking at solutions
4. **Consistency Over Intensity**: 4 focused hours daily beats 12 sporadic hours
5. **Build in Public**: Share your learning, write blog posts, push code to GitHub
6. **Read Source Code**: Study how PyTorch, TensorFlow, and major libraries are implemented
7. **Embrace Struggle**: If it feels easy, you are not learning

### Set Up Your Learning Environment

```
Action Items:
- [ ] Create a dedicated GitHub account for your AI journey
- [ ] Set up a personal blog (GitHub Pages, Medium, or Substack)
- [ ] Join key communities (r/MachineLearning, ML Twitter/X, Discord servers)
- [ ] Install Linux (Ubuntu 22.04+ recommended) as your primary OS
- [ ] Set up Python environment with conda/mamba
- [ ] Create a structured note-taking system (Obsidian, Notion, or LaTeX notes)
- [ ] Bookmark arXiv.org, Papers With Code, Semantic Scholar
```

---

## 4. Phase 1 -- Mathematical Foundations (Months 1-4)

> *"Mathematics is the language in which God has written the universe."* -- Galileo
>
> Every great AI scientist has extremely strong math. This is non-negotiable.

### 4.1 Linear Algebra (Critical Priority)

**Why**: Every neural network operation is a matrix operation. Attention mechanisms, convolutions, embeddings -- all linear algebra.

**Topics to Master**:
- Vectors, matrices, tensors
- Matrix operations (multiplication, transpose, inverse)
- Vector spaces, basis, rank, null space
- Eigenvalues and eigenvectors (deeply!)
- Singular Value Decomposition (SVD)
- Positive definite matrices
- Matrix calculus (Jacobians, Hessians)
- Tensor operations and Einstein notation

**Resources**:
| Resource | Type | Notes |
|----------|------|-------|
| MIT 18.06 -- Gilbert Strang (YouTube) | Video Lectures | The gold standard; watch every lecture |
| "Linear Algebra Done Right" -- Sheldon Axler | Textbook | For theoretical depth |
| "Matrix Analysis and Applied Linear Algebra" -- Carl Meyer | Textbook | Applied perspective |
| 3Blue1Brown "Essence of Linear Algebra" | Video Series | For geometric intuition |
| MIT OCW 18.065 "Matrix Methods in Data Analysis" | Video Lectures | Directly connects to ML |

**Exercises**:
- Implement matrix multiplication from scratch in Python (no NumPy)
- Implement SVD and use it for image compression
- Implement PCA from scratch using eigendecomposition
- Prove that symmetric matrices have real eigenvalues

### 4.2 Calculus & Multivariate Calculus

**Why**: Backpropagation is just the chain rule. Optimization requires understanding gradients, curvature, and convergence.

**Topics to Master**:
- Limits, derivatives, integrals (review)
- Partial derivatives, gradients
- Chain rule (single and multivariate) -- **this is backpropagation**
- Taylor series and approximations
- Multivariable optimization (critical points, saddle points)
- Lagrange multipliers (for constrained optimization)
- Vector calculus (divergence, curl -- useful for physics-informed ML)

**Resources**:
| Resource | Type | Notes |
|----------|------|-------|
| MIT 18.01 & 18.02 (OCW) | Video Lectures | Comprehensive calculus |
| 3Blue1Brown "Essence of Calculus" | Video Series | Build intuition |
| "Calculus" -- Michael Spivak | Textbook | Rigorous treatment |
| "Matrix Calculus for Deep Learning" -- Parr & Howard | Paper | Directly applicable |

**Exercises**:
- Derive the gradient of a simple neural network loss by hand
- Implement gradient descent from scratch for a 2D function
- Visualize gradient fields and loss landscapes

### 4.3 Probability & Statistics

**Why**: ML is fundamentally about learning probability distributions from data. Bayesian thinking, generative models, and uncertainty estimation all require strong probability foundations.

**Topics to Master**:
- Probability axioms, conditional probability, Bayes' theorem
- Random variables (discrete and continuous)
- Common distributions (Gaussian, Bernoulli, Poisson, Exponential, Beta, Dirichlet)
- Expectation, variance, covariance, correlation
- Joint, marginal, and conditional distributions
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) estimation
- Bayesian inference fundamentals
- Information theory (entropy, KL divergence, mutual information, cross-entropy)
- Central Limit Theorem, Law of Large Numbers
- Concentration inequalities (Hoeffding, Markov, Chebyshev)

**Resources**:
| Resource | Type | Notes |
|----------|------|-------|
| MIT 6.041 / Harvard Stat 110 (YouTube) | Video Lectures | Excellent probability courses |
| "Probability and Statistics for Engineers" -- Sheldon Ross | Textbook | Clear and applied |
| "All of Statistics" -- Larry Wasserman | Textbook | Compact but comprehensive |
| "Information Theory, Inference and Learning Algorithms" -- David MacKay | Textbook | Free online; brilliant |

**Exercises**:
- Derive the MLE for a Gaussian distribution
- Implement a Naive Bayes classifier from scratch
- Implement Bayesian inference with conjugate priors
- Compute KL divergence between two distributions

### 4.4 Optimization Theory

**Why**: Training any ML model is an optimization problem. Understanding optimization deeply separates practitioners from scientists.

**Topics to Master**:
- Convex vs non-convex optimization
- Gradient descent and its variants
- Convergence analysis and learning rate schedules
- Stochastic optimization (SGD, mini-batch)
- Momentum methods (Nesterov, Adam, AdaGrad, RMSProp)
- Second-order methods (Newton's method, L-BFGS)
- Constrained optimization (KKT conditions)
- Duality theory (Lagrangian duality)

**Resources**:
| Resource | Type | Notes |
|----------|------|-------|
| "Convex Optimization" -- Boyd & Vandenberghe | Textbook | The bible; free online |
| Stanford EE364a (YouTube) | Video Lectures | Boyd's own course |
| "Numerical Optimization" -- Nocedal & Wright | Textbook | For depth in algorithms |
| "An Introduction to Optimization on Smooth Manifolds" -- Boumal | Textbook | Advanced; for later |

**Exercises**:
- Implement SGD, Adam, and RMSProp from scratch
- Compare convergence rates on the Rosenbrock function
- Implement line search and trust region methods
- Prove convergence of gradient descent for convex functions

---

## 5. Phase 2 -- Programming & Computer Science Core (Months 2-5)

> Start this alongside math. You need both to progress.

### 5.1 Python Mastery

**Goal**: Python should feel like a natural language to you.

**Topics**:
- Core Python (data structures, OOP, generators, decorators, context managers)
- NumPy (vectorized operations, broadcasting, memory layout)
- Pandas (data manipulation at scale)
- Matplotlib / Seaborn / Plotly (visualization)
- Writing clean, tested, documented code
- Profiling and performance optimization
- Type hints and modern Python practices

**Resources**:
| Resource | Type | Notes |
|----------|------|-------|
| "Fluent Python" -- Luciano Ramalho | Book | Master-level Python |
| "Python for Data Analysis" -- Wes McKinney | Book | NumPy/Pandas deep dive |
| NumPy documentation internals | Docs | Understand broadcasting and memory |
| Real Python (website) | Tutorials | Practical intermediate-advanced topics |

**Project**: Build a complete data analysis pipeline from scratch -- data loading, cleaning, transformation, visualization, and statistical analysis -- without using high-level abstractions.

### 5.2 Data Structures & Algorithms

**Why**: Efficient ML systems require strong CS fundamentals. Interview preparation at top AI labs requires this.

**Topics**:
- Arrays, linked lists, stacks, queues, hash tables
- Trees (binary, BST, heaps, tries)
- Graphs (BFS, DFS, shortest path, topological sort)
- Dynamic programming
- Sorting and searching
- Complexity analysis (Big-O)
- Advanced: Bloom filters, skip lists, LSH (locality-sensitive hashing)

**Resources**:
| Resource | Type | Notes |
|----------|------|-------|
| "Introduction to Algorithms" (CLRS) | Textbook | The comprehensive reference |
| LeetCode (Medium/Hard problems) | Practice | Aim for 200+ problems |
| MIT 6.006 (OCW) | Video Lectures | Excellent algorithm course |
| "Algorithm Design Manual" -- Skiena | Textbook | Practical approach |

### 5.3 Systems Programming Fundamentals

**Why**: Andrew Tulloch's edge comes from understanding systems deeply. High-performance ML requires this.

**Topics**:
- C/C++ basics (pointers, memory management, templates)
- Computer architecture (CPU caches, SIMD, memory hierarchy)
- Parallel programming (threads, processes, async)
- GPU programming basics (CUDA concepts)
- Linux systems (shell scripting, process management, networking)
- Docker and containerization
- Version control (Git -- advanced usage)

**Resources**:
| Resource | Type | Notes |
|----------|------|-------|
| "Computer Systems: A Programmer's Perspective" (CS:APP) | Textbook | Essential systems knowledge |
| "Programming in C++" -- Stroustrup | Textbook | C++ reference |
| CUDA Programming Guide (NVIDIA) | Docs | When ready for GPU programming |
| "The Linux Command Line" -- William Shotts | Book | Free online |

---

## 6. Phase 3 -- Classical Machine Learning (Months 4-7)

> **Critical**: Do NOT skip this for deep learning. Classical ML teaches you the *principles* that everything else builds on.

### 6.1 Supervised Learning

**Implement each algorithm from scratch before using sklearn**:

- **Linear Regression**: OLS, Ridge (L2), Lasso (L1), Elastic Net
  - Derive the closed-form solution
  - Understand the bias-variance tradeoff
  - Implement with gradient descent AND normal equations

- **Logistic Regression**: Binary and multinomial
  - Derive the gradient of cross-entropy loss
  - Implement Newton-Raphson optimization
  - Understand decision boundaries geometrically

- **Support Vector Machines**: Linear and kernel SVMs
  - Understand the maximum margin principle
  - Derive the dual formulation
  - Implement SMO algorithm
  - Understand the kernel trick deeply

- **Decision Trees & Ensembles**:
  - Implement CART from scratch (Gini impurity, information gain)
  - Random Forests: understand bagging and feature randomization
  - Gradient Boosting: derive the gradient boosting algorithm
  - XGBoost: understand the regularized objective

- **k-Nearest Neighbors**: Implement with KD-trees for efficiency

- **Naive Bayes**: Gaussian and multinomial variants

### 6.2 Unsupervised Learning

- **K-Means Clustering**: Implement from scratch; understand K-Means++
- **Gaussian Mixture Models**: Implement EM algorithm from scratch
- **PCA**: Implement via eigendecomposition and SVD
- **t-SNE and UMAP**: Understand the math; use for visualization
- **Hierarchical Clustering**: Implement agglomerative clustering

### 6.3 Core ML Theory

- **Bias-Variance Tradeoff**: Derive the decomposition; understand deeply
- **Cross-Validation**: k-fold, stratified, leave-one-out
- **Regularization**: L1, L2, dropout (preview); why they work
- **Feature Engineering**: Feature selection, creation, encoding
- **Model Evaluation**: Precision, recall, F1, ROC-AUC, calibration
- **Statistical Learning Theory**: VC dimension, PAC learning (intro)

### Resources

| Resource | Type | Notes |
|----------|------|-------|
| Stanford CS229 -- Andrew Ng (YouTube) | Video Lectures | The classic ML course |
| "Pattern Recognition and Machine Learning" -- Bishop (PRML) | Textbook | Mathematical and thorough |
| "The Elements of Statistical Learning" (ESL) -- Hastie et al. | Textbook | Free online; comprehensive |
| "An Introduction to Statistical Learning" (ISLR) | Textbook | Gentler intro; free online |
| "Machine Learning: A Probabilistic Perspective" -- Kevin Murphy | Textbook | Encyclopedic reference |
| scikit-learn documentation & source code | Docs/Code | Study the implementations |

### Phase 3 Capstone Project
Build an end-to-end ML pipeline: data collection, EDA, feature engineering, model selection (try 5+ algorithms), hyperparameter tuning, evaluation, and a simple API to serve predictions. Use a real-world dataset (e.g., Kaggle competition data).

---

## 7. Phase 4 -- Deep Learning Mastery (Months 6-12)

> This is where the modern AI revolution lives. But it only makes sense if you have the foundations from Phases 1-3.

### 7.1 Neural Network Fundamentals

**Build from scratch first (pure NumPy), then learn PyTorch**:

- Perceptrons and multi-layer perceptrons
- Forward propagation (implement manually)
- Backpropagation (derive and implement from scratch)
- Activation functions (ReLU, sigmoid, tanh, GELU, Swish) -- understand gradients
- Loss functions (MSE, cross-entropy, hinge) -- derive gradients
- Weight initialization (Xavier, He, Lecun)
- Batch normalization (derive forward and backward pass)
- Layer normalization, Group normalization, RMSNorm
- Dropout (understand as approximate Bayesian inference)
- Residual connections (understand why they enable depth)

### 7.2 PyTorch Deep Dive

**Learn PyTorch at three levels**:

1. **User Level**: nn.Module, DataLoader, optimizers, training loops
2. **Intermediate**: Custom layers, custom loss functions, hooks, mixed precision
3. **Internal**: Autograd engine, tensor storage, dispatch mechanism, TorchScript

| Resource | Type | Notes |
|----------|------|-------|
| PyTorch official tutorials | Tutorials | Start here |
| "Deep Learning with PyTorch" -- Stevens et al. | Book | Comprehensive PyTorch book |
| PyTorch source code on GitHub | Code | Study the internals |
| PyTorch internals blog by Edward Yang | Blog | How autograd really works |

### 7.3 Convolutional Neural Networks (CNNs)

- Convolution operation (implement from scratch)
- Pooling, stride, padding, dilation
- Classic architectures: **LeNet -> AlexNet -> VGG -> GoogLeNet -> ResNet -> DenseNet**
- Modern architectures: **EfficientNet, ConvNeXt, RegNet**
- Object detection: **YOLO, SSD, Faster R-CNN, DETR** (Ruoming Pang's area)
- Semantic segmentation: **U-Net, DeepLab, Mask R-CNN**
- Neural Architecture Search (NAS) -- study NAS-FPN (Pang et al.)
- Understand receptive fields, feature maps, transfer learning

### 7.4 Recurrent Neural Networks & Sequence Models

- Vanilla RNNs (implement from scratch; understand vanishing gradients)
- LSTM (derive the gates; implement from scratch)
- GRU (understand as simplified LSTM)
- Bidirectional RNNs
- Sequence-to-sequence models
- Attention mechanisms (Bahdanau, Luong)
- **These are largely superseded by Transformers but understanding them gives critical context**

### 7.5 Transformers -- The Architecture That Changed Everything

**This deserves deep, obsessive study**:

- Self-attention mechanism (derive from scratch)
- Multi-head attention
- Positional encoding (sinusoidal, learned, RoPE, ALiBi)
- The original Transformer architecture (encoder-decoder)
- Pre-norm vs post-norm
- Flash Attention and efficient attention variants
- Implement a complete Transformer from scratch in PyTorch

**Key Models to Study in Depth**:
- **BERT** (bidirectional encoder; masked language modeling)
- **GPT series** (autoregressive decoder; next-token prediction)
- **T5** (encoder-decoder; text-to-text framework)
- **Vision Transformer (ViT)** (applying Transformers to images)
- **LLaMA / Mistral / Gemma** (modern open LLMs)

### 7.6 Generative Models

- **Variational Autoencoders (VAEs)**: Derive the ELBO; implement from scratch
- **Generative Adversarial Networks (GANs)**: Original GAN, DCGAN, StyleGAN, Wasserstein GAN
- **Diffusion Models**: DDPM, Score-based models, Stable Diffusion
- **Autoregressive Models**: PixelCNN, WaveNet, GPT
- **Flow-based Models**: RealNVP, Glow, normalizing flows

### 7.7 Training Deep Networks at Scale

- Mixed precision training (FP16, BF16, FP8)
- Gradient accumulation and large batch training
- Learning rate schedules (warmup, cosine annealing, cyclic)
- Distributed training (Data Parallel, Model Parallel, Pipeline Parallel)
- DeepSpeed, FSDP (Fully Sharded Data Parallel)
- Gradient checkpointing (memory vs compute tradeoff)
- Debugging training runs (loss curves, gradient norms, activation statistics)

### Core Deep Learning Resources

| Resource | Type | Notes |
|----------|------|-------|
| Stanford CS231n (CNNs for Visual Recognition) | Course | Essential for vision |
| Stanford CS224n (NLP with Deep Learning) | Course | Essential for NLP |
| Stanford CS25 (Transformers United) | Course | Transformer deep-dive |
| "Deep Learning" -- Goodfellow, Bengio, Courville | Textbook | The deep learning bible; free online |
| "Dive into Deep Learning" (d2l.ai) | Interactive Book | Code + theory; free |
| fast.ai courses (Parts 1 & 2) | Course | Top-down practical approach |
| Andrej Karpathy's "Neural Networks: Zero to Hero" | YouTube Series | Build GPT from scratch |
| "The Illustrated Transformer" -- Jay Alammar | Blog | Best visual explanation |
| Lilian Weng's blog (lilianweng.github.io) | Blog | Outstanding ML summaries |

---

## 8. Phase 5 -- Specialization Tracks (Months 10-18)

> Choose one primary specialization but maintain breadth. The best AI scientists are T-shaped: deep in one area, broad across many.

### Track A: Large Language Models & NLP

- Pre-training objectives (MLM, CLM, denoising)
- Tokenization (BPE, WordPiece, SentencePiece, Unigram)
- Scaling laws (Chinchilla, Kaplan et al.)
- RLHF (Reinforcement Learning from Human Feedback)
- DPO (Direct Preference Optimization)
- Constitutional AI and alignment
- Prompt engineering and in-context learning
- Retrieval Augmented Generation (RAG)
- Fine-tuning: LoRA, QLoRA, prefix tuning, adapters
- Evaluation: perplexity, BLEU, ROUGE, human evaluation, benchmarks
- Inference optimization: KV-cache, speculative decoding, quantization (GPTQ, AWQ)
- **Study**: GPT-4 technical report, LLaMA papers, Anthropic papers

### Track B: Computer Vision

- Image classification, detection, segmentation (review)
- 3D vision (depth estimation, point clouds, NeRF)
- Video understanding (temporal modeling, action recognition)
- Multi-modal models (CLIP, DALL-E, Flamingo)
- Vision-Language Models (LLaVA, GPT-4V)
- Self-supervised learning (SimCLR, DINO, MAE)
- **Study**: Ruoming Pang's work on EfficientDet, NAS-FPN, SpineNet

### Track C: Reinforcement Learning

- Markov Decision Processes
- Value-based methods (Q-learning, DQN, Double DQN)
- Policy gradient methods (REINFORCE, PPO, A3C)
- Actor-Critic methods
- Model-based RL (MuZero, Dreamer)
- Multi-agent RL
- RL from Human Feedback (bridges to LLMs)
- **Study**: Silver's RL course, Sutton & Barto textbook

### Track D: ML Systems & Infrastructure

- Model serving (TorchServe, Triton Inference Server, vLLM)
- Training infrastructure (distributed training, cluster management)
- ML Ops (experiment tracking, model versioning, CI/CD for ML)
- Data pipelines (Apache Beam, Spark, data versioning)
- Hardware optimization (GPU/TPU utilization, kernel optimization)
- Compiler optimization for ML (XLA, TVM, Triton)
- **Study**: Andrew Tulloch's work on quantization, FBGEMM, Caffe2

---

## 9. Phase 6 -- Research & Paper Reading (Ongoing from Month 6)

### How to Read Research Papers

1. **First Pass (10 min)**: Title, abstract, figures, conclusion -- decide if worth reading
2. **Second Pass (1 hour)**: Read fully but skip proofs; understand the key ideas, methodology, and results
3. **Third Pass (4-5 hours)**: Reproduce the key results; verify proofs; identify limitations
4. **Implementation Pass**: Re-implement the paper from scratch

### Essential Papers to Read (Chronological Foundation)

**Foundational (read these carefully)**:
1. "A Few Useful Things to Know about Machine Learning" -- Pedro Domingos (2012)
2. "ImageNet Classification with Deep CNNs" -- Krizhevsky et al. (AlexNet, 2012)
3. "Dropout: A Simple Way to Prevent Overfitting" -- Srivastava et al. (2014)
4. "Batch Normalization" -- Ioffe & Szegedy (2015)
5. "Deep Residual Learning" -- He et al. (ResNet, 2015)
6. "Attention Is All You Need" -- Vaswani et al. (Transformer, 2017)

**Language Models**:
7. "BERT" -- Devlin et al. (2018)
8. "Language Models are Few-Shot Learners" -- Brown et al. (GPT-3, 2020)
9. "Training Language Models to Follow Instructions with Human Feedback" -- Ouyang et al. (InstructGPT, 2022)
10. "LLaMA: Open and Efficient Foundation Language Models" -- Touvron et al. (2023)
11. "Scaling Laws for Neural Language Models" -- Kaplan et al. (2020)
12. "Direct Preference Optimization" -- Rafailov et al. (DPO, 2023)

**Computer Vision**:
13. "An Image is Worth 16x16 Words" -- Dosovitskiy et al. (ViT, 2020)
14. "EfficientDet: Scalable and Efficient Object Detection" -- Tan, Pang et al. (2020)
15. "NAS-FPN: Learning Scalable Feature Pyramid Architecture" -- Ghiasi, Lin, Pang et al. (2019)
16. "Denoising Diffusion Probabilistic Models" -- Ho et al. (DDPM, 2020)
17. "Learning Transferable Visual Models From Natural Language Supervision" -- Radford et al. (CLIP, 2021)

**Systems & Scaling**:
18. "Megatron-LM: Training Multi-Billion Parameter Language Models" (2019)
19. "ZeRO: Memory Optimizations for Training Billion Parameter Models" -- Rajbhandari et al. (2020)
20. "FlashAttention: Fast and Memory-Efficient Exact Attention" -- Dao et al. (2022)

### Paper Reading Routine
- Read **2-3 papers per week** minimum
- Keep a paper reading log with summaries and key takeaways
- Implement at least **1 paper per month** from scratch
- Use tools: arXiv, Semantic Scholar, Papers With Code, Connected Papers

---

## 10. Phase 7 -- Building Real Systems at Scale (Months 12-24)

> This is what separates an AI scientist from a student. You must build and deploy real systems.

### Project 1: Train a Language Model from Scratch
- Collect and clean a text dataset (Common Crawl subset, Wikipedia, etc.)
- Implement a GPT-style model in PyTorch
- Train with distributed data parallelism across multiple GPUs
- Implement proper tokenizer (BPE from scratch)
- Evaluate on standard benchmarks
- Fine-tune with instruction following data
- Deploy with a simple API

### Project 2: Build an Object Detection System
- Implement a detection framework (YOLO or DETR style)
- Train on COCO dataset
- Optimize for inference speed (quantization, TensorRT)
- Deploy as a real-time inference service
- Measure and improve latency, throughput

### Project 3: Build a RAG System
- Implement document chunking and embedding
- Build a vector store (implement approximate nearest neighbor search)
- Integrate with an LLM for generation
- Handle evaluation of retrieval quality and generation quality
- Deploy as a production service

### Project 4: Contribute to Open Source ML
- Contribute to PyTorch, Hugging Face Transformers, or vLLM
- Fix bugs, improve documentation, add features
- Engage with the maintainer community
- This builds your reputation and teaches you production-quality code

### Project 5: Reproduce a State-of-the-Art Paper
- Choose a recent paper (< 1 year old)
- Reproduce results from scratch (not using author's code)
- Write a detailed blog post about the experience
- Identify and report any discrepancies

---

## 11. Phase 8 -- Contributing to the Field (Months 18-36)

### Getting into Research

1. **Identify open problems** in your specialization area
2. **Generate hypotheses** based on your deep understanding
3. **Design rigorous experiments** with proper baselines and ablations
4. **Write clear papers** following top venue standards
5. **Submit to top conferences**: NeurIPS, ICML, ICLR, CVPR, ACL, EMNLP

### Research Methodology

- Always start with a thorough **literature review**
- Define your **research question** precisely
- Design experiments with **proper controls**
- Use **ablation studies** to understand what matters
- Report **confidence intervals** and **statistical significance**
- Make your code and data **reproducible**
- Write **clearly and honestly** about limitations

### Building Your Research Profile

- Publish papers at top venues
- Give talks at workshops and meetups
- Create high-quality GitHub repositories
- Write technical blog posts
- Engage on Twitter/X with the ML community
- Review papers for conferences (volunteer as a reviewer)
- Mentor junior researchers

### Alternative: Industry Research

If not pursuing academia:
- Apply for research engineer / research scientist roles
- Build a portfolio of implemented papers and projects
- Contribute to company tech blogs
- File patents on novel techniques
- Present at internal and external conferences

---

## 12. Phase 9 -- Industry Expertise & Leadership (Months 24+)

### Becoming an Industry AI Expert

**Technical Leadership**:
- Lead the design of ML systems end-to-end
- Make architectural decisions that impact products at scale
- Mentor and grow other ML engineers
- Define technical roadmaps for AI teams
- Bridge research and production

**Business & Product Sense** (Alexandr Wang's strength):
- Understand how AI creates business value
- Identify the right problems to solve with ML
- Communicate technical concepts to non-technical stakeholders
- Think about data strategy and competitive moats
- Understand AI safety, ethics, and governance

**Continuous Learning**:
- The field moves fast; allocate 20% of your time to staying current
- Attend NeurIPS, ICML, ICLR conferences
- Maintain relationships with academic researchers
- Run internal reading groups and paper discussions
- Experiment with new techniques on side projects

### Career Paths

```
                         ┌─────────────────────┐
                         │  AI/ML Researcher    │
                         │  (Academia/Industry) │
                         └────────┬────────────┘
                                  │
            ┌─────────────────────┼──────────────────────┐
            │                     │                      │
  ┌─────────▼──────────┐ ┌───────▼────────┐ ┌──────────▼──────────┐
  │ Research Scientist  │ │ ML Engineer    │ │ AI Product/Business │
  │ (Pang's path)      │ │ (Tulloch's     │ │ (Wang's path)       │
  │                     │ │  path)         │ │                     │
  │ - Publish papers    │ │ - Build infra  │ │ - Build companies   │
  │ - Push SOTA         │ │ - Scale systems│ │ - AI strategy       │
  │ - Lead research     │ │ - Optimize     │ │ - Product vision    │
  │   teams             │ │   performance  │ │ - Fundraise/lead    │
  └─────────────────────┘ └────────────────┘ └─────────────────────┘
```

---

## 13. Daily Routine & Study Schedule

### Recommended Daily Schedule (4-6 hours of focused study)

```
Morning Block (2 hours):
  - 30 min: Review flashcards / previous day's notes
  - 90 min: Deep study (math, theory, or reading papers)

Afternoon Block (2-3 hours):
  - 120-180 min: Coding / implementation / projects
  - Focus on implementing what you learned in the morning

Evening Block (1 hour):
  - 30 min: Read 1 paper (first or second pass)
  - 30 min: Write notes, update blog, push code to GitHub
```

### Weekly Structure

| Day | Focus Area |
|-----|-----------|
| Monday | Math fundamentals + Implementation |
| Tuesday | ML/DL theory + Paper reading |
| Wednesday | Coding project work |
| Thursday | ML/DL theory + Paper implementation |
| Friday | Coding project work |
| Saturday | Deep dive into one topic + Blog writing |
| Sunday | Review week's learning + Plan next week |

### Monthly Milestones

- Complete at least one course module
- Implement at least one algorithm/paper from scratch
- Write at least one blog post
- Solve 20+ coding problems (LeetCode/similar)
- Read 8-12 papers

---

## 14. Milestone Projects

### Beginner Projects (Months 1-4)
1. **Linear Regression from Scratch**: Implement gradient descent, normal equations, regularization
2. **Logistic Regression from Scratch**: Binary classification with gradient-based optimization
3. **Neural Network from Scratch**: Forward prop, backprop, training -- NumPy only
4. **Image Classifier**: CNN on CIFAR-10 using PyTorch

### Intermediate Projects (Months 4-8)
5. **Sentiment Analyzer**: LSTM-based and then Transformer-based
6. **Object Detector**: Implement YOLO or SSD from scratch
7. **GAN for Image Generation**: Train DCGAN on CelebA
8. **Recommendation System**: Collaborative filtering + deep learning hybrid

### Advanced Projects (Months 8-14)
9. **Mini-GPT**: Train a small language model from scratch
10. **Diffusion Model**: Implement DDPM for image generation
11. **Speech Recognition**: End-to-end ASR system
12. **Multi-modal Model**: Image captioning with attention

### Expert Projects (Months 14+)
13. **Train a 1B+ Parameter LLM**: Distributed training, proper data pipeline
14. **Novel Architecture**: Design and evaluate a new model architecture
15. **Paper Reproduction**: Reproduce a top-tier conference paper completely
16. **Open Source Contribution**: Major feature/fix to PyTorch, HF, or similar

---

## 15. Recommended Hardware & Tools Setup

### Development Machine

```
Minimum Viable Setup:
  - Modern laptop with 16+ GB RAM
  - Google Colab Pro ($10/month) for GPU access
  - Lambda Cloud or RunPod for larger experiments

Ideal Home Setup:
  - Desktop with NVIDIA RTX 4090 (24GB VRAM) or RTX 3090
  - 64+ GB RAM
  - 2+ TB NVMe SSD
  - Ubuntu 22.04 LTS

For Serious Training:
  - Cloud instances: AWS p4d/p5, GCP A100/H100, Lambda Labs
  - Consider university/company GPU clusters
```

### Essential Software Stack

```
Languages:       Python, C++ (for understanding internals)
ML Frameworks:   PyTorch (primary), JAX (secondary)
Data:            NumPy, Pandas, Polars
Visualization:   Matplotlib, Seaborn, Weights & Biases
Experiment:      W&B, MLflow, or TensorBoard
Notebooks:       Jupyter Lab, VS Code with Jupyter extension
IDE:             VS Code or Cursor with Copilot
Version Control: Git + GitHub
Containers:      Docker
Cloud:           AWS/GCP/Azure basics
```

---

## 16. Key Conferences, Journals & Communities

### Top Conferences (ranked by ML impact)
1. **NeurIPS** (Neural Information Processing Systems) -- December
2. **ICML** (International Conference on Machine Learning) -- July
3. **ICLR** (International Conference on Learning Representations) -- May
4. **CVPR** (Computer Vision and Pattern Recognition) -- June
5. **ACL** (Association for Computational Linguistics) -- July
6. **EMNLP** (Empirical Methods in NLP) -- October/November
7. **AAAI** (Association for the Advancement of AI) -- February
8. **ICCV** (International Conference on Computer Vision) -- October (biannual)

### Online Communities
- **Twitter/X**: Follow top researchers (#MachineLearning, #NLProc, #ComputerVision)
- **Reddit**: r/MachineLearning, r/deeplearning, r/LocalLLaMA
- **Discord**: EleutherAI, Hugging Face, PyTorch
- **Hacker News**: AI-related discussions
- **Papers With Code**: Track SOTA across tasks

### Key People to Follow
- **Andrej Karpathy** (education, LLMs)
- **Yann LeCun** (Meta AI, fundamental research)
- **Ilya Sutskever** (Safe Superintelligence)
- **Geoffrey Hinton** (godfather of deep learning)
- **Fei-Fei Li** (Stanford, computer vision)
- **Yoshua Bengio** (MILA, deep learning theory)
- **Jason Wei** (OpenAI, chain-of-thought)
- **Tri Dao** (FlashAttention, Together AI)
- **Sasha Rush** (Cornell, efficient methods)

---

## 17. Common Pitfalls to Avoid

### 1. Tutorial Hell
**Problem**: Watching course after course without building anything.
**Solution**: Follow the 70/30 rule -- 70% building, 30% learning.

### 2. Skipping Math
**Problem**: Jumping straight to deep learning frameworks without understanding the math.
**Solution**: You cannot innovate if you don't understand *why* things work. Invest in math.

### 3. Only Using High-Level APIs
**Problem**: Using `model.fit()` without understanding what happens underneath.
**Solution**: Implement everything from scratch at least once.

### 4. Chasing Every New Paper
**Problem**: FOMO about the latest arxiv papers without deep understanding of fundamentals.
**Solution**: Master the fundamentals first. New papers will make much more sense.

### 5. Not Building a Portfolio
**Problem**: Learning in isolation without any public proof of your skills.
**Solution**: Push code to GitHub, write blog posts, build in public from day one.

### 6. Ignoring Software Engineering
**Problem**: Writing messy, untested, undocumented code.
**Solution**: Clean code, proper testing, documentation -- these matter in industry.

### 7. Not Reading Papers
**Problem**: Relying only on courses and tutorials.
**Solution**: Start reading papers from Month 6. There is no substitute.

### 8. Comparing Your Progress to Others
**Problem**: Feeling behind because someone on Twitter trained a model in a week.
**Solution**: Focus on your own path. Depth takes time. Consistency wins.

---

## 18. Final Words

The path to becoming an AI expert is long and demanding, but it is deeply rewarding. The people you admire -- Alexandr Wang, Andrew Tulloch, Ruoming Pang -- all share one thing: they put in thousands of hours of deep, focused work.

**Key takeaways**:

1. **Start now**. The best time to begin was yesterday. The second best time is today.
2. **Build foundations**. Math and CS fundamentals are the bedrock everything else rests on.
3. **Implement everything**. Understanding comes from building, not just reading.
4. **Be patient**. This is a marathon, not a sprint. Give yourself 3-5 years.
5. **Stay curious**. The best researchers are driven by genuine curiosity, not just career goals.
6. **Contribute**. Give back to the community through open source, blog posts, and mentoring.
7. **Think big**. AI is transforming every industry. The problems worth solving are enormous.

> *"The people who are crazy enough to think they can change the world are the ones who do."*

**You have the passion. Now channel it into disciplined, consistent action. The world needs more great AI scientists.**

---

## Appendix: Complete Reading List (Priority Order)

### Textbooks (Must-Read)
1. "Deep Learning" -- Goodfellow, Bengio, Courville
2. "Pattern Recognition and Machine Learning" -- Bishop
3. "Linear Algebra Done Right" -- Axler
4. "Convex Optimization" -- Boyd & Vandenberghe
5. "The Elements of Statistical Learning" -- Hastie, Tibshirani, Friedman
6. "Reinforcement Learning: An Introduction" -- Sutton & Barto
7. "Information Theory, Inference and Learning Algorithms" -- MacKay
8. "Speech and Language Processing" -- Jurafsky & Martin
9. "Computer Vision: Algorithms and Applications" -- Szeliski

### Online Courses (Must-Take)
1. Stanford CS229 -- Machine Learning (Andrew Ng)
2. Stanford CS231n -- CNNs for Visual Recognition
3. Stanford CS224n -- NLP with Deep Learning
4. MIT 18.06 -- Linear Algebra (Gilbert Strang)
5. Fast.ai -- Practical Deep Learning
6. Andrej Karpathy -- Neural Networks: Zero to Hero
7. Stanford CS25 -- Transformers United
8. DeepMind/UCL -- Reinforcement Learning
9. Stanford CS324 -- Large Language Models

---

*Last updated: February 2026*
*This is a living document. Update it as the field evolves.*

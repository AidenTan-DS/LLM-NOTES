# 📚 Large Language Models - Complete Learning Notes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![LaTeX](https://img.shields.io/badge/Made%20with-LaTeX-1f425f.svg)](https://www.latex-project.org/)

> **Complete Learning Guide: From Transformer to LangChain**
>
> Week 1: Fundamentals → Week 2: Fine-tuning → Week 3: Alignment → Week 4: Applications
>
> *By Aiden Tan*

---

## 📥 Quick Download

| Version | File | Description |
|---------|------|-------------|
| English | [`llm_notes_english.pdf`](./llm_notes_english.pdf) | Full PDF with watermark |


---

## 📖 Table of Contents

- [Course Overview](#course-overview)
- [Week 1: Transformer Architecture \& Pre-training](#week-1-transformer-architecture--pre-training-fundamentals)
- [Week 2: Instruction Fine-tuning \& LoRA](#week-2-instruction-fine-tuning--parameter-efficient-methods)
- [Week 3: RLHF - Aligning with Human Preferences](#week-3-rlhf--aligning-models-with-human-preferences)
- [Week 4: Model Optimization, RAG \& LangChain](#week-4-model-optimization-rag--llm-application-frameworks)
- [Final Summary](#course-summary)

---

## Course Overview

| Week 1 | Week 2 | Week 3 | Week 4 |
|--------|--------|--------|--------|
| 🔵 Transformer | 🟢 Instruction Fine-tuning | 🔴 RLHF | 🟠 Model Optimization |
| 🔵 Model Types | 🟢 Full SFT vs LoRA | 🔴 Reward Model | 🟠 RAG |
| 🔵 Pre-training | 🟢 Evaluation & Benchmarks | 🔴 PPO + KL Penalty | 🟠 CoT / PAL |
| 🔵 Quantization + ICL | 🟢 PEFT Techniques | 🔴 Preference Alignment | 🟠 ReAct / LangChain |

> 💡 **Who is this for?** If you're new to LLMs, this note will help you understand each concept with plenty of examples.

---

# Week 1: Transformer Architecture & Pre-training Fundamentals

> **Goal:** Understand the backbone of LLMs — the Transformer architecture, and design principles of different model types.

## 1.1 Transformer Architecture

### What is Transformer?

> Transformer is a neural network architecture proposed by Google in 2017, serving as the foundation for all modern LLMs. Its core innovation is the **Self-Attention mechanism**, which allows the model to "see" all positions in the input sequence simultaneously, rather than processing sequentially like RNNs.

### Core Components

| Component | Function |
|-----------|----------|
| **Self-Attention** | Computes relevance between each token and all other tokens |
| **Multi-Head Attention** | Multiple attention heads in parallel, capturing different relationships |
| **Feed-Forward Network** | Applies non-linear transformation to each position independently |
| **Layer Normalization** | Stabilizes training and accelerates convergence |
| **Positional Encoding** | Injects position information (attention is position-agnostic) |

### 📝 Example: Intuitive Understanding of Self-Attention

Input sentence: `"The cat sat on the mat"`

When processing `"sat"`, Self-Attention computes its relevance with every word:
- `"cat"` → High relevance (who sat? the cat)
- `"mat"` → Medium relevance (where? on the mat)
- `"The"` → Low relevance (article, less important)

This enables the model to understand semantic relationships between words.

### Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I have that can be matched?"
- **V (Value):** "What content to return when matched?"
- **√d_k:** Scaling factor to prevent large dot products

---

## 1.2 Three Model Architecture Types

| Type | Structure | Best For | Examples |
|------|-----------|----------|----------|
| **Encoder-Only** | Encoder only | Understanding: classification, NER, sentiment | BERT, RoBERTa |
| **Decoder-Only** | Decoder only | Generation: text generation, dialogue | GPT series, LLaMA |
| **Encoder-Decoder** | Both | Seq2Seq: translation, summarization | T5, BART, FLAN-T5 |

### Encoder-Only Models

**Key Characteristics:**
- Uses **Bidirectional Attention**: each token sees all tokens (left and right)
- Excels at "understanding" tasks, not generation
- Pre-training: Masked Language Modeling (MLM)

**📝 Example: BERT's MLM Pre-training**
```
Input:  "The [MASK] sat on the mat"
Task:   Predict what [MASK] is
Answer: "cat"
```

### Decoder-Only Models

**Key Characteristics:**
- Uses **Causal/Masked Attention**: each token only sees tokens to its left
- Excels at "generation" tasks: predicts next token sequentially
- Pre-training: Next Token Prediction

**📝 Example: GPT's Autoregressive Generation**
```
Prompt: "The weather today is"

Generation (token by token):
1. "The weather today is" → "sunny"
2. "The weather today is sunny" → "and"
3. "The weather today is sunny and" → "warm"
...continues until end token
```

### Encoder-Decoder Models

**Key Characteristics:**
- Encoder "understands" input, Decoder "generates" output
- Connected via Cross-Attention
- Best for tasks where input/output formats differ

> 💡 **What is FLAN-T5?**
> 
> FLAN-T5 = T5 model + Instruction Fine-tuning
> 
> FLAN stands for "Fine-tuned Language Net" — Google fine-tuned T5 on massive instruction data, making it better at following instructions.

---

## 1.3 Pre-training

> Pre-training uses **large-scale unlabeled text** to train the model. The goal is to learn fundamental language patterns: grammar, semantics, world knowledge. After pre-training, the model has general language abilities but doesn't know how to "follow instructions."

### Pre-training Objectives

| Model Type | Objective | Method |
|------------|-----------|--------|
| Encoder-Only | MLM | Randomly mask 15% of tokens, predict them |
| Decoder-Only | Next Token | Given prefix, predict the next token |
| Encoder-Decoder | Span Corruption | Mask contiguous spans, recover them |

**📝 Example: T5's Span Corruption**
```
Original:        "The quick brown fox jumps over the lazy dog"
Corrupted Input: "The <X> fox <Y> the lazy dog"
Target Output:   "<X> quick brown <Y> jumps over"
```

---

## 1.4 Model Quantization

> Large models have billions of parameters (GPT-3 has 175B). Using FP32 directly means huge memory (700GB), slow inference, and high costs. **Quantization** reduces numerical precision to compress models and speed up computation.

### Precision Comparison

| Precision | Size per Param | Range | Use Case |
|-----------|---------------|-------|----------|
| FP32 (Single) | 4 bytes | High precision | Training (default) |
| FP16 (Half) | 2 bytes | Medium | Training/Inference |
| BF16 | 2 bytes | Wider range | Training (stable) |
| INT8 | 1 byte | Integer, limited | Inference deployment |
| INT4 | 0.5 bytes | Very limited | Extreme compression |

**📝 Example: LLaMA-7B**

| Precision | Model Size | Memory Required |
|-----------|-----------|-----------------|
| FP32 | 28 GB | ~32 GB |
| FP16 | 14 GB | ~16 GB |
| INT8 | 7 GB | ~8 GB |
| INT4 | 3.5 GB | ~4 GB |

> From FP32 to INT4, model size reduces by **8x**!

---

## 1.5 In-Context Learning (ICL)

> **In-Context Learning** allows the model to perform tasks **without updating parameters** — just by providing task descriptions and examples in the prompt. You don't need training, just "tell" the model what to do.

### Three Modes

| Mode | Definition | Prompt Example |
|------|------------|----------------|
| **Zero-shot** | Instruction only, no examples | `"Translate to French: Hello"` |
| **One-shot** | 1 example provided | `"English: Hello → French: Bonjour. English: Goodbye → French: ?"` |
| **Few-shot** | 3-10 examples provided | Multiple input-output pairs |

**📝 Example: Few-shot Sentiment Classification**
```
Classify the sentiment as positive or negative.

Review: "This movie was amazing!" → Sentiment: positive
Review: "Terrible waste of time." → Sentiment: negative  
Review: "Best purchase I ever made!" → Sentiment: positive

Review: "I regret buying this product." → Sentiment: ___
```
**Model Output:** `negative`

### ICL Pros and Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| No training required, zero cost | Sensitive to prompt format |
| Fast iteration (just modify prompt) | Limited by context length |
| Great for rapid prototyping | Example order can affect results |
| No data pipeline needed | Usually worse than fine-tuning |

---

## 📌 Week 1 Key Takeaways

1. **Transformer** is the foundation of all LLMs, with Self-Attention as its core
2. **Three architectures**: Encoder-Only (understanding), Decoder-Only (generation), Encoder-Decoder (seq2seq)
3. **Pre-training** teaches general language abilities, but not instruction-following
4. **Quantization** (FP16/INT8) significantly reduces computational costs
5. **ICL** enables task completion via prompt examples, no training required

---

# Week 2: Instruction Fine-tuning & Parameter-Efficient Methods

> **Goal:** Learn how to make models "follow instructions" — from temporary ICL adaptation to actually changing model behavior through fine-tuning.

## 2.1 Why Fine-tuning?

> While ICL is convenient, it has fundamental limitations:
> - Parameters unchanged — behavior change is "temporary"
> - Must repeat examples every inference, wasting tokens
> - Complex tasks perform worse than fine-tuning
> - Cannot make the model "remember" certain behavior patterns
>
> **Fine-tuning updates model parameters, making behavior changes "permanent."**

## 2.2 Instruction Fine-tuning (SFT)

> **Instruction Fine-tuning** (also called Supervised Fine-tuning, SFT) trains the model with "instruction-response" pairs, teaching it to understand various instruction formats, generate appropriate responses, and refuse inappropriate requests. This is the key step transforming a pre-trained model into an "assistant."

### Training Objective

$$\max_\theta \sum_{(x,y) \in D} \log \pi_\theta(y \mid x)$$

- **x**: instruction + input
- **y**: expected output (ground truth)
- **π_θ**: model's output probability distribution
- **Goal**: maximize probability of generating correct answers

---

## 2.3 Full Fine-tuning vs PEFT (LoRA)

### Full Fine-tuning

Updates **all** (or most) model parameters.

| ✅ Pros | ❌ Cons |
|---------|---------|
| Best performance | High compute cost |
| High data distribution fit | Prone to overfitting |
| Can learn complex patterns | **Catastrophic Forgetting** |
| | Need to store full model per task |

> ⚠️ **Catastrophic Forgetting:** When training on a new task, the model may "forget" general abilities learned during pre-training. Example: After fine-tuning for summarization, the model might become worse at translation.

### PEFT: Parameter-Efficient Fine-Tuning

> Don't update all parameters — train only a **small number of extra parameters** to achieve near full fine-tuning performance.

**Benefits:**
- Dramatically reduced memory requirements
- Faster training
- Can save different small adapters for different tasks
- Reduced catastrophic forgetting

### LoRA: Low-Rank Adaptation

> LoRA's key insight: weight changes during task adaptation are **low-rank**. We can approximate weight changes with the product of two small matrices.

$$W' = W + \Delta W = W + BA$$

- **W ∈ ℝ^(d×d)**: Original weights (frozen, not updated)
- **B ∈ ℝ^(d×r)**, **A ∈ ℝ^(r×d)**: LoRA matrices (trainable)
- **r ≪ d**: rank, typically 4, 8, 16, or 32
- Parameters reduce from d² to 2dr

**📝 Example: LoRA Parameter Calculation**

Original attention weight W_Q ∈ ℝ^(4096×4096):

| Method | Trainable Parameters |
|--------|---------------------|
| Full Fine-tuning | 4096 × 4096 = **16.8M** |
| LoRA (rank=8) | (8 × 4096) + (4096 × 8) = **65K** |

> **256x reduction!**

### Full vs LoRA Comparison

| Aspect | Full Fine-tuning | LoRA |
|--------|-----------------|------|
| Trainable Parameters | 100% | 0.1% - 1% |
| GPU Memory | High | Low |
| Training Speed | Slow | Fast |
| Multi-task Deployment | Store multiple full models | Same backbone + multiple adapters |
| Performance | Best | Close (usually minor gap) |
| Catastrophic Forgetting | Severe | Mild |

---

## 2.4 Evaluation & Benchmarks

### BLEU (Bilingual Evaluation Understudy)

> **Core Idea:** How many n-grams in generated text appear in reference (**Precision**)

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

- Commonly used for **machine translation**
- BP = Brevity Penalty (penalizes short outputs)

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

> **Core Idea:** How many n-grams in reference are covered by generated text (**Recall**)

| Variant | Description |
|---------|-------------|
| ROUGE-1 | Unigram recall |
| ROUGE-2 | Bigram recall |
| ROUGE-L | Based on Longest Common Subsequence (LCS) |

Commonly used for **summarization**.

> ⚠️ **Limitations of Overlap-based Metrics:**
> - Unfriendly to paraphrases (same meaning, different words = low score)
> - Don't measure safety, helpfulness, truthfulness
> - May not align with human preferences
>
> This is why we need **RLHF** in Week 3!

---

## 📌 Week 2 Key Takeaways

1. **Instruction Fine-tuning** teaches models to follow instructions
2. **Full Fine-tuning** performs best but is expensive with forgetting risk
3. **LoRA** uses low-rank decomposition, training only 0.1%-1% parameters
4. **BLEU** measures precision, **ROUGE** measures recall
5. Traditional metrics have limitations, can't measure safety and preferences

---

# Week 3: RLHF — Aligning Models with Human Preferences

> **Goal:** Understand how to use reinforcement learning to make models not just "correct" but "good" — safe, helpful, and aligned with human values.

## 3.1 Why RLHF?

> Instruction Fine-tuning makes models "follow instructions," but has issues:
> - Ground truth answers aren't necessarily the "best" answers
> - Can't express "this response is better than that" preferences
> - May generate harmful, unsafe content
> - May "confidently hallucinate" (make things up)
>
> **RLHF introduces human preferences, teaching models to generate what humans actually want.**

**📝 Example: SFT vs RLHF**

**Question:** `"How do I pick a lock?"`

| Model | Response |
|-------|----------|
| **SFT** | "First, you need a tension wrench and pick. Insert the tension wrench..." *(Technically correct, but could enable illegal activity)* |
| **RLHF** | "I can't provide instructions for picking locks as it could enable illegal entry. If you're locked out, I recommend calling a licensed locksmith." *(Safer, more socially appropriate)* |

---

## 3.2 RLHF Training Pipeline

| Stage | Name | What It Does |
|-------|------|--------------|
| 1 | Supervised Fine-tuning | Train with ground truth for basic instruction-following |
| 2 | Reward Model Training | Collect human preferences, train a "scorer" |
| 3 | RL Fine-tuning (PPO) | Use Reward Model scores as rewards to optimize policy |

---

## 3.3 Reward Model

> Reward Model is a "judge" that scores model outputs:
> - **Input:** prompt + response
> - **Output:** scalar score r(x, y) ∈ ℝ
> - Higher score = response better aligns with human preferences

### Training Data: Preference Pairs

**📝 Example: Preference Data Collection**

**Prompt:** `"Explain quantum computing to a child."`

| Response | Content |
|----------|---------|
| A | "Quantum computing uses qubits that can be 0 and 1 simultaneously through superposition, enabling parallel computation..." |
| B | "Imagine a magic coin that can be heads AND tails at the same time! Quantum computers use these special coins to solve puzzles really fast." |

**Human Label:** B > A *(B is better for explaining to a child)*

### Reward Model Training Objective

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( r(x, y_w) - r(x, y_l) \right) \right]$$

- **(x, y_w, y_l)**: prompt, preferred response, less preferred response
- **Goal**: make r(x, y_w) > r(x, y_l)

---

## 3.4 PPO: Proximal Policy Optimization

> PPO is a stable policy gradient algorithm:
> - Clipped objective prevents overly large updates
> - More stable than vanilla policy gradient
> - Widely used in RLHF

### RLHF Optimization Objective

$$\max_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) - \beta \cdot \text{KL}\left( \pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x) \right) \right]$$

- **r(x, y)**: Reward Model's score
- **KL(·‖·)**: KL divergence, measures distribution difference
- **π_ref**: Reference model (usually SFT model, frozen)
- **β**: KL penalty coefficient

### Core Components

| Component | Symbol | Role |
|-----------|--------|------|
| Policy Model | π_θ | Model being optimized, generates responses |
| Reference Model | π_ref | Frozen, provides KL anchor |
| Reward Model | r(x,y) | Scores responses |
| Value Head | V(s) | Estimates state value, computes advantage |

---

## 3.5 Importance of KL Penalty

> Without KL constraint, the model might:
> - **Reward Hacking**: Find ways to "cheat" the Reward Model for high scores
> - **Mode Collapse**: Only generate certain fixed patterns
> - **Distribution Shift**: Drift away from pre-trained language abilities
>
> KL Penalty forces the model to stay close to the reference model.

**📝 Example: Reward Hacking**

**Scenario:** Reward Model gives high scores for "polite" responses

**Without KL constraint:** Model learns to add "Thank you so much for asking! I really appreciate your question!" to every response. Gets high reward but isn't helpful.

**With KL constraint:** Model is constrained to normal language patterns.

---

## 📌 Week 3 Key Takeaways

1. **RLHF** optimizes using human preferences (not ground truth)
2. **Reward Model** is the "judge" that scores responses
3. **PPO** is a stable policy optimization algorithm using Advantage to reduce variance
4. **KL Penalty** prevents reward hacking and distribution shift
5. RLHF makes models not just "correct" but "good, safe, helpful"

---

# Week 4: Model Optimization, RAG & LLM Application Frameworks

> **Goal:** Learn how to make models smaller and faster (optimization), provide new knowledge without retraining (RAG), solve complex problems (reasoning techniques), and connect to external tools (Orchestration).

## 4.1 Model Inference Optimization

> Trained large models are often too big, too slow, and too expensive. We need to make models smaller and faster **without significantly sacrificing performance**.

### Three Optimization Techniques

| Technique | Core Idea | Pros | Cons |
|-----------|-----------|------|------|
| **Quantization** | Reduce numerical precision | Simple, effective | Extreme quant. hurts |
| **Pruning** | Remove unimportant params | High compression | Needs retraining |
| **Distillation** | Small model learns from large | Good small model perf. | Requires training |

### Pruning

> Many neural network parameters contribute little to the output (close to 0). Pruning **removes these unimportant parameters** to compress the model.

| Pruning Type | Description |
|--------------|-------------|
| **Unstructured** | Remove individual weights (sparse matrix), high compression but hard to accelerate |
| **Structured** | Remove entire neurons/channels/layers, hardware-friendly but lower compression |

### Knowledge Distillation

> Train a **small model (Student)** to mimic the behavior of a **large model (Teacher)**. Teacher's soft labels contain richer information than hard labels.

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(y, \hat{y}_{student}) + (1-\alpha) \cdot \mathcal{L}_{KL}(p_{teacher}, p_{student})$$

**📝 Example: DistilBERT**
- Teacher: BERT-base (110M parameters)
- Student: DistilBERT (66M parameters, **40% reduction**)
- Performance: Retains **97%** of BERT's performance
- Speed: **60% faster**

---

## 4.2 RAG: Retrieval-Augmented Generation

> LLM knowledge has limitations:
> - **Knowledge cutoff**: Training data has a cutoff date
> - **Private data**: Doesn't know your company's internal documents
> - **Hallucination**: May fabricate information
> - **Update cost**: Retraining is too expensive
>
> **RAG retrieves relevant documents at inference time, providing LLM with latest/private knowledge without retraining!**

### RAG Workflow

```
Step 1: User Query
        "What is our company's vacation policy?"
                    ↓
Step 2: Retrieve Relevant Documents
        Find most relevant chunks from knowledge base
                    ↓
Step 3: Build Augmented Prompt
        Context: [Retrieved document content]
        Question: What is our company's vacation policy?
                    ↓
Step 4: LLM Generates Answer
        Generate accurate answer based on retrieved context
```

### RAG Core Components

| Component | Role |
|-----------|------|
| **Document Loader** | Load various document formats (PDF, Word, web, etc.) |
| **Text Splitter** | Split long documents into small chunks |
| **Embedding Model** | Convert text into vector representations |
| **Vector Store** | Store and retrieve vectors (FAISS, Pinecone, etc.) |
| **Retriever** | Find most relevant chunks for a query |
| **LLM** | Generate final answer based on retrieved content |

---

## 4.3 Advanced Reasoning Techniques

### Chain of Thought (CoT)

> Make the model **show its reasoning steps** instead of jumping to the answer. Like a teacher asking students to "show your work."

**📝 Example: Standard vs CoT**

**Problem:** Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many balls now?

**Standard Prompting:**
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many balls now?
A: 11
```
*(Model might get it wrong)*

**Chain of Thought:**
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many balls now?
A: Let's think step by step.
   Roger started with 5 balls.
   He bought 2 cans, each with 3 balls.
   So he bought 2 × 3 = 6 balls.
   Total = 5 + 6 = 11 balls.
   The answer is 11.
```

> 💡 **Zero-shot CoT:** Just add **"Let's think step by step."** at the end of your prompt!

### PAL: Program-Aided Language Models

> LLMs are good at understanding problems and writing code, but bad at precise calculation.
>
> **PAL's solution:**
> - LLM handles: Understanding problem → Generating Python code
> - Python handles: Executing code → Returning precise answer

**📝 Example:**

**Problem:** A store sells apples for $0.75 each. If you buy 12 apples and pay with a $20 bill, how much change?

**LLM generates:**
```python
apple_price = 0.75
num_apples = 12
payment = 20

total_cost = apple_price * num_apples
change = payment - total_cost
print(change)  # Output: 11.0
```

---

## 4.4 Orchestration: LLM as Reasoning Engine

> In modern LLM applications, the LLM serves as a **Reasoning Engine**, coordinated by an Orchestration layer:

```
User Input → Prompt → LLM → Orchestration → Action/API Call
```

- **LLM**: Understands intent, reasons, generates instructions/code
- **Orchestration**: Parses LLM output, executes actual actions

### ReAct: Reasoning + Acting

> ReAct combines Reasoning and Acting, making the LLM alternate between:
> - **Thought**: Think about what to do next
> - **Action**: Execute an action (call a tool)
> - **Observation**: Observe the action's result

**📝 Example:**

**Question:** What is the population of the capital of France?

```
Thought 1: I need to find the capital of France first.
Action 1: Search["capital of France"]
Observation 1: The capital of France is Paris.

Thought 2: Now I need to find the population of Paris.
Action 2: Search["population of Paris"]
Observation 2: The population of Paris is approximately 2.1 million.

Thought 3: I have the answer now.
Action 3: Finish["The population of Paris is approximately 2.1 million."]
```

### LangChain: Packaging It All Together

> LangChain is an open-source framework that packages all these techniques:

| LangChain Component | Corresponding Concept |
|--------------------|----------------------|
| `LLMChain` | Basic Prompt → LLM → Output |
| `RetrievalQA` | RAG implementation |
| `Agent + Tools` | ReAct + Orchestration |
| `PythonREPL Tool` | PAL (let LLM execute Python) |
| `ConversationBufferMemory` | Conversation history management |

---

## 📌 Week 4 Key Takeaways

1. **Optimization trio**: Quantization (reduce precision), Pruning (remove params), Distillation (train smaller model)
2. **RAG** lets LLMs access latest/private knowledge without retraining
3. **Chain of Thought** makes models show reasoning steps, improving accuracy
4. **PAL** lets LLM write code, Python executes for precise computation
5. **Orchestration** uses LLM as reasoning engine, coordinates tool execution
6. **ReAct** = Thought + Action + Observation loop
7. **LangChain** packages these techniques into an easy-to-use framework

---

# Course Summary

## Complete LLM Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 0: Pre-training                                       │
│  Large-scale unlabeled text → Learn general language         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: In-Context Learning (Optional)                     │
│  Temporarily adapt via prompt + examples (no training)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Instruction Fine-tuning (SFT)                      │
│  Train with instruction-response data (Full or LoRA)         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: RLHF                                               │
│  Preference alignment with Reward Model + PPO                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Deployment & Applications                          │
│  Optimization | RAG | CoT/PAL | ReAct/LangChain              │
└─────────────────────────────────────────────────────────────┘
```

## One-Sentence Summary for Each Concept

### Week 1 - Fundamentals
| Concept | Summary |
|---------|---------|
| Transformer | Backbone of LLMs, Self-Attention is its core |
| Encoder/Decoder | Understanding vs Generation, different architectures for different tasks |
| Pre-training | Learn general language abilities from massive text |
| ICL | No training, teach model via prompt examples |

### Week 2 - Fine-tuning
| Concept | Summary |
|---------|---------|
| SFT | Train with ground truth, teach model to follow instructions |
| LoRA | Train only low-rank adapters, efficient and cheap |
| BLEU/ROUGE | Evaluation metrics based on n-gram overlap |

### Week 3 - Alignment
| Concept | Summary |
|---------|---------|
| RLHF | Optimize using human preferences instead of ground truth |
| Reward Model | The "judge" that scores responses |
| PPO | Stable policy optimization algorithm |
| KL Penalty | Prevents model from "cheating" or drifting too far |

### Week 4 - Optimization & Applications
| Concept | Summary |
|---------|---------|
| Quantization | Reduce precision, shrink model size |
| Pruning | Remove unimportant parameters |
| Distillation | Small model learns from large model |
| RAG | Retrieval-augmented generation, access new knowledge |
| CoT | Show reasoning steps |
| PAL | LLM writes code, Python executes |
| ReAct | Thought-Action-Observation loop |
| LangChain | Framework packaging all these techniques |

---

## 🎯 After This Course, You Should Be Able To...

1. ✅ Explain how Transformer and Self-Attention work
2. ✅ Distinguish Encoder-Only, Decoder-Only, and Encoder-Decoder models
3. ✅ Understand the roles and differences of Pre-training, ICL, SFT, and RLHF
4. ✅ Implement LoRA fine-tuning and understand its advantages
5. ✅ Explain how RLHF makes models safer and more helpful
6. ✅ Use BLEU, ROUGE, and other metrics for evaluation
7. ✅ Use Quantization, Pruning, and Distillation to optimize models
8. ✅ Build RAG systems to give models external knowledge
9. ✅ Use CoT and PAL to improve model reasoning
10. ✅ Understand ReAct framework and LangChain applications

---



⭐ **If you find this helpful, please give it a star!**

# Option-3-Generative-AI [Togo-AI-Labs]

## 📌 Overview

This project reproduces and extends the core idea of *“Language Models are Few-Shot Learners”* (Brown et al., NeurIPS 2020). The original paper demonstrated that large-scale models like GPT-3 can perform well in few-shot settings without fine-tuning.

Since GPT-3 scale training is infeasible in constrained environments, this project uses **GPT-2** as a proxy to explore:

* **Few-Shot prompting** (baseline GPT-2 without fine-tuning).
* **Full fine-tuning** of GPT-2 on a conversational dataset.
* **LoRA-based parameter-efficient fine-tuning** (PEFT), which adapts fewer parameters but aims for comparable performance.

The task is **conversational response generation** on a dataset of Reddit-style dialogues.

---

## ⚙️ Project Setup

### Environment

* Google Colab / Local PyTorch
* **Libraries**:

  * `torch==2.4.1`
  * `transformers==4.44.2`
  * `peft==0.13.2`
  * `datasets`, `evaluate`, `nltk`, `pandas`, `sentence-transformers`

### Installation

```bash
!pip uninstall -y torch torchvision torchaudio
!pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
!pip install transformers==4.44.2 datasets evaluate nltk pandas peft==0.13.2 sentence-transformers
```

---

## 📂 Dataset

* Source: [Kaggle Reddit Conversations](https://www.kaggle.com/datasets/jerryqu/reddit-conversations)
* Format: CSV file has 3 columns, remove 3rd column entirely as it is not needed. Title 1st column as `question` and 2nd as `answer`.
* Processing:

  * Conversations structured as:

    ```
    <start> Human: [Q] <turn> Bot: [A] <turn> ... <end>
    ```
  * Tokenized with GPT-2 tokenizer + special tokens (`<start>`, `<end>`, `<turn>`).
  * Filtered to minimum **6 turns** per conversation.
  * \~1000 samples after preprocessing.

---

## 🚀 Experiments

### 1. Few-Shot Baseline (Vanilla GPT-2)

* Handcrafted exemplar conversation prompts.
* Tested on 5 fixed queries:

  1. *Hi!!!!! It's nice to see you again!*
  2. *Looking for interesting podcasts*
  3. *what do you usually do for fun?*
  4. *First time with wifi on a plane and oh god is it glorious*
  5. *I've always wanted to go to England.*

⚠️ Outputs were **often incoherent/gibberish**, showing limitations of GPT-2 without fine-tuning.

---

### 2. Full Fine-Tuning (GPT-2)

* Optimizer: **AdamW** (lr=5e-6).
* Schedule: Linear warmup (10 steps).
* Batch size: 4, Max length: 512.
* Epochs: 5.

✅ Results: More coherent and relevant than few-shot, but still simplistic.

---

### 3. LoRA Fine-Tuning (PEFT)

* Config:

  * Rank `r=32`, Alpha=64, Dropout=0.05.
  * Applied on attention layers (`c_attn`, `c_proj`).
* Optimizer: AdamW (lr=2e-4).
* Epochs: 10.

✅ Results: Comparable to full fine-tuning, more efficient. Sometimes quirky.

---

## 📊 Evaluation

### Metrics

* **Perplexity** (fluency).
* **BLEU** (relevance to reference response).

| Model                 | Perplexity ↓ | BLEU ↑ |
| --------------------- | ------------ | ------ |
| Few-Shot GPT-2        | 40.57        | 0.115  |
| Full Fine-Tuned GPT-2 | 47.13        | 0.119  |
| LoRA Fine-Tuned GPT-2 | 95.82        | 0.199  |

> LoRA achieves higher BLEU despite worse perplexity, suggesting **more relevant but noisier outputs**.

---

## 🧩 Creative Extension

* **LoRA for efficiency**: Updates <1% parameters, aligning with modern PEFT trends.
* **LLM-as-a-judge**: Used 7 LLMs (GPT-4o-mini, Grok-4, Gemini, Claude, Qwen, DeepSeek, Kimi-K2) to rank models on 9 conversational metrics (coherence, creativity, safety, efficiency, etc.). 
  <img width="732" height="495" alt="1 gpt4" src="https://github.com/user-attachments/assets/d369424d-5a0f-4b8b-82da-cba8e7f2277c" />
 <img width="737" height="491" alt="2 Gemi" src="https://github.com/user-attachments/assets/d018ad70-4fe4-4853-831b-8833afe439a5" />
 <img width="731" height="487" alt="3 claude-4" src="https://github.com/user-attachments/assets/f042c1c1-3c53-47f6-83c1-cd6f21ea5311" />
 <img width="746" height="497" alt="Screenshot 2025-09-21 134109" src="https://github.com/user-attachments/assets/b24f0c8c-27bf-470a-8568-936125fbefc5" />
 <img width="641" height="432" alt="5 deep" src="https://github.com/user-attachments/assets/8f91f1e6-2256-4bb5-901f-94d6cad63c97" />
 <img width="641" height="427" alt="6 grok" src="https://github.com/user-attachments/assets/7c07991e-1b1b-44a4-8333-24cf539fce96" />
 <img width="641" height="430" alt="7 qwen" src="https://github.com/user-attachments/assets/7a827ac3-03ab-4149-9105-492328eccc34" />

* Results shows:

  * Full fine-tuning best for **relevance & fluency**.
  * LoRA preferred for **creativity & diversity**.
  * Few-shot excelled in **efficiency/safety**.


---

## 📌 Key Insights

1. **Scaling works**: Larger models (GPT-3) benefit more from few-shot than smaller ones (GPT-2).
2. **Fine-tuning improves coherence** on small models, but risks overfitting on limited datasets.
3. **LoRA is efficient**: Provides a strong tradeoff between compute cost and quality.
4. **Evaluation challenges**: Human/LLM judgments vary, showing subjectivity.
5. **Future Work**:

   * Test on larger datasets.
   * Explore hybrid prompting + LoRA approaches.
   * Address ethical concerns (bias, misuse, toxicity).

---

## 📁 Repository Structure

```
├── dialogs.csv          # Dataset (upload manually in Colab)
├── gpt2_finetune.ipynb  # Main Colab script
├── Report.pdf           # Full project report
└── README.md            # Project documentation
```

---

## 🙌 Acknowledgements

* Brown et al., *“Language Models are Few-Shot Learners”*, NeurIPS 2020.
* Togo AI Lab
* Hugging Face Transformers & PEFT libraries.
* Kaggle Reddit Conversations dataset.




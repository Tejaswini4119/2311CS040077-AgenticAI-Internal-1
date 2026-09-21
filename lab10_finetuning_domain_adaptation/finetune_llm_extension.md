# Extending This Lab to Real LLM Fine-Tuning

`finetune_domain_adaptation.py` demonstrates the fine-tuning **workflow**
(synthetic data generation → base training → domain-gap evaluation →
fine-tuning → after evaluation) using a lightweight TF-IDF + logistic
regression classifier, because this environment has no GPU and no access to
model-weight hosts (only PyPI/npm package registries are reachable).

The exact same workflow structure applies when fine-tuning a real LLM. Here
is how each step maps onto the OpenAI fine-tuning API (Anthropic does not
currently offer public fine-tuning; use RAG/prompt-engineering with Claude,
or OpenAI/open-weight models for true fine-tuning):

## 1. Data preparation
Convert `build_dataset()`'s output into the JSONL format required by the
fine-tuning API:

```jsonl
{"messages": [{"role": "system", "content": "Classify the IT support ticket."}, {"role": "user", "content": "SAP GUI throws a Z-transaction error..."}, {"role": "assistant", "content": "software"}]}
```

## 2. Upload training data
```python
from openai import OpenAI
client = OpenAI()
file = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")
```

## 3. Launch a fine-tuning job (this is the real analogue of `continue_training()`)
```python
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",  # check current fine-tunable models
)
```

## 4. Poll for completion, then evaluate
```python
status = client.fine_tuning.jobs.retrieve(job.id)
# once status.status == "succeeded":
fine_tuned_model_id = status.fine_tuned_model
```
Run the SAME `evaluate()`-style before/after comparison used in this lab:
score the base model and the fine-tuned model on your held-out domain test
set, and report the accuracy delta exactly as `finetune_domain_adaptation.py`
does.

## 5. Use the fine-tuned model
```python
resp = client.chat.completions.create(
    model=fine_tuned_model_id,
    messages=[{"role": "user", "content": "Docking station USB-C hub keeps disconnecting"}],
)
```

## For open-weight models (LoRA / QLoRA)
If you have GPU access, the same base→domain-gap→fine-tune→re-evaluate loop
applies to a local open-weight model (e.g. Llama, Mistral) via `peft` +
`transformers`:

```python
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
model = get_peft_model(base_model, lora_config)
# train on the same combined base+domain dataset built by build_dataset()
```

The measurement discipline (train/val/test split, before/after accuracy,
checking for regression on the general test set) is identical regardless of
which model class you fine-tune - that discipline, not the specific model
API, is the actual point of this lab.

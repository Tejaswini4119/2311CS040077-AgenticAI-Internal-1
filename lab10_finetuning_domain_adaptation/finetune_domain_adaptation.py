"""
Lab 6: Fine-Tuning for Domain Adaptation
==========================================
Demonstrates the full fine-tuning WORKFLOW - data prep, train/val split,
training, before/after evaluation, model persistence - using a lightweight
model that trains in seconds on CPU with no GPU or model download required:
a TF-IDF + logistic-regression text classifier adapted from a generic base
to a specific domain (IT support ticket routing).

Why not fine-tune a transformer/LLM directly?
This sandbox has no GPU and no access to model-weight hosts (only PyPI/npm
registries are reachable), so downloading a pretrained transformer or
calling a hosted fine-tuning API isn't possible here. The workflow shown
below - baseline vs. domain-adapted model, tracked with the same train/val/
test discipline - is exactly the workflow you'd use to fine-tune a real LLM
(e.g. via the OpenAI fine-tuning API or a local LoRA run); only the model
class differs. See `finetune_llm_extension.md` for how to point this same
pipeline at a real hosted fine-tuning API once you have GPU/API access.

Domain task: classify IT support tickets into categories
    {hardware, software, network, account_access, billing}

Pipeline
--------
    1. Generate synthetic labeled tickets (base/general phrasing) for a
       BASE dataset, and a smaller DOMAIN dataset with company-specific
       jargon/abbreviations the base model has never seen.
    2. Train a BASE model on general data only.
    3. Evaluate the base model on domain data -> establishes the "domain
       gap" (expected to perform worse on jargon-heavy domain tickets).
    4. FINE-TUNE (continue training / retrain) on base + domain data.
    5. Evaluate the fine-tuned model on the same domain test set ->
       demonstrates the improvement from domain adaptation.

Run:
    python finetune_domain_adaptation.py
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import print_header

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

random.seed(11)

CATEGORIES = ["hardware", "software", "network", "account_access", "billing"]


# ---------------------------------------------------------------------- #
# Synthetic data generation
# ---------------------------------------------------------------------- #
BASE_TEMPLATES = {
    "hardware": [
        "My laptop screen is cracked and needs replacement.",
        "The office printer is not turning on.",
        "My keyboard keys are sticking and unresponsive.",
        "The monitor shows no signal when I connect it.",
        "My mouse stopped working after the update.",
    ],
    "software": [
        "The application crashes every time I open a large file.",
        "I can't install the latest version of the software.",
        "The program freezes during startup.",
        "I'm getting an error message when saving my document.",
        "The app keeps showing a blank screen after login.",
    ],
    "network": [
        "I can't connect to the office WiFi network.",
        "My internet connection keeps dropping every few minutes.",
        "VPN access is not working from home.",
        "The website loads very slowly on my connection.",
        "I'm unable to reach the shared network drive.",
    ],
    "account_access": [
        "I forgot my password and can't log in.",
        "My account got locked after too many failed attempts.",
        "I need my account permissions updated for a new role.",
        "Multi-factor authentication isn't sending me a code.",
        "I can't access my email account this morning.",
    ],
    "billing": [
        "I was charged twice for the same subscription this month.",
        "My invoice shows an incorrect amount.",
        "I need a refund for a cancelled service.",
        "The billing portal isn't showing my latest payment.",
        "I want to update my payment method on file.",
    ],
}

# Domain-specific jargon a base/general model would NOT have seen -
# simulates a company's internal tooling/abbreviation language.
DOMAIN_TEMPLATES = {
    "hardware": [
        "My ThinkPad's screen has visible bezel cracking, need an RMA ticket.",
        "The Zebra label printer in receiving bay 3 won't power cycle.",
        "Docking station USB-C hub keeps disconnecting my peripherals.",
    ],
    "software": [
        "SAP GUI throws a Z-transaction error code on module load.",
        "The internal CRM tool (SFDC-lite) won't sync with Outlook.",
        "Our ETL pipeline dashboard, Nightingale, is stuck on 'loading'.",
    ],
    "network": [
        "Can't reach the VPC through the corp VPN gateway (gw-03).",
        "The Cisco AnyConnect client fails the SSO handshake on floor 4.",
        "SDWAN tunnel to the Austin branch keeps flapping.",
    ],
    "account_access": [
        "My Okta SSO tile for Workday is grayed out after the SCIM sync.",
        "AD group membership for 'FIN-Analysts' wasn't provisioned on hire.",
        "LDAP bind is failing for my service account after rotation.",
    ],
    "billing": [
        "The Zuora invoice for our Q3 SKU-4471 renewal is duplicated.",
        "NetSuite shows a mismatched AR balance for account ACC-8821.",
        "Our Stripe webhook for invoice.paid isn't updating the ledger.",
    ],
}


def build_dataset(templates: dict, n_per_class: int = 20) -> Tuple[List[str], List[str]]:
    """Expand a small template set into a larger synthetic dataset via light
    paraphrase-style perturbation (prefix/suffix variation) - a common,
    cheap synthetic-data technique when real labeled data is scarce."""
    prefixes = ["", "Hi team, ", "Urgent: ", "Ticket update: ", "FYI - ", "Please help - "]
    suffixes = ["", " Please advise.", " Thanks!", " This is blocking my work.", " Can someone assist?"]

    texts, labels = [], []
    for label, examples in templates.items():
        for _ in range(n_per_class):
            base = random.choice(examples)
            text = random.choice(prefixes) + base + random.choice(suffixes)
            texts.append(text)
            labels.append(label)
    return texts, labels


# ---------------------------------------------------------------------- #
# Model wrapper
# ---------------------------------------------------------------------- #
@dataclass
class TicketClassifier:
    vectorizer: TfidfVectorizer = None
    model: LogisticRegression = None

    def train(self, texts: List[str], labels: List[str]) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)

    def continue_training(self, texts: List[str], labels: List[str]) -> None:
        """
        'Fine-tuning' step: refit on the union of prior + new domain data.
        (Logistic regression doesn't support incremental fine-tuning of a
        frozen base the way a neural net does, so this demonstrates the
        --data-centric-- side of domain adaptation: retraining a model on
        base + domain-specific data. See finetune_llm_extension.md for how
        this maps onto real LLM fine-tuning APIs, which DO support true
        incremental fine-tuning from a base checkpoint.)
        """
        self.train(texts, labels)

    def evaluate(self, texts: List[str], labels: List[str]) -> dict:
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, zero_division=0, output_dict=True)
        return {"accuracy": acc, "report": report, "predictions": list(preds)}

    def save(self, path: str) -> None:
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, path)

    @classmethod
    def load(cls, path: str) -> "TicketClassifier":
        data = joblib.load(path)
        obj = cls()
        obj.vectorizer, obj.model = data["vectorizer"], data["model"]
        return obj


def train_test_split_simple(texts, labels, test_frac=0.3, seed=11):
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    split = int(len(idx) * (1 - test_frac))
    train_idx, test_idx = idx[:split], idx[split:]
    return ([texts[i] for i in train_idx], [labels[i] for i in train_idx],
            [texts[i] for i in test_idx], [labels[i] for i in test_idx])


def main():
    parser = argparse.ArgumentParser(description="Lab 6: Fine-Tuning for Domain Adaptation")
    parser.add_argument("--n_per_class_base", type=int, default=30)
    parser.add_argument("--n_per_class_domain", type=int, default=15)
    parser.add_argument("--model_out", default="domain_adapted_model.joblib")
    args = parser.parse_args()

    print_header("STEP 1 - GENERATING SYNTHETIC DATASETS")
    base_texts, base_labels = build_dataset(BASE_TEMPLATES, args.n_per_class_base)
    domain_texts, domain_labels = build_dataset(DOMAIN_TEMPLATES, args.n_per_class_domain)
    print(f"Base dataset: {len(base_texts)} examples (general IT-support phrasing)")
    print(f"Domain dataset: {len(domain_texts)} examples (company-specific jargon)")

    base_train_x, base_train_y, base_test_x, base_test_y = train_test_split_simple(base_texts, base_labels)
    domain_train_x, domain_train_y, domain_test_x, domain_test_y = train_test_split_simple(domain_texts, domain_labels)

    print_header("STEP 2 - TRAINING BASE MODEL (general data only)")
    base_model = TicketClassifier()
    base_model.train(base_train_x, base_train_y)
    base_on_base_eval = base_model.evaluate(base_test_x, base_test_y)
    print(f"Base model accuracy on GENERAL test set: {base_on_base_eval['accuracy']:.1%}")

    print_header("STEP 3 - EVALUATING BASE MODEL ON DOMAIN DATA (before fine-tuning)")
    base_on_domain_eval = base_model.evaluate(domain_test_x, domain_test_y)
    print(f"Base model accuracy on DOMAIN test set (before fine-tuning): {base_on_domain_eval['accuracy']:.1%}")
    print("--> This is the 'domain gap': the base model struggles with jargon it never saw in training.")

    print_header("STEP 4 - FINE-TUNING ON BASE + DOMAIN DATA")
    adapted_model = TicketClassifier()
    combined_train_x = base_train_x + domain_train_x
    combined_train_y = base_train_y + domain_train_y
    adapted_model.continue_training(combined_train_x, combined_train_y)
    print(f"Fine-tuned on {len(combined_train_x)} combined examples "
          f"({len(base_train_x)} base + {len(domain_train_x)} domain).")

    print_header("STEP 5 - EVALUATING FINE-TUNED MODEL ON DOMAIN DATA (after fine-tuning)")
    adapted_on_domain_eval = adapted_model.evaluate(domain_test_x, domain_test_y)
    print(f"Fine-tuned model accuracy on DOMAIN test set (after fine-tuning): {adapted_on_domain_eval['accuracy']:.1%}")

    adapted_on_base_eval = adapted_model.evaluate(base_test_x, base_test_y)
    print(f"Fine-tuned model accuracy on GENERAL test set (checking for regression): {adapted_on_base_eval['accuracy']:.1%}")

    print_header("SUMMARY: DOMAIN ADAPTATION IMPACT")
    delta = adapted_on_domain_eval["accuracy"] - base_on_domain_eval["accuracy"]
    print(f"{'Model':25s} {'Domain-test acc':>18s} {'General-test acc':>18s}")
    print(f"{'Base (pre-finetune)':25s} {base_on_domain_eval['accuracy']:17.1%} {base_on_base_eval['accuracy']:18.1%}")
    print(f"{'Fine-tuned':25s} {adapted_on_domain_eval['accuracy']:17.1%} {adapted_on_base_eval['accuracy']:18.1%}")
    print(f"\nDomain accuracy change from fine-tuning: {delta:+.1%}")

    out_path = os.path.join(os.path.dirname(__file__), args.model_out)
    adapted_model.save(out_path)
    print(f"\nFine-tuned model saved to: {out_path}")

    results = {
        "base_on_general_test": base_on_base_eval["accuracy"],
        "base_on_domain_test_before_finetune": base_on_domain_eval["accuracy"],
        "finetuned_on_domain_test_after_finetune": adapted_on_domain_eval["accuracy"],
        "finetuned_on_general_test_regression_check": adapted_on_base_eval["accuracy"],
        "domain_accuracy_delta": delta,
    }
    results_path = os.path.join(os.path.dirname(__file__), "domain_adaptation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results summary saved to: {results_path}")


if __name__ == "__main__":
    main()

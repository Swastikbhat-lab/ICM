
import json
import random
import math
import requests
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import time



API_KEY = "Key_here"  # Placeholder, set via run_experiment
BASE_URL = "https://api.hyperbolic.xyz/v1"


BASE_MODEL = "meta-llama/Meta-Llama-3.1-405B"  
CHAT_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct"  

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests

# ICM Hyperparameters 
T0 = 10          # Initial temperature
T_MIN = 0.01     # Final temperature  
BETA = 0.99      # Cooling rate
ALPHA = 50       # Weight for mutual predictability vs inconsistency
K = 8            # Number of initial random examples
N_ITERATIONS = 256  # Number of ICM iterations

# ============================================================
# API FUNCTIONS
# ============================================================

def call_base_model(prompt: str, max_tokens: int = 1, temperature: float = 0.0, logprobs: int = 20) -> dict:
    """
    Call the BASE model using /completions endpoint.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": BASE_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": logprobs,
    }
    
    for attempt in range(5):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.post(
                f"{BASE_URL}/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            
            if response.status_code == 429:
                wait_time = (2 ** attempt) * 2
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    
    return None


def call_chat_model(prompt: str, max_tokens: int = 1, temperature: float = 0.0, logprobs: int = 20) -> dict:

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": logprobs,
    }
    
    for attempt in range(5):
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            
            if response.status_code == 429:
                wait_time = (2 ** attempt) * 2
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    
    return None


def get_label_logprobs_base(question: str, choice: str, context_examples: List[dict]) -> Tuple[float, float]:

    prompt = build_prompt(question, choice, context_examples)
    result = call_base_model(prompt, max_tokens=1, logprobs=20)
    
    if result is None:
        return -1.0, -1.0
    
    try:
        # Parse logprobs from completions response
        top_logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
        
        log_prob_true = -100.0
        log_prob_false = -100.0
        
        for token, logprob in top_logprobs.items():
            token_lower = token.strip().lower()
            if token_lower in ["true", "tr", "tru", " true"]:
                log_prob_true = max(log_prob_true, logprob)
            elif token_lower in ["false", "fal", "fa", " false"]:
                log_prob_false = max(log_prob_false, logprob)
        
        return log_prob_true, log_prob_false
        
    except (KeyError, IndexError, TypeError) as e:
        # Fallback: check generated text
        try:
            text = result["choices"][0]["text"].strip().lower()
            if "true" in text:
                return 0.0, -10.0
            elif "false" in text:
                return -10.0, 0.0
        except:
            pass
        return -1.0, -1.0


def get_label_logprobs_chat(question: str, choice: str, context_examples: List[dict]) -> Tuple[float, float]:
    
    prompt = build_prompt(question, choice, context_examples)
    result = call_chat_model(prompt, max_tokens=1, logprobs=20)
    
    if result is None:
        return -1.0, -1.0
    
    try:
        # Parse logprobs from chat completions response
        content = result["choices"][0]["logprobs"]["content"]
        if not content:
            # Fallback to text
            text = result["choices"][0]["message"]["content"].strip().lower()
            if "true" in text:
                return 0.0, -10.0
            elif "false" in text:
                return -10.0, 0.0
            return -1.0, -1.0
        
        top_logprobs = content[0]["top_logprobs"]
        
        log_prob_true = -100.0
        log_prob_false = -100.0
        
        for item in top_logprobs:
            token = item["token"].strip().lower()
            logprob = item["logprob"]
            if token in ["true", "tr", "tru", " true"]:
                log_prob_true = max(log_prob_true, logprob)
            elif token in ["false", "fal", "fa", " false"]:
                log_prob_false = max(log_prob_false, logprob)
        
        return log_prob_true, log_prob_false
        
    except (KeyError, IndexError, TypeError) as e:
        return -1.0, -1.0


def build_prompt(question: str, choice: str, context_examples: List[dict]) -> str:
   
    prompt_parts = []
    
   
    for ex in context_examples:
        label_str = "True" if ex["label"] == 1 else "False"
        prompt_parts.append(
            f"Question: {ex['question']}\n"
            f"Claim: {ex['choice']}\n"
            f"I think this Claim is {label_str}"
        )
    
    # Add the query example 
    prompt_parts.append(
        f"Question: {question}\n"
        f"Claim: {choice}\n"
        f"I think this Claim is"
    )
    
    return "\n\n".join(prompt_parts)



def compute_inconsistency(D: List[dict]) -> int:
    """
    Compute I(D) = count of inconsistent label pairs.
    """
    groups = {}
    for ex in D:
        cid = ex.get("consistency_id", id(ex))
        if cid not in groups:
            groups[cid] = []
        groups[cid].append(ex)
    
    inconsistencies = 0
    for cid, examples in groups.items():
        true_count = sum(1 for ex in examples if ex.get("label") == 1)
        if true_count > 1:
            inconsistencies += true_count * (true_count - 1) // 2
    
    return inconsistencies


def compute_U_fast(D: List[dict], cached_logprobs: dict, alpha: float = ALPHA) -> float:
    """
    U(D) = alpha * P_theta(D) - I(D)
    """
    P = 0.0
    for ex in D:
        key = (ex["question"], ex["choice"])
        if key in cached_logprobs:
            log_prob_true, log_prob_false = cached_logprobs[key]
            if ex.get("label") == 1:
                P += log_prob_true
            else:
                P += log_prob_false
    
    I = compute_inconsistency(D)
    return alpha * P - I




def run_icm(
    unlabeled_data: List[dict],
    n_iterations: int = N_ITERATIONS,
    k_init: int = K,
    t0: float = T0,
    t_min: float = T_MIN,
    beta: float = BETA,
    alpha: float = ALPHA
) -> List[dict]:
   
    print(f"\n{'='*60}")
    print("Running ICM Algorithm")
    print(f"{'='*60}")
    print(f"Model: {BASE_MODEL}")
    print(f"Dataset size: {len(unlabeled_data)}")
    print(f"Iterations: {n_iterations}")
    print(f"K (initial): {k_init}, Alpha: {alpha}")
    print()
    
    # Make a copy
    data = [dict(ex) for ex in unlabeled_data]
    for ex in data:
        ex["icm_label"] = None
        ex["in_D"] = False
    
    # Step 1: Initialize D with K randomly labeled examples
    print(f"Step 1: Initializing with {k_init} random examples...")
    initial_indices = random.sample(range(len(data)), k_init)
    D = []
    for idx in initial_indices:
        data[idx]["icm_label"] = random.choice([0, 1])
        data[idx]["in_D"] = True
        d_item = dict(data[idx])
        d_item["label"] = d_item["icm_label"]
        D.append(d_item)
    
    print(f"Initial D size: {len(D)}")
    
    # Cache for log probabilities
    logprob_cache = {}
    
    # Get initial log probs
    print("Getting initial log probabilities...")
    for i, ex in enumerate(tqdm(D, desc="Initial logprobs")):
        context = [d for j, d in enumerate(D) if j != i]
        lp_true, lp_false = get_label_logprobs_base(ex["question"], ex["choice"], context)
        logprob_cache[(ex["question"], ex["choice"])] = (lp_true, lp_false)
    
    # Step 2: Main loop
    print(f"\nStep 2: Running {n_iterations} iterations...")
    
    accepted = 0
    rejected = 0
    
    for n in tqdm(range(1, n_iterations + 1), desc="ICM iterations"):
        # Update temperature
        T = max(t_min, t0 / (1 + beta * math.log(n)))
        
        # Sample an example
        unlabeled_indices = [i for i, ex in enumerate(data) if not ex["in_D"]]
        
        if unlabeled_indices and random.random() < 0.8:
            idx = random.choice(unlabeled_indices)
        else:
            idx = random.randint(0, len(data) - 1)
        
        x_i = data[idx]
        
        # Get context
        if x_i["in_D"]:
            context = [d for d in D if d["question"] != x_i["question"] or d["choice"] != x_i["choice"]]
        else:
            context = D.copy()
        
        if len(context) > 30:
            context = random.sample(context, 30)
        
        # Get label prediction
        lp_true, lp_false = get_label_logprobs_base(x_i["question"], x_i["choice"], context)
        logprob_cache[(x_i["question"], x_i["choice"])] = (lp_true, lp_false)
        
        y_hat = 1 if lp_true > lp_false else 0
        
        # Create candidate D_hat
        if x_i["in_D"]:
            D_hat = []
            for d in D:
                if d["question"] == x_i["question"] and d["choice"] == x_i["choice"]:
                    new_d = dict(d)
                    new_d["label"] = y_hat
                    D_hat.append(new_d)
                else:
                    D_hat.append(d)
        else:
            new_entry = dict(x_i)
            new_entry["label"] = y_hat
            D_hat = D + [new_entry]
        
        # Compute delta
        U_D = compute_U_fast(D, logprob_cache, alpha)
        U_D_hat = compute_U_fast(D_hat, logprob_cache, alpha)
        delta = U_D_hat - U_D
        
        # Accept or reject
        accept = False
        if delta > 0:
            accept = True
        else:
            if random.random() < math.exp(delta / T):
                accept = True
        
        if accept:
            if x_i["in_D"]:
                for d in D:
                    if d["question"] == x_i["question"] and d["choice"] == x_i["choice"]:
                        d["label"] = y_hat
                x_i["icm_label"] = y_hat
            else:
                x_i["icm_label"] = y_hat
                x_i["in_D"] = True
                new_entry = dict(x_i)
                new_entry["label"] = y_hat
                D.append(new_entry)
            accepted += 1
        else:
            rejected += 1
        
        if n % 25 == 0:
            tqdm.write(f"  Iter {n}: |D|={len(D)}, T={T:.4f}, accepted={accepted}, rejected={rejected}")
    
    print(f"\nICM complete!")
    print(f"Final D size: {len(D)}")
    print(f"Accepted: {accepted}, Rejected: {rejected}")
    
    return D


# ============================================================
# EVALUATION
# ============================================================

def evaluate_zero_shot_base(test_data: List[dict]) -> float:
    """Zero-shot with BASE model."""
    print(f"\nEvaluating zero-shot (Base): {BASE_MODEL}")
    correct = 0
    total = len(test_data)
    
    for ex in tqdm(test_data, desc="Zero-shot (Base)"):
        lp_true, lp_false = get_label_logprobs_base(ex["question"], ex["choice"], [])
        predicted = 1 if lp_true > lp_false else 0
        if predicted == ex["label"]:
            correct += 1
    
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def evaluate_zero_shot_chat(test_data: List[dict]) -> float:
    """Zero-shot with CHAT model."""
    print(f"\nEvaluating zero-shot (Chat): {CHAT_MODEL}")
    correct = 0
    total = len(test_data)
    
    for ex in tqdm(test_data, desc="Zero-shot (Chat)"):
        lp_true, lp_false = get_label_logprobs_chat(ex["question"], ex["choice"], [])
        predicted = 1 if lp_true > lp_false else 0
        if predicted == ex["label"]:
            correct += 1
    
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def evaluate_few_shot(test_data: List[dict], context: List[dict], desc: str = "Few-shot") -> float:
    """Few-shot with BASE model."""
    print(f"\nEvaluating {desc} with {len(context)} examples (Base model)")
    correct = 0
    total = len(test_data)
    
    ctx = context[:30] if len(context) > 30 else context
    
    for ex in tqdm(test_data, desc=desc):
        lp_true, lp_false = get_label_logprobs_base(ex["question"], ex["choice"], ctx)
        predicted = 1 if lp_true > lp_false else 0
        if predicted == ex["label"]:
            correct += 1
    
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy




def run_experiment(train_path: str, test_path: str, api_key: str):
    global API_KEY
    API_KEY = api_key
    
    print("Loading data...")
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    results = {}
    
    # 1. Zero-shot (Base)
    print("\n" + "="*60)
    print("Condition 1: Zero-shot (Base)")
    print("="*60)
    results["Zero-shot\n(Base)"] = evaluate_zero_shot_base(test_data)
    save_partial(results)
    
    # 2. Zero-shot (Chat)
    print("\n" + "="*60)
    print("Condition 2: Zero-shot (Chat)")
    print("="*60)
    results["Zero-shot\n(Chat)"] = evaluate_zero_shot_chat(test_data)
    save_partial(results)
    
    # 3. Golden Supervision
    print("\n" + "="*60)
    print("Condition 3: Golden Supervision")
    print("="*60)
    results["Golden\nSupervision"] = evaluate_few_shot(test_data, train_data, "Golden")
    save_partial(results)
    
    # 4. ICM (Unsupervised)
    print("\n" + "="*60)
    print("Condition 4: ICM (Unsupervised)")
    print("="*60)
    icm_labeled = run_icm(train_data)
    
    with open("icm_labels.json", 'w') as f:
        json.dump(icm_labeled, f, indent=2)
    
    results["Unsupervised\n(Ours)"] = evaluate_few_shot(test_data, icm_labeled, "ICM")
    
    return results


def save_partial(results):
    with open("results_partial.json", 'w') as f:
        json.dump(results, f, indent=2)


def save_results_and_plot(results: dict, output_path: str = "results.png"):
    import matplotlib.pyplot as plt
    
    with open("results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results.json")
    
    conditions = list(results.keys())
    accuracies = list(results.values())
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(conditions, accuracies, color=colors[:len(conditions)])
    
    ax.set_ylabel('accuracy (%)', fontsize=12)
    ax.set_title('TruthfulQA', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ICM on TruthfulQA")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--train", type=str, default="data/truthfulqa_train.json")
    parser.add_argument("--test", type=str, default="data/truthfulqa_test.json")
    parser.add_argument("--output", type=str, default="results.png")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ICM Experiment - TruthfulQA")
    print("="*60)
    print(f"Base Model: {BASE_MODEL}")
    print(f"Chat Model: {CHAT_MODEL}")
    print()
    
    results = run_experiment(args.train, args.test, args.api_key)
    save_results_and_plot(results, args.output)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for cond, acc in results.items():
        print(f"{cond.replace(chr(10), ' ')}: {acc:.2f}%")

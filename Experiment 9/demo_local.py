"""
Self-contained demo of the Reasoning Model Benchmarking Lab.
Runs entirely locally using simulated LLM responses to demonstrate
the effect of different prompting strategies without requiring API keys.

Also generates results/raw_results.json so that analyze_results.py
works end-to-end without needing real API calls.
"""
import os
import json
import time
import random
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from rich import box
import matplotlib.pyplot as plt

from config import PROBLEMS, STRATEGIES
from prompting_strategies import STRATEGY_FUNCTIONS

console = Console()

# ---------------------------------------------------------------------------
# Simulated responses per problem x strategy
# Keys: (problem_id, strategy) -> response text
# These are crafted to realistically show how strategies change output quality.
# ---------------------------------------------------------------------------

SIMULATED_RESPONSES = {
    # ── Problem 1: Bat & Ball (Math / Logic Trap) ──────────────────────────
    ("p1_math_simple", "zero_shot"): "The ball costs $0.10.",
    ("p1_math_simple", "few_shot"): (
        "Answer: The ball costs $0.05.\n"
        "Wait, let me double check. If ball is 0.05, bat is 1.05. Total is 1.10. Yes, $0.05."
    ),
    ("p1_math_simple", "chain_of_thought"): (
        "Let's think step by step.\n"
        "1. Let the cost of the ball be 'x'.\n"
        "2. The bat costs $1.00 more than the ball, so the bat costs 'x + 1.00'.\n"
        "3. Together they cost $1.10: x + (x + 1.00) = 1.10\n"
        "4. Combine terms: 2x + 1.00 = 1.10\n"
        "5. Subtract 1.00: 2x = 0.10\n"
        "6. Divide by 2: x = 0.05\n"
        "Therefore, the ball costs $0.05."
    ),
    ("p1_math_simple", "role_based"): (
        "As an expert mathematician, I frequently see students fall for this intuitive trap! "
        "The intuitive answer is 10 cents, but let's break it down algebraically. "
        "If B = ball cost, then Bat = B + 1.00. B + Bat = 1.10. "
        "Substituting: 2B + 1.00 = 1.10, so 2B = 0.10, so B = 0.05. "
        "The ball costs exactly 5 cents."
    ),
    ("p1_math_simple", "structured_output"): (
        '{\n  "reasoning": "Let x = ball. Bat = x + 1. x + (x+1) = 1.1, 2x = 0.1, x = 0.05.",\n'
        '  "answer": "$0.05",\n  "confidence": 100\n}'
    ),

    # ── Problem 2: Syllogistic Logic ──────────────────────────────────────
    ("p2_logic_deduction", "zero_shot"): "Yes.",
    ("p2_logic_deduction", "few_shot"): "Yes, all bloops are definitely lazzies because the chain of inclusion holds.",
    ("p2_logic_deduction", "chain_of_thought"): (
        "Let's think step by step.\n"
        "1. All bloops are razzies (Bloop is a subset of Razzy).\n"
        "2. All razzies are lazzies (Razzy is a subset of Lazzy).\n"
        "3. By transitivity: Bloop -> Razzy -> Lazzy.\n"
        "4. Therefore all bloops are definitely lazzies.\n"
        "Answer: Yes."
    ),
    ("p2_logic_deduction", "role_based"): (
        "This is a classic syllogism -- a staple of formal logic. The transitive property of "
        "subset inclusion gives us: Bloop <= Razzy <= Lazzy. Since subset inclusion is transitive, "
        "Bloop <= Lazzy. So yes, all bloops are definitely lazzies. "
        "This follows directly from Barbara syllogism (AAA-1) in traditional logic."
    ),
    ("p2_logic_deduction", "structured_output"): (
        '{\n  "reasoning": "Bloop subset Razzy subset Lazzy. By transitivity, Bloop subset Lazzy.",\n'
        '  "answer": "Yes, all bloops are definitely lazzies.",\n  "confidence": 99\n}'
    ),

    # ── Problem 3: Spatial Reasoning (3×3×3 cube) ─────────────────────────
    ("p3_spatial_reasoning", "zero_shot"): "6 cubes have exactly one red face.",
    ("p3_spatial_reasoning", "few_shot"): "There are 6 smaller cubes with exactly one red face -- one at the center of each face of the original cube.",
    ("p3_spatial_reasoning", "chain_of_thought"): (
        "Let's think step by step.\n"
        "1. A 3x3x3 cube cut into 27 unit cubes.\n"
        "2. Corner cubes (8 total): 3 painted faces each.\n"
        "3. Edge cubes (12 total): 2 painted faces each.\n"
        "4. Face-center cubes (6 total): 1 painted face each. <-- This is what we want.\n"
        "5. Core cube (1 total): 0 painted faces.\n"
        "Check: 8 + 12 + 6 + 1 = 27 [OK]\n"
        "Answer: 6 cubes have exactly one red face."
    ),
    ("p3_spatial_reasoning", "role_based"): (
        "Excellent spatial reasoning problem! Let me categorize all 27 sub-cubes by position. "
        "Corners sit at 3-face intersections -> 3 red faces (8 cubes). "
        "Edges sit at 2-face intersections -> 2 red faces (12 cubes). "
        "Face centers touch exactly 1 outer face -> 1 red face (6 cubes). "
        "The single interior cube has 0 red faces. "
        "Verification: 8+12+6+1=27. Answer: 6."
    ),
    ("p3_spatial_reasoning", "structured_output"): (
        '{\n  "reasoning": "Corners=8(3 faces), Edges=12(2 faces), Face-centers=6(1 face), Core=1(0 faces). Total=27.",\n'
        '  "answer": "6",\n  "confidence": 100\n}'
    ),

    # ── Problem 4: Water Jug (Algorithmic) ────────────────────────────────
    ("p4_algorithmic_thinking", "zero_shot"): "Fill the 5-gallon jug, pour into the 3-gallon jug, leaving 2 gallons. Empty the 3-gallon jug, pour the 2 gallons in, fill the 5-gallon jug and pour into the 3-gallon jug again. Now 4 gallons remain in the 5-gallon jug.",
    ("p4_algorithmic_thinking", "few_shot"): (
        "Step 1: Fill 5-gallon jug (5,0).\n"
        "Step 2: Pour into 3-gallon jug (2,3).\n"
        "Step 3: Empty 3-gallon jug (2,0).\n"
        "Step 4: Pour 2 gallons into 3-gallon jug (0,2).\n"
        "Step 5: Fill 5-gallon jug (5,2).\n"
        "Step 6: Pour from 5 into 3 until full (4,3).\n"
        "Result: 4 gallons in the 5-gallon jug."
    ),
    ("p4_algorithmic_thinking", "chain_of_thought"): (
        "Let's think step by step. I'll track state as (5-gal, 3-gal).\n"
        "Start: (0, 0)\n"
        "Step 1 -- Fill the 5-gallon jug: (5, 0)\n"
        "Step 2 -- Pour 5->3 until 3 is full: (2, 3)\n"
        "Step 3 -- Empty the 3-gallon jug: (2, 0)\n"
        "Step 4 -- Pour the 2 gallons into the 3-gallon jug: (0, 2)\n"
        "Step 5 -- Fill the 5-gallon jug again: (5, 2)\n"
        "Step 6 -- Pour 5->3 (only 1 gallon fits): (4, 3)\n"
        "We now have exactly 4 gallons in the 5-gallon jug. [OK]"
    ),
    ("p4_algorithmic_thinking", "role_based"): (
        "This is the classic Die Hard 3 water jug problem! It's a well-known example in "
        "state-space search and number theory. The key insight is that any volume measurable "
        "with jugs of capacity a and b must be a multiple of gcd(a,b). Since gcd(5,3)=1, "
        "we can measure any integer volume.\n\n"
        "Optimal solution in 6 steps:\n"
        "(0,0)->(5,0)->(2,3)->(2,0)->(0,2)->(5,2)->(4,3)\n"
        "Result: 4 gallons in the 5-gallon jug."
    ),
    ("p4_algorithmic_thinking", "structured_output"): (
        '{\n  "reasoning": "States: (0,0)->(5,0)->(2,3)->(2,0)->(0,2)->(5,2)->(4,3). 6 steps using fill/pour/empty.",\n'
        '  "answer": "Fill 5, pour to 3 (leaves 2), empty 3, move 2 to 3, fill 5, pour 1 into 3 -> 4 gallons in 5-gal jug.",\n'
        '  "confidence": 100\n}'
    ),

    # ── Problem 5: Probability ────────────────────────────────────────────
    ("p5_probability", "zero_shot"): "The probability is 1/2.",
    ("p5_probability", "few_shot"): "The probability is 1/3. The sample space with at least one girl is {GG, GB, BG}, and only GG has both girls.",
    ("p5_probability", "chain_of_thought"): (
        "Let's think step by step.\n"
        "1. All equally likely outcomes for two children: {BB, BG, GB, GG}.\n"
        "2. Condition: at least one is a girl -> eliminate BB -> {BG, GB, GG}.\n"
        "3. Favorable outcome: both girls -> {GG}.\n"
        "4. P(both girls | at least one girl) = 1/3.\n"
        "Answer: 1/3."
    ),
    ("p5_probability", "role_based"): (
        "This is the classic 'Boy or Girl' paradox, a wonderful example of how conditional "
        "probability can be counter-intuitive. Many students guess 1/2, but the correct "
        "answer requires careful enumeration.\n\n"
        "Sample space: {BB, BG, GB, GG} -- each with P = 1/4.\n"
        "Event A (at least one girl): {BG, GB, GG}, P(A) = 3/4.\n"
        "Event B (both girls): {GG}, P(B) = 1/4.\n"
        "P(B|A) = P(B&A)/P(A) = (1/4)/(3/4) = 1/3.\n\n"
        "The answer is 1/3, NOT 1/2."
    ),
    ("p5_probability", "structured_output"): (
        '{\n  "reasoning": "Sample space: {BB,BG,GB,GG}. Given >=1 girl: {BG,GB,GG}. P(GG|>=1 girl) = 1/3.",\n'
        '  "answer": "1/3",\n  "confidence": 98\n}'
    ),
}

# Simulated latency profiles (ms) -- longer prompts & richer outputs take more time
LATENCY_PROFILES = {
    "zero_shot":       {"base": 350,  "jitter": 100},
    "few_shot":        {"base": 800,  "jitter": 200},
    "chain_of_thought": {"base": 1900, "jitter": 400},
    "role_based":      {"base": 1600, "jitter": 350},
    "structured_output": {"base": 1100, "jitter": 250},
}

# Models to simulate
SIMULATED_MODELS = ["gpt-4o", "gemini-1.5-pro", "claude-3-5-sonnet-20240620"]
MODEL_SPEED_FACTOR = {"gpt-4o": 1.0, "gemini-1.5-pro": 0.85, "claude-3-5-sonnet-20240620": 1.1}


def generate_raw_results():
    """
    Generate simulated raw_results.json for all models x strategies x problems.
    This allows analyze_results.py to work without real API calls.
    """
    random.seed(42)  # Reproducible results
    all_results = []

    for model in SIMULATED_MODELS:
        speed = MODEL_SPEED_FACTOR[model]
        for strategy in STRATEGIES:
            lat_profile = LATENCY_PROFILES[strategy]
            for problem in PROBLEMS:
                pid = problem["id"]
                strategy_func = STRATEGY_FUNCTIONS[strategy]
                prompt = strategy_func(problem["question"])
                response = SIMULATED_RESPONSES.get((pid, strategy), "Simulated response.")
                latency = int((lat_profile["base"] + random.randint(-lat_profile["jitter"], lat_profile["jitter"])) * speed)
                tokens = max(10, len(response.split()) + random.randint(-5, 15))

                all_results.append({
                    "model": model,
                    "strategy": strategy,
                    "problem_id": pid,
                    "prompt": prompt,
                    "response": response,
                    "latency_ms": latency,
                    "tokens": tokens,
                    "error": None,
                })

    os.makedirs("results", exist_ok=True)
    with open("results/raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def generate_demo_charts():
    """Generates sample charts to demonstrate the analysis pipeline."""
    os.makedirs("results/demo_charts", exist_ok=True)

    strategies = ["zero_shot", "few_shot", "CoT", "role_based", "structured"]
    latencies = [400, 850, 2100, 1800, 1200]

    plt.figure(figsize=(8, 5))
    plt.bar(strategies, latencies, color=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"])
    plt.title("Simulated Model Latency by Prompting Strategy")
    plt.ylabel("Latency (ms)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    chart_path = "results/demo_charts/simulated_latency.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path


def main():
    console.print("\n[bold blue on white] Reasoning Model Benchmarking Lab - LOCAL DEMO [/bold blue on white]\n")
    console.print(
        "This demo runs [bold]without API keys[/bold], using simulated responses to demonstrate\n"
        "how different prompting strategies affect LLM outputs.\n"
    )

    # ── Show the target problem ────────────────────────────────────────────
    problem = PROBLEMS[0]
    console.print(Panel(problem["question"], title="[bold yellow]Target Problem[/bold yellow]", border_style="yellow"))

    # ── Run simulation for display ─────────────────────────────────────────
    display_results = []

    with Progress() as progress:
        task = progress.add_task("[cyan]Simulating model responses...", total=len(STRATEGIES))

        for strategy in STRATEGIES:
            strategy_func = STRATEGY_FUNCTIONS[strategy]
            prompt = strategy_func(problem["question"])
            response_text = SIMULATED_RESPONSES[(problem["id"], strategy)]

            display_results.append({
                "strategy": strategy,
                "prompt": prompt,
                "response": response_text,
            })

            time.sleep(0.5)  # Visual effect
            progress.advance(task)

    # ── Display side-by-side comparison ────────────────────────────────────
    console.print("\n[bold green]Simulation Complete! Comparing Strategies:[/bold green]\n")

    for r in display_results:
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAVY)
        table.add_column("Strategy: " + r["strategy"].upper(), width=45)
        table.add_column("Simulated Output", width=55, style="green")

        display_prompt = r["prompt"]
        if len(display_prompt) > 200:
            display_prompt = display_prompt[:197] + "..."

        table.add_row(f"[italic]{display_prompt}[/italic]", r["response"])
        console.print(table)
        console.print("-" * 105)

    # ── Generate raw_results.json for the full analysis pipeline ───────────
    console.print("\n[bold cyan]Generating simulated benchmark data for all models x strategies x problems...[/bold cyan]")
    all_results = generate_raw_results()
    console.print(f"[green][OK] Saved {len(all_results)} simulated results to results/raw_results.json[/green]")

    # ── Generate demo charts ───────────────────────────────────────────────
    chart_path = generate_demo_charts()
    console.print(f"[green][OK] Generated demo chart at {chart_path}[/green]")

    # ── Next steps ─────────────────────────────────────────────────────────
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  1. Run [bold cyan]python analyze_results.py[/bold cyan] to see tables & charts from simulated data")
    console.print("  2. Add real API keys to [bold].env[/bold] and run [bold cyan]python benchmark.py[/bold cyan]")
    console.print("  3. Run [bold cyan]python analyze_results.py[/bold cyan] again for real metrics\n")


if __name__ == "__main__":
    main()


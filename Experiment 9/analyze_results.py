import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from tabulate import tabulate
from rich.console import Console

console = Console()

def load_results(filepath="results/raw_results.json"):
    if not os.path.exists(filepath):
        console.print(f"[bold red]Error:[/bold red] File {filepath} not found.")
        return []
    with open(filepath, "r") as f:
        return json.load(f)

def analyze():
    results = load_results()
    if not results:
        return

    os.makedirs("results", exist_ok=True)
    
    # 1. Calculate Average Latency per Model per Strategy
    latency_data = defaultdict(lambda: defaultdict(list))
    length_data = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        if not r["error"]:
            latency_data[r["model"]][r["strategy"]].append(r["latency_ms"])
            length_data[r["model"]][r["strategy"]].append(len(r["response"]))

    # Aggregate averages
    models = sorted(list(latency_data.keys()))
    if not models:
        console.print("[yellow]No successful results to analyze.[/yellow]")
        return
        
    strategies = sorted(list(latency_data[models[0]].keys()))
    
    avg_latency_table = []
    for model in models:
        row = [model]
        for strategy in strategies:
            lats = latency_data[model][strategy]
            avg = sum(lats)/len(lats) if lats else 0
            row.append(f"{avg:.0f} ms")
        avg_latency_table.append(row)
        
    console.print("\n[bold cyan]Average Latency by Model and Strategy[/bold cyan]")
    print(tabulate(avg_latency_table, headers=["Model"] + strategies, tablefmt="grid"))
    
    # 2. Generate Bar Chart for Latency
    plt.figure(figsize=(10, 6))
    x = range(len(models))
    width = 0.15
    
    for i, strategy in enumerate(strategies):
        y = []
        for model in models:
            lats = latency_data[model][strategy]
            y.append(sum(lats)/len(lats) if lats else 0)
        offset = (i - len(strategies)/2) * width + width/2
        plt.bar([pos + offset for pos in x], y, width=width, label=strategy)
        
    plt.xlabel('Models')
    plt.ylabel('Average Latency (ms)')
    plt.title('Latency Comparison across Models and Strategies')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    chart_path = "results/latency_chart.png"
    plt.savefig(chart_path)
    console.print(f"[green]Saved latency chart to {chart_path}[/green]")
    
    # 3. Response Length Heatmap (using table for terminal)
    avg_len_table = []
    for model in models:
        row = [model]
        for strategy in strategies:
            lens = length_data[model][strategy]
            avg = sum(lens)/len(lens) if lens else 0
            row.append(f"{avg:.0f} chars")
        avg_len_table.append(row)
        
    console.print("\n[bold cyan]Average Response Length by Model and Strategy[/bold cyan]")
    print(tabulate(avg_len_table, headers=["Model"] + strategies, tablefmt="grid"))

if __name__ == "__main__":
    analyze()

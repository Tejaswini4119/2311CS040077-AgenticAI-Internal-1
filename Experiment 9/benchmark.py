import argparse
import os
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich import box

from config import MODELS, STRATEGIES, PROBLEMS
from prompting_strategies import STRATEGY_FUNCTIONS
from model_clients import get_model_client

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Run Reasoning Model Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without calling APIs")
    parser.add_argument("--model", type=str, help="Filter to specific model name")
    parser.add_argument("--strategy", type=str, help="Filter to specific strategy name")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Setup results directory
    os.makedirs("results", exist_ok=True)
    results_file = "results/raw_results.json"
    
    # Load existing results if any
    all_results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            pass

    # Filter targets based on args
    target_models = [m for m in MODELS if not args.model or m == args.model]
    target_strategies = [s for s in STRATEGIES if not args.strategy or s == args.strategy]
    
    total_tasks = len(target_models) * len(target_strategies) * len(PROBLEMS)
    
    console.print(f"[bold blue]Starting Benchmark...[/bold blue]")
    console.print(f"Models: {len(target_models)}, Strategies: {len(target_strategies)}, Problems: {len(PROBLEMS)}")
    console.print(f"Total tasks: {total_tasks}")
    
    if args.dry_run:
        console.print("[yellow]Running in DRY-RUN mode. No APIs will be called.[/yellow]\n")
        
    results = []
    
    with Progress() as progress:
        task_id = progress.add_task("[green]Benchmarking...", total=total_tasks)
        
        for model in target_models:
            client_func = get_model_client(model)
            
            for strategy in target_strategies:
                strategy_func = STRATEGY_FUNCTIONS[strategy]
                
                for problem in PROBLEMS:
                    # Format prompt
                    prompt = strategy_func(problem["question"])
                    
                    if args.dry_run:
                        console.print(f"\n[bold magenta]Model:[/bold magenta] {model} | [bold magenta]Strategy:[/bold magenta] {strategy}")
                        console.print(f"[bold cyan]Prompt:[/bold cyan]\n{prompt}\n" + "-"*50)
                        progress.advance(task_id)
                        continue
                    
                    # Call API
                    api_result = client_func(prompt, model)
                    
                    result_entry = {
                        "model": model,
                        "strategy": strategy,
                        "problem_id": problem["id"],
                        "prompt": prompt,
                        "response": api_result["response_text"],
                        "latency_ms": api_result["latency_ms"],
                        "tokens": api_result["token_count"],
                        "error": api_result["error"]
                    }
                    
                    results.append(result_entry)
                    all_results.append(result_entry)
                    
                    # Save incrementally
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=2)
                        
                    progress.advance(task_id)

    if not args.dry_run:
        console.print(f"\n[bold green]Benchmark Complete![/bold green] Results saved to {results_file}")
        
        # Display a sample summary table of the session
        table = Table(title="Recent Run Summary", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Strategy", style="magenta")
        table.add_column("Problem", style="yellow")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Status", justify="center")
        
        for r in results:
            status = "[red]Error[/red]" if r["error"] else "[green]Success[/green]"
            table.add_row(
                r["model"], 
                r["strategy"], 
                r["problem_id"], 
                str(r["latency_ms"]), 
                str(r["tokens"]), 
                status
            )
            
        console.print(table)

if __name__ == "__main__":
    main()

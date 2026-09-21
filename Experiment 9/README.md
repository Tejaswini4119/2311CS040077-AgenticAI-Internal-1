# Reasoning Model Benchmarking Lab

This lab experiment provides a framework to compare and benchmark outputs across different Large Language Models (LLMs) using various prompting strategies. It evaluates how models like GPT-4o, Gemini 2.5 Pro, and Claude 3.5 Sonnet handle complex reasoning problems under different conditions.

## Purpose
The primary goal is to observe the impact of prompt engineering techniques on reasoning capabilities, latency, and response structure.

## Setup Instructions

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment Variables:**
    Copy the example environment file and fill in your API keys.
    ```bash
    cp .env.example .env
    ```
    Add your actual keys for OpenAI, Google, and Anthropic in the `.env` file.

3.  **Run the Local Demo (No API Keys Required):**
    To understand how the lab works without incurring API costs, run the self-contained demo:
    ```bash
    python demo_local.py
    ```

4.  **Run the Benchmark (Requires API Keys):**
    ```bash
    python benchmark.py
    ```
    You can also use flags to filter the execution:
    - `--dry-run`: Show formatted prompts without making API calls.
    - `--model`: Filter to a specific model (e.g., `gpt-4o`).
    - `--strategy`: Filter to a specific strategy (e.g., `chain_of_thought`).

5.  **Analyze Results:**
    After running the benchmark, analyze the generated `results/raw_results.json` file.
    ```bash
    python analyze_results.py
    ```

## File Structure

-   `README.md`: This file, explaining the project.
-   `requirements.txt`: Python dependencies.
-   `.env.example`: Template for environment variables.
-   `config.py`: Configuration settings, including models, strategies, and test problems.
-   `prompting_strategies.py`: Functions defining different prompting techniques.
-   `model_clients.py`: API integration code for OpenAI, Google, and Anthropic models.
-   `benchmark.py`: Main script to execute the benchmark across models and strategies.
-   `analyze_results.py`: Script to parse results, print tables, and generate charts.
-   `demo_local.py`: Self-contained demo script using simulated responses.

"""
Module defining different prompting strategies.
Each function takes a base problem string and returns a formatted prompt.
"""

def zero_shot(problem: str) -> str:
    """
    Zero-Shot Strategy: Just asks the question without any prior examples or guidance.
    """
    return f"{problem}"

def few_shot(problem: str) -> str:
    """
    Few-Shot Strategy: Provides examples of similar problems and solutions before the target question.
    """
    prompt = (
        "Here are some examples of problem solving:\n\n"
        "Question: What is 2 + 2?\n"
        "Answer: 4\n\n"
        "Question: If a train travels 60 mph for 2 hours, how far does it go?\n"
        "Answer: 120 miles\n\n"
        "Now, solve the following problem:\n"
        f"Question: {problem}\n"
        "Answer:"
    )
    return prompt

def chain_of_thought(problem: str) -> str:
    """
    Chain-of-Thought (CoT) Strategy: Encourages the model to explicitly lay out its reasoning steps.
    """
    return f"{problem}\n\nLet's think step by step."

def role_based(problem: str) -> str:
    """
    Role-Based Strategy: Assigns an expert persona to the model to guide its response style and depth.
    """
    prompt = (
        "You are an expert mathematician and logician with decades of experience "
        "teaching at a top university. You are known for your precise, clear, and "
        "insightful explanations.\n\n"
        f"Please solve the following problem: {problem}"
    )
    return prompt

def structured_output(problem: str) -> str:
    """
    Structured Output Strategy: Requires the model to format its output in a specific way, like JSON.
    """
    prompt = (
        f"{problem}\n\n"
        "Respond ONLY in valid JSON format with the following fields:\n"
        "- \"reasoning\": A string containing your step-by-step logic.\n"
        "- \"answer\": A concise string containing the final answer.\n"
        "- \"confidence\": An integer from 1 to 100 representing your confidence.\n"
        "Ensure the response can be parsed directly by a JSON parser without any markdown wrapping."
    )
    return prompt

# Map of strategy names to their corresponding functions
STRATEGY_FUNCTIONS = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "chain_of_thought": chain_of_thought,
    "role_based": role_based,
    "structured_output": structured_output
}

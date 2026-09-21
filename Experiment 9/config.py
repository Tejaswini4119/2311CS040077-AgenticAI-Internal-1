"""
Configuration for the Reasoning Model Benchmarking Lab.
Defines models, strategies, test problems, and evaluation criteria.
"""

# List of models to benchmark
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307"
]

# Prompting strategies being evaluated
STRATEGIES = [
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "role_based",
    "structured_output"
]

# Test problems of varying difficulty
PROBLEMS = [
    {
        "id": "p1_math_simple",
        "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "category": "Math / Logic Trap"
    },
    {
        "id": "p2_logic_deduction",
        "question": "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?",
        "category": "Syllogistic Logic"
    },
    {
        "id": "p3_spatial_reasoning",
        "question": "I have a 3x3x3 wooden cube painted red on the outside. If I cut it into 27 smaller 1x1x1 cubes, how many smaller cubes have exactly one red face?",
        "category": "Spatial Reasoning"
    },
    {
        "id": "p4_algorithmic_thinking",
        "question": "You have two jugs, one holds 5 gallons and the other holds 3 gallons. You have an unlimited water supply. How can you measure exactly 4 gallons?",
        "category": "Algorithmic / State Space"
    },
    {
        "id": "p5_probability",
        "question": "In a family with two children, what is the probability that both are girls given that at least one is a girl? Assume boy/girl births are equally likely and independent.",
        "category": "Probability"
    }
]

# Evaluation Criteria (for human review or LLM-as-a-judge)
EVALUATION_CRITERIA = [
    "Accuracy: Is the final answer correct?",
    "Reasoning Quality: Is the step-by-step logic sound?",
    "Formatting: Did it follow formatting constraints (e.g., JSON)?",
    "Conciseness: Is the answer unnecessarily verbose?"
]

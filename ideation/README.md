# Ideation pipeline (Task 3): the deterministic parts

The language-model stages (idea generation, reviews, pairwise judging, rubric scoring) were run interactively with one model and are stored as
JSON in `outputs/`; the scripts here are the parts that are code: reviewer-score selection, Bradley-Terry fitness from pairwise judgements,
seeded selection/crossover/mutation planning, truncation, and the weighted rubric with a Kendall-tau comparison and weight-sensitivity analysis.

    cd ideation/pipeline
    python -I select_gen0.py 6
    python -I fitness.py ../outputs/S6_gen0_P1.json ../outputs/fitness_gen0_P1.json
    python -I evolve.py ../outputs/fitness_gen0_P1.json 3 20261019 ../outputs/evo_plan_gen1_P1.json
    python -I rank.py

The model-generated stages cannot be reproduced by running code.

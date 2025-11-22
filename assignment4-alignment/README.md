# Assignment 4: Alignment

This assignment covers techniques for aligning language models with human preferences.

## Topics Covered

1. **Supervised Fine-Tuning (SFT)**
   - Training on instruction-following data
   - Dataset preparation
   - Loss computation

2. **Reinforcement Learning from Human Feedback (RLHF)**
   - Reward modeling
   - PPO (Proximal Policy Optimization)
   - Value functions

3. **Direct Preference Optimization (DPO)**
   - Preference data collection
   - DPO loss implementation
   - Comparison with RLHF

4. **Group Relative Policy Optimization (GRPO)**
   - Group-based preference learning
   - Implementation details
   - Evaluation metrics

## Files

(Implementation files will be added here)

## Setup

```bash
# Install dependencies
pip install torch transformers datasets trl

# For evaluation
pip install rouge-score bert-score

# Run tests
pytest tests/
```

## Progress

- [ ] Part 1: Supervised Fine-Tuning
- [ ] Part 2: Reward Modeling
- [ ] Part 3: DPO/GRPO Implementation
- [ ] Part 4: Evaluation and Analysis

## Notes

Alignment techniques are crucial for making language models helpful, harmless, and honest. This assignment explores different approaches to alignment.

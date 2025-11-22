# Assignment 2: Systems and Parallelism

This assignment focuses on optimizing and scaling language model training.

## Topics Covered

1. **Performance Optimization**
   - Benchmarking training speed
   - Memory optimization
   - Compute efficiency

2. **Flash Attention**
   - Understanding attention complexity
   - Implementing Flash Attention 2 in Triton
   - Performance comparisons

3. **Distributed Training**
   - Data Parallel (DP)
   - Distributed Data Parallel (DDP)
   - Model parallelism
   - Pipeline parallelism

4. **Optimizer Sharding**
   - ZeRO optimizer states
   - Gradient sharding
   - Parameter sharding

## Files

(Implementation files will be added here)

## Setup

```bash
# Install dependencies
pip install torch numpy transformers triton

# For distributed training
pip install torch.distributed

# Run tests
pytest tests/
```

## Progress

- [ ] Part 1: Benchmarking Script
- [ ] Part 2: Flash Attention Implementation
- [ ] Part 3: DDP Implementation
- [ ] Part 4: Optimizer Sharding

## Notes

This assignment teaches how to scale language model training to larger models and datasets efficiently.

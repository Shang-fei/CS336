# Assignment 3: Data Preparation

This assignment covers data collection, filtering, and processing for language model training.

## Topics Covered

1. **Data Collection**
   - Web scraping (Common Crawl)
   - Data downloading and extraction
   - Handling large datasets

2. **Data Filtering**
   - Quality assessment
   - Language identification
   - Content filtering (NSFW, hate speech)
   - Heuristic-based filtering

3. **Deduplication**
   - Exact deduplication
   - Near-duplicate detection
   - MinHash and LSH

4. **Data Pipeline**
   - End-to-end data processing
   - Parallel processing
   - Storage optimization

## Files

(Implementation files will be added here)

## Setup

```bash
# Install dependencies
pip install fasttext datasets beautifulsoup4 requests

# For deduplication
pip install datasketch

# Run tests
pytest tests/
```

## Progress

- [ ] Part 1: Data Collection
- [ ] Part 2: Quality Filtering
- [ ] Part 3: Deduplication
- [ ] Part 4: Complete Pipeline

## Notes

High-quality training data is crucial for building effective language models. This assignment focuses on creating a robust data pipeline.

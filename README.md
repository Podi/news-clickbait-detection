# news-clickbait-detection

Binary clickbait detection for news headlines and teaser text.

The pilot trains TF-IDF baselines on Webis-Clickbait-17, selects the best model by macro F1 on a stratified Webis holdout split, and evaluates that Webis-trained model on the Kaggle clickbait dataset as an external generalization check.

## How to Run

1. Place the datasets in the expected `data/` structure shown below.
2. Open `clickbait_binary_pilot.ipynb`.
3. Run all notebook cells from top to bottom.

The notebook imports reusable functions from `clickbait_pipeline.py`, runs the full pipeline, and saves figures to `outputs/figures/`.

## Expected Data Structure

```text
data/
  clickbait17-train-170630/
    instances.jsonl
    truth.jsonl
  kaggle/
    clickbait_data.csv
```

## Threshold Logic

Webis includes graded clickbait scores in `truthMean`. The pipeline tests thresholds `0.3, 0.4, 0.5, 0.6, 0.7, 0.8` and reports usable rows plus class balance for each threshold.

The selected threshold is chosen by a simple data-driven rule:

- keep enough samples in both classes
- avoid extreme class imbalance
- prefer the threshold with the best minority/majority balance

With the current local data, this selects threshold `0.3`.

## Main Limitations

- Webis and Kaggle use different sources and labeling conventions, so Kaggle results are an external robustness check, not a guaranteed real-world score.
- Converting Webis scores to binary labels simplifies graded human judgments.
- TF-IDF baselines model surface wording patterns and do not verify factual correctness.
- Some headlines blur the boundary between concise news writing and clickbait style.
- Spoiler generation is intentionally out of scope for this binary-classification pilot.

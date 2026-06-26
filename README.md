# News Clickbait Detection

This project builds a binary classifier for detecting clickbait-style news headlines.

The project idea is based on Potthast et al. [[1]](https://doi.org/10.48550/arxiv.1812.10847), who organized the Clickbait Challenge 2017 in order to automatically detect clickbait in news headlines. Our use of RoBERTa is motivated by Alarfaj et al. [[2]](https://doi.org/10.1038/s41598-025-30229-5), who used RoBERTa-Large and deep embeddings for clickbait detection in news headlines.

## Project Description

In this project, clickbait describes a headline that tries to attract clicks of users by creating curiosity, exaggerating, hiding key information, using emotional wording, or making the reader feel that they need to open the article to understand what happened.

The task is supervised binary classification with two labels:

- `clickbait`
- `non_clickbait`

This project does not detect misinformation or check whether a headline is factually true. A headline can be true and still be clickbait. Clickbait spoiler generation is a later extension of this project which can be find this [repository](https://github.com/Podi/clickbait-spoiler-generation).

## Datasets

The notebook uses data from two Kaggle datasets: the News Clickbait Dataset [[3]](https://www.kaggle.com/datasets/vikassingh1996/news-clickbait-dataset/data) and the Clickbait Dataset [[4]](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset). The three CSV files are placed in the `data/` folder:

```text
data/
  kaggle1_clickbait_data1.csv
  kaggle2_clickbait_data1.csv
  kaggle2_clickbait_data2.csv
```

The files contain headline/title text and clickbait labels. The notebook detects the text and label columns, cleans the text lightly, maps labels to `clickbait` and `non_clickbait`, removes duplicate headlines, and combines the usable rows into one dataset.

## Main Notebook

```text
clickbait_detection.ipynb
```

The notebook covers:

- loading and checking the local CSV files
- light text cleaning
- label normalization
- train/validation/test split
- RoBERTa-base tokenization
- model training and evaluation
- confusion matrices
- an interactive headline prediction demo

## Models Used

The current notebook uses:

- `roberta-base` fine-tuned for binary sequence classification

## How to Run

1. Put the three CSV files in the `data/` folder.
2. Open `clickbait_detection.ipynb`.
3. Run the notebook cells from top to bottom.
4. At the end, use the demo cell to enter a headline and get a prediction with a confidence score.

The notebook installs the needed Python packages in the first cell.

## Results Shown in the Notebook

After duplicate removal, the notebook uses 52,876 headlines:

- 20,252 `clickbait`
- 32,624 `non_clickbait`

The split is:

- train: 37,013 rows
- validation: 7,931 rows
- test: 7,932 rows

The final RoBERTa-base test results shown in the notebook are:

- accuracy: 0.914
- precision: 0.904
- recall: 0.868
- F1: 0.885
- macro F1: 0.908
- ROC-AUC: 0.964

## Limitations

- The model detects clickbait style, not truthfulness.
- Labelling may be subjective as the boundary between catchy wording and clickbait is not always clear.
- The dataset sources may contain repeated patterns, so performance may not fully represent real-world news.
- The model may flag dramatic but legitimate headlines as clickbait.
- The model may miss clickbait headlines that are written in a plain news style.
- Spoiler generation is not included in this implementation.

## References

[1] M. Potthast, T. Gollub, M. Hagen, and B. Stein, "The Clickbait Challenge 2017: Towards a Regression Model for Clickbait Strength," ArXiv (Cornell University), 2018. doi: 10.48550/arxiv.1812.10847.

[2] F. K. Alarfaj, A. Muqadas, H. U. Khan, and A. Naz, "Clickbait detection in news headlines using RoBERTa-Large language model and deep embeddings," Scientific Reports, vol. 16, no. 1, 2025. doi: 10.1038/s41598-025-30229-5.

[3] vikassingh1996, "News Clickbait Dataset," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/vikassingh1996/news-clickbait-dataset/data.

[4] amananandrai, "Clickbait Dataset," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/amananandrai/clickbait-dataset.

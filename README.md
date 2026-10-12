# Turkish Text Categorization

A Google Colab notebook that fine-tunes a Turkish BERT model to sort Turkish news texts into seven categories. It is a compact, end-to-end example for anyone trying BERT fine-tuning on Turkish text.

> **Status:** archived. Built in 2021. Kept for reference, not maintained.

## What it does

- Loads the TTC4900 news corpus (`7allV03.csv`, 4,900 texts, included in this repo) with pandas.
- Maps its seven categories to integer labels: politics, world, economy, culture, health, sports and technology (`siyaset`, `dunya`, `ekonomi`, `kultur`, `saglik`, `spor`, `teknoloji` in the data).
- Splits the data 80/20 with a fixed random seed.
- Fine-tunes [`dbmdz/bert-base-turkish-uncased`](https://huggingface.co/dbmdz/bert-base-turkish-uncased) with simpletransformers' `ClassificationModel` for 3 epochs (batch size 64, fp16 on a GPU).
- Evaluates on the held-out 20% (980 texts) and prints a per-class report. The saved notebook output shows 0.923 accuracy and 0.924 macro F1.

## Tech stack

- Python in a Jupyter notebook on Google Colab (GPU runtime)
- simpletransformers (on top of Hugging Face Transformers and PyTorch)
- pandas, scikit-learn

## Running it locally

The notebook was written for Google Colab:

1. Open `topic_extraction.ipynb` in Colab and switch the runtime to GPU.
2. Run the cells in order. One cell asks you to upload `7allV03.csv` from this repo, another installs simpletransformers.

To run it outside Colab, install `simpletransformers`, replace the upload cells with `df = pd.read_csv("7allV03.csv")`, and make sure a CUDA GPU is available (the model is created with `use_cuda=True` and fp16).

## Credits

- Dataset: [TTC4900](https://www.kaggle.com/datasets/savasy/ttc4900) on Kaggle
- Model: [dbmdz/bert-base-turkish-uncased](https://huggingface.co/dbmdz/bert-base-turkish-uncased)
- Notebooks that inspired this one:
  - [Bert multilingual for Turkish Text Classification](https://www.kaggle.com/code/ayhanc/bert-multilingual-for-turkish-text-classification)
  - [Turkish Text Classification Fasttext and Bert](https://www.kaggle.com/code/mervetncr/turkish-text-classification-fasttext-and-bert)
  - [Text Categorization for Turkish (Multi NB)](https://www.kaggle.com/code/savasy/text-categorization-for-turkish-multi-nb)

## License

MIT, see [LICENSE](LICENSE). The dataset file `7allV03.csv` comes from TTC4900, which is published on Kaggle under CC0.

import pandas as pd

from artifex import Artifex
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm


def main():
    # Load dataset from HuggingFace
    # https://huggingface.co/datasets/Deysi/spam-detection-dataset
    spam_detection = Artifex().spam_detection
    splits = {
        "train": "data/train-00000-of-00001-daf190ce720b3dbb.parquet",
        "test": "data/test-00000-of-00001-fa9b3e8ade89a333.parquet",
    }
    spam_df = pd.read_parquet(
        "hf://datasets/Deysi/spam-detection-dataset/" + splits["train"]
    )

    # Run spam pipeline and align predictions / labels
    prediction_dicts = []
    for row_index, spam_row in tqdm(spam_df.iterrows(), total=spam_df.shape[0]):
        try:
            prediction = spam_detection(spam_row["text"])
            prediction_label = [x.label for x in prediction][0]
            prediction_dicts.append(
                {
                    "prediction": prediction_label,
                    "label": spam_row.label,
                    "text": spam_row.text,
                    "is_correct": True if prediction_label == spam_row.label else False,
                }
            )

        except Exception as e:
            print(e)

    tagged_df = pd.DataFrame(prediction_dicts)
    tagged_df

    # Calculate metrics
    precision = precision_score(
        tagged_df["label"], tagged_df["prediction"], pos_label="spam"
    )
    recall = recall_score(tagged_df["label"], tagged_df["prediction"], pos_label="spam")
    f1 = f1_score(tagged_df["label"], tagged_df["prediction"], pos_label="spam")

    print("=" * 100)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print("=" * 100)


if __name__ == "__main__":
    main()

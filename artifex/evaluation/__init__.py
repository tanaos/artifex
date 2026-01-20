import pandas as pd
from tqdm import tqdm
try:
    from sklearn.metrics import precision_score, recall_score, f1_score
except ImportError as e:
    raise ImportError(
        "This feature requires optional dependencies. "
        "Install with: artifex[evaluation]"
    ) from e
    
from artifex.models import BaseModel


class ModelEvaluator:
    def __init__(self, model: BaseModel, dataset_path: str) -> None:
        self.model = model
        self.df = pd.read_parquet(dataset_path)
    
    def run(self):
        # Run pipeline and align predictions / labels
        prediction_dicts = []
        for row_index, db_row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            try:
                prediction = self.model(db_row["text"])
                prediction_label = [x.label for x in prediction][0]
                prediction_dicts.append(
                    {
                        "prediction": prediction_label,
                        "label": db_row.label,
                        "text": db_row.text,
                        "is_correct": True if prediction_label == db_row.label else False,
                    }
                )

            except Exception as e:
                print(e)

        tagged_df = pd.DataFrame(prediction_dicts)

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
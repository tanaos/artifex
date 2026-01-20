import pandas as pd
import transformers
from rich.console import Console
from rich.progress import Progress

try:
    from sklearn.metrics import precision_score, recall_score, f1_score
except ImportError as e:
    raise ImportError(
        "This feature requires optional dependencies. "
        "Install with: artifex[evaluation]"
    ) from e
    
from artifex.models import BaseModel


transformers.logging.set_verbosity_error()
console = Console()


class ModelEvaluator:
    def __init__(self, model: BaseModel, dataset_path: str) -> None:
        self.model = model
        self.df = pd.read_parquet(dataset_path)
        self.df = self.df[:10]
    
    def run(self):
        prediction_dicts = []
        
        with Progress() as progress:
            task = progress.add_task("Evaluating model...", total=self.df.shape[0])
            
            for row_index, db_row in self.df.iterrows():
                prediction = self.model(db_row["text"])
                prediction_label = [x.label for x in prediction][0]
                prediction_dicts.append({
                    "prediction": prediction_label,
                    "label": db_row.label,
                    "text": db_row.text,
                    "is_correct": prediction_label == db_row.label,
                })
                progress.update(task, advance=1)

        tagged_df = pd.DataFrame(prediction_dicts)

        # Calculate metrics
        precision = precision_score(
            tagged_df["label"], tagged_df["prediction"], pos_label="spam"
        )
        recall = recall_score(tagged_df["label"], tagged_df["prediction"], pos_label="spam")
        f1 = f1_score(tagged_df["label"], tagged_df["prediction"], pos_label="spam")

        console.print(
            f"\n📝 Evaluation summary:\n----------\n- Precision: {precision}\n- Recall: {recall}\n- F1: {f1}"
        )
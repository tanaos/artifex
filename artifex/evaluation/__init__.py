import pandas as pd
import os
import sys
from rich.progress import Progress

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
        self.df = self.df[:20]
    
    def run(self):
        # Run pipeline and align predictions / labels
        prediction_dicts = []
        
        with Progress(refresh_per_second=10) as progress:
            task = progress.add_task("Evaluating model...", total=self.df.shape[0])
            
            for row_index, db_row in self.df.iterrows():
                try:
                    # Use os-level file descriptor redirection to suppress output
                    # Save current stdout/stderr file descriptors
                    stdout_fd = sys.stdout.fileno()
                    stderr_fd = sys.stderr.fileno()
                    
                    # Save copies of the original stdout/stderr
                    stdout_copy = os.dup(stdout_fd)
                    stderr_copy = os.dup(stderr_fd)
                    
                    # Create a temporary file to redirect to /dev/null
                    devnull_fd = os.open(os.devnull, os.O_WRONLY)
                    
                    # Redirect stdout and stderr to /dev/null at the file descriptor level
                    os.dup2(devnull_fd, stdout_fd)
                    os.dup2(devnull_fd, stderr_fd)
                    
                    prediction = self.model(db_row["text"])
                    
                    # Restore original stdout/stderr
                    os.dup2(stdout_copy, stdout_fd)
                    os.dup2(stderr_copy, stderr_fd)
                    
                    # Close the duplicated and devnull file descriptors
                    os.close(stdout_copy)
                    os.close(stderr_copy)
                    os.close(devnull_fd)
                    
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
                
                progress.update(task, advance=1)
                progress.refresh()

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
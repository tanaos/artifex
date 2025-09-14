from transformers import Trainer, TrainerState, TrainingArguments, TrainerCallback, TrainerControl # type: ignore
from transformers.trainer_utils import TrainOutput
from typing import Any, Dict
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

"""
Patches for HuggingFace classes to improve user experience.
"""

console = Console()

class SilentTrainer(Trainer):
    """
    A regular transformers.Trainer which prevents the tedious final training summary dictionary
    from being printed to the console (since as of now, there is no built-in way to disable it).
    """
    
    def train(self, *args: Any, **kwargs: Any) -> TrainOutput:
        import builtins
        orig_print = builtins.print

        def silent_print(*a: Any, **k: Any) -> None:
            # Only suppress the summary dictionary
            if (
                len(a) == 1
                and isinstance(a[0], dict)
                and "train_runtime" in a[0]
            ):
                return
            return orig_print(*a, **k)

        builtins.print = silent_print
        try:
            return super().train(*args, **kwargs) # type: ignore
        finally:
            builtins.print = orig_print
            
            
class RichProgressCallback(TrainerCallback):
    """
    A custom TrainerCallback that uses Rich to display a progress bar during training.
    """
    
    progress: Progress
    task: int

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Called at the beginning of training.
        """
        
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            transient=True
        )
        self.task = self.progress.add_task("Training model...", total=state.max_steps)
        self.progress.start()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Called at the end of each training step.
        """
        
        self.progress.update(self.task, completed=state.global_step) # type: ignore

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        Called at the end of training.
        """
        
        self.progress.stop()
        console.print("[green]✔ Training model[/green]")
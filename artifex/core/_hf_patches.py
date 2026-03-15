from transformers import Trainer, TrainerState, TrainingArguments, TrainerCallback, TrainerControl, \
    Seq2SeqTrainer
from transformers.trainer_utils import TrainOutput
from typing import Any, Dict, Optional
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskID
from rich.console import Console
import cognitor
import time

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
            return super().train(*args, **kwargs)
        finally:
            builtins.print = orig_print


class SilentSeq2SeqTrainer(Seq2SeqTrainer):
    """
    A regular transformers.Seq2SeqTrainer which prevents the tedious final training summary
    dictionary from being printed to the console.
    """

    def train(self, *args: Any, **kwargs: Any) -> TrainOutput:
        import builtins
        orig_print = builtins.print

        def silent_print(*a: Any, **k: Any) -> None:
            if (
                len(a) == 1
                and isinstance(a[0], dict)
                and "train_runtime" in a[0]
            ):
                return
            return orig_print(*a, **k)

        builtins.print = silent_print
        try:
            return super().train(*args, **kwargs)
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
        
        self.progress.update(TaskID(self.task), completed=state.global_step)

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


class CognitorTrainingCallback(TrainerCallback):
    """
    A TrainerCallback that records per-step and per-epoch training metrics using Cognitor.
    """

    def __init__(self, cognitor_instance: cognitor.Cognitor) -> None:
        self._cognitor = cognitor_instance
        self._step_start_time: Optional[float] = None
        self._last_step_duration: Optional[float] = None

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any]
    ) -> None:
        self._step_start_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any]
    ) -> None:
        if self._step_start_time is not None:
            self._last_step_duration = time.time() - self._step_start_time
            self._step_start_time = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        if logs is None or "loss" not in logs:
            return

        samples_per_second: Optional[float] = None
        if self._last_step_duration and self._last_step_duration > 0:
            effective_batch_size = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
            samples_per_second = effective_batch_size / self._last_step_duration

        with self._cognitor.train() as t:
            t.epoch = int(state.epoch) if state.epoch is not None else None
            t.step = state.global_step
            t.train_loss = logs.get("loss")
            t.learning_rate = logs.get("learning_rate")
            t.gradient_norm = logs.get("grad_norm")
            t.samples_per_second = samples_per_second

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        if metrics is None:
            return
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return
        with self._cognitor.train() as t:
            t.epoch = int(state.epoch) if state.epoch is not None else None
            t.step = state.global_step
            t.val_loss = val_loss
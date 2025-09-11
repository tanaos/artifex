from rich.console import Console

console = Console()

with console.status("Initializing Artifex..."):
    from synthex import Synthex
    from typing import Optional
    from transformers import logging as hf_logging
    import datasets # type: ignore
    
    from .core import auto_validate_methods
    from .models import Guardrail, IntentClassifier
    from .config import config
console.print(f"[green]✔ Initializing Artifex[/green]")
    

if config.DEFAULT_HUGGINGFACE_LOGGING_LEVEL.lower() == "error":
    hf_logging.set_verbosity_error()

# Disable the progress bar from the datasets library, as it interferes with rich's progress bar.
datasets.disable_progress_bar()

    
@auto_validate_methods
class Artifex:
    """
    Artifex is a library for easily training small, private AI models without data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes Artifex with an API key for authentication.
        Args:
            api_key (Optional[str]): The API key to use for authentication. If not provided, attempts to load 
                it from the .env file.
        """
        
        if not api_key:
            api_key=config.API_KEY
        self._synthex_client = Synthex(api_key=api_key)
        self._guardrail = None
        self._intent_classifier = None

    @property
    def guardrail(self) -> Guardrail:
        """
        Lazy loads the Guardrail instance.
        Returns:
            Guardrail: An instance of the Guardrail class.
        """
        
        if self._guardrail is None:
            with console.status("Loading Guardrail model..."):
                self._guardrail = Guardrail(synthex=self._synthex_client)
        return self._guardrail
    
    @property
    def intent_classifier(self) -> IntentClassifier:
        """
        Lazy loads the IntentClassifier instance.
        Returns:
            IntentClassifier: An instance of the IntentClassifier class.
        """
        
        if self._intent_classifier is None:
            with console.status("Loading Intent Classifier model..."):
                self._intent_classifier = IntentClassifier(synthex=self._synthex_client)
        return self._intent_classifier
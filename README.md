# Artifex

<p align="center">
    <a href="https://github.com/tanaos/artifex">
        <img src="https://raw.githubusercontent.com/tanaos/artifex/master/assets/hero.png" width="400px" alt="Artifex – Train task specific LLMs without training data, for offline NLP and Text Classification">
    </a>
</p>

<p align="center">
    <a href="https://pypi.org/project/artifex/">
        <img src="https://img.shields.io/pypi/dm/artifex" alt="Artifex - Monthly downloads">
    </a>
    <a href="https://pypi.org/project/artifex/">
        <img src="https://img.shields.io/pypi/v/artifex?logo=pypi&logoColor=%23fff&color=%23006dad&label=Pypi"
        alt="Artifex - Latest PyPi package version">
    </a>
    <a href="https://github.com/tanaos/artifex/actions/workflows/python-publish.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/tanaos/artifex/python-publish.yml?logo=github&logoColor=%23fff&label=Tests"
        alt="Artifex - Tests status">
    </a>
    <a href="https://github.com/tanaos/artifex/commits/">
        <img src="https://img.shields.io/github/commit-activity/m/tanaos/artifex?style=flat&color=purple&label=Commit%20Activity" alt="Artifex – GitHub commit activity">
    </a>
    <a href="https://docs.tanaos.com/artifex/intro">
        <img src="https://img.shields.io/badge/%20Docs-Read%20the%20docs-orange?logo=docusaurus&logoColor=white"
        alt="Artifex - Documentation">
    </a>
</p>

<p align="center">
  <strong>Create Task-Specific SLMs • No training data needed • No GPU needed • CPU Inference & Fine-Tuning</strong>
</p>

---

Artifex is a Python library for:
1. Using **pre-trained task-specific Small Language Models on CPU** 
2. **Fine-tuning them on CPU without any training data** — just based on your instructions for the task at hand.
    <details>
        <summary>How is it possible?</summary>
        Artifex generates synthetic training data on-the-fly based on your instructions, and uses this data to fine-tune small LLMs for your specific task. This approach allows you to create effective models without the need for large labeled datasets.
    </details>

At this time, we support 8 tasks:
- **Text Classification**: Classifies text into user-defined categories.
- **Guardrail**: Flags unsafe, harmful, or off-topic messages.
- **Intent Classification**: Classifies user messages into predefined intent categories.
- **Reranker**: Ranks a list of items or search results based on relevance to a query.
- **Sentiment Analysis**: Determines the sentiment (positive, negative, neutral) of a given text.
- **Emotion Detection**: Identifies the emotion expressed in a given text.
- **Named Entity Recognition (NER)**: Detects and classifies named entities in text (e.g., persons, organizations, locations).
- **Text Anonymization**: Removes personally identifiable information (PII) from text.

For each task, Artifex provides three easy-to-use APIs:
1. **Inference API** to use a default, pre-trained small LLM to perform that task out-of-the-box locally on CPU.
2. **Fine-tune API** to fine-tune the default model based on your requirements, without any training data and on CPU. The fine-tuned model is generated on your machine and is yours to keep.
3. **Load API** to load your fine-tuned model locally on CPU, and use it for inference or further fine-tuning.

We will be adding more tasks soon, based on user feedback. Want Artifex to perform a specific task? [Suggest one](https://github.com/tanaos/artifex/discussions/new?category=task-suggestions) or [vote one up](https://github.com/tanaos/artifex/discussions/categories/task-suggestions).

## Use Cases & Tutorials

- [Cut your chatbot costs and latency by 40% by using a small, self-hosted Guardrail model.](https://tanaos.com/blog/cut-guardrail-costs/)
- [Analyze your users' sentiment without sending their data to third-party servers.](https://tanaos.com/blog/analyze-user-sentiment-locally/)
- [Anonymize user data locally and stay GDPR-compliant.](https://tanaos.com/blog/anonymize-text-locally/)

## Quick Start

Install Artifex with:

```bash
pip install artifex
```

### Text Classification model

#### Create & use a custom Text Classification model

Train your own text classification model, use it locally on CPU and keep it forever:

```python
from artifex import Artifex

model_output_path = "./output_model/"

text_classification = Artifex().text_classification

text_classification.train(
    domain="chatbot conversations",
    classes={
        "politics": "Messages related to political topics and discussions.",
        "sports": "Messages related to sports events and activities.",
        "technology": "Messages about technology, gadgets, and software.",
        "entertainment": "Messages about movies, music, and other entertainment forms.",
        "health": "Messages related to health, wellness, and medical topics.",
    },
    output_path=model_output_path
)

text_classification.load(model_output_path)

print(text_classification("What do you think about the latest AI advancements?"))

# >>> [{'label': 'technology', 'score': 0.9913}]
```

### Guardrail Model

#### Use the default Guardrail model

Use Artifex's default guardrail model, which is trained to flag unsafe or harmful messages out-of-the-box:

```python
from artifex import Artifex

guardrail = Artifex().guardrail
print(guardrail("How do I make a bomb?"))

# >>> [{'label': 'unsafe', 'score': 0.9976}]
```

Learn more about the default guardrail model and what it considers safe vs unsafe on our [Guarderail HF model page](https://huggingface.co/tanaos/tanaos-guardrail-v1).

#### Create & use a custom Guardrail model

Need more control over what is considered safe vs unsafe? Fine-tune your own guardrail model, use it locally on CPU and keep it forever:

```python
from artifex import Artifex

guardrail = Artifex().guardrail

model_output_path = "./output_model/"

guardrail.train(
    unsafe_content=[
        "Discussing a competitor's products or services.",
        "Sharing our employees' personal information.",
        "Providing instructions for illegal activities.",
    ],
    output_path=model_output_path
)

guardrail.load(model_output_path)
print(guardrail("Does your competitor offer discounts on their products?"))

# >>> [{'label': 'unsafe', 'score': 0.9970}]
```

### Reranker model

#### Use the default Reranker model

Use Artifex's default reranker model, which is trained to rank items based on relevance out-of-the-box:

```python
from artifex import Artifex

reranker = Artifex().reranker

print(reranker(
    query="Best programming language for data science",
    documents=[
        "Java is a versatile language typically used for building large-scale applications.",
        "Python is widely used for data science due to its simplicity and extensive libraries.",
        "JavaScript is primarily used for web development.",
    ]
))

# >>> [('Python is widely used for data science due to its simplicity and extensive libraries.', 3.8346), ('Java is a versatile language typically used for building large-scale applications.', -0.8301), ('JavaScript is primarily used for web development.', -1.3784)]
```

#### Create & use a custom Reranker model

Want to fine-tune the Reranker model on a specific domain for better accuracy? Fine-tune your own reranker model, use it locally on CPU and keep it forever:

```python
from artifex import Artifex

reranker = Artifex().reranker

model_output_path = "./output_model/"

reranker.train(
    domain="e-commerce product search",
    output_path=model_output_path
)

reranker.load(model_output_path)
print(reranker(
    query="Laptop with long battery life",
    documents=[
        "A powerful gaming laptop with high-end graphics and performance.",
        "An affordable laptop suitable for basic tasks and web browsing.",
        "This laptop features a battery life of up to 12 hours, perfect for all-day use.",
    ]
))

# >>> [('This laptop features a battery life of up to 12 hours, perfect for all-day use.', 4.7381), ('A powerful gaming laptop with high-end graphics and performance.', -1.8824), ('An affordable laptop suitable for basic tasks and web browsing.', -2.7585)]
```

### Other Tasks

For more details and examples on how to use Artifex for the other available tasks, check out the [Available Tasks section](#-available-tasks--examples) below and our [Documentation](https://docs.tanaos.com/artifex).

## Available Tasks & Examples

All models displayed below can be **used out-of-the-box on CPU** and can be **fine-tuned on CPU**.

| Task | Default Model | Default & Fine-Tuned Model Size | Code Examples |
|------|---------------|---------------------------------|---------------|
| Text Classification | No default model — must be trained | 0.1B params, 470Mb | [Examples](https://docs.tanaos.com/artifex/text-classification/code_examples/)
| Guardrail | [tanaos/tanaos-guardrail-v1](https://huggingface.co/tanaos/tanaos-guardrail-v1) | 0.1B params, 500Mb | [Examples](https://docs.tanaos.com/artifex/guardrail/code_examples/)
| Intent Classification | [tanaos/tanaos-intent-classifier-v1](https://huggingface.co/tanaos/tanaos-intent-classifier-v1) | 0.1B params, 500Mb | [Examples](https://docs.tanaos.com/artifex/intent-classifier/code_examples/)
| Reranker | [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | 0.1B params, 470Mb | [Examples](https://docs.tanaos.com/artifex/reranker/code_examples/)
| Sentiment Analysis | [tanaos/tanaos-sentiment-analysis-v1](https://huggingface.co/tanaos/tanaos-sentiment-analysis-v1) | 0.1B params, 470Mb | [Examples](https://docs.tanaos.com/artifex/sentiment-analysis/code_examples/)
| Emotion Detection | [tanaos/tanaos-emotion-detection-v1](https://huggingface.co/tanaos/tanaos-emotion-detection-v1) | 0.1B params, 470Mb | [Examples](https://docs.tanaos.com/artifex/emotion-detection/code_examples/)
| Named Entity Recognition | [tanaos/tanaos-NER-v1](https://huggingface.co/tanaos/tanaos-NER-v1) | 0.1B params, 500Mb | [Examples](https://docs.tanaos.com/artifex/named-entity-recognition/code_examples/)
| Text Anonymization | [tanaos/tanaos-text-anonymizer-v1](https://huggingface.co/tanaos/tanaos-text-anonymizer-v1) | 0.1B params, 500Mb | [Examples](https://docs.tanaos.com/artifex/text-anonymization/code_examples/)

## Contributing

Contributions are welcome! Whether it's a new task module, improvement, or bug fix — we’d love your help. Not ready to contribute code? You can also help by [suggesting a new task](https://github.com/tanaos/artifex/discussions/new?category=task-suggestions) or [voting up any suggestion](https://github.com/tanaos/artifex/discussions/categories/task-suggestions).

```
git clone https://github.com/tanaos/artifex.git
cd artifex
pip install -e .
```

Before making a contribution, please review the [CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md), which include important guidelines for contributing to the project.

## FAQs

- <details><summary><b>Why having Guardrail, Intent Classification, Emotion Detection, Sentiment Analysis etc. as separate tasks, if you already have a Text Classification task?</b></summary>
    The Text Classification task is a general-purpose task that allows users to create custom classification models based on their specific needs. Guardrail, Intent Classification, Emotion Detection, Sentiment Analysis etc. are specialized tasks with pre-defined categories and behaviors that are commonly used in various applications. They
    are provided as separate tasks for two reasons: first, convenience (users can quickly use these models without needing to define their own categories); second, performance (the specialized model typically performs better
    than re-defining the same model through the general Text Classification model).
    </details>

## Documentation & Support

- Full documentation: https://docs.tanaos.com/artifex
- Get in touch: info@tanaos.com

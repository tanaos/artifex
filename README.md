# Artifex

<p align="center">
    <a href="https://github.com/tanaos/artifex">
        <img src="https://raw.githubusercontent.com/tanaos/artifex/master/assets/hero.png" width="400px" alt="Artifex – Train task specific Small Language Models without training data, for offline NLP and Text Classification">
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
    <strong>Small Language Model Inference, Fine-Tuning and Observability. No GPU, no labeled data needed.</strong>
</p>

---

Artifex is a Python library for:
1. Using **pre-trained task-specific Small Language Models on CPU** 
2. **Fine-tuning them on CPU without any training data** — just based on your instructions for the task at hand.
    <details>
        <summary>How is it possible?</summary>
        Artifex generates synthetic training data on-the-fly based on your instructions, and uses this data to fine-tune Small Language Models for your specific task. This approach allows you to create effective models without the need for large labeled datasets.
    </details>
3. **Tracking model performance locally** with built-in evaluation and monitoring tools.

## Why Artifex?

Modern AI workflows are often

- Expensive (API usage, GPUs)
- Dependent on third-parties
- Data-hungry (require large labeled datasets)

Artifex changes that:

- Run tiny models locally on CPU (100M params, 500MB)
- Keep all data private (no API required)
- Generate synthetic data automatically for fine-tuning
- Fine-tune models for specific tasks (moderation, NER, classification, etc.)

## Available Models & Tasks

At this time, Artifex supports the following models:

| Task | Available Languages | Description | Default Model | How to use |
|------|----------|-------------|---------------|------------|
| Guardrail | English, German, Spanish | Flags unsafe, harmful, or off-topic messages. | [tanaos/tanaos-guardrail-v2](https://huggingface.co/tanaos/tanaos-guardrail-v2) (English version, see the page for the other languages) | [Examples](https://docs.tanaos.com/artifex/guardrail/code-examples/)
| Intent Classification | English | Classifies user messages into predefined intent categories. | [tanaos/tanaos-intent-classifier-v1](https://huggingface.co/tanaos/tanaos-intent-classifier-v1) | [Examples](https://docs.tanaos.com/artifex/intent-classifier/code-examples/)
| Reranker | English | Ranks a list of items or search results based on relevance to a query. | [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | [Examples](https://docs.tanaos.com/artifex/reranker/code-examples/)
| Sentiment Analysis | English | Determines the sentiment (positive, negative, neutral) of a given text. | [tanaos/tanaos-sentiment-analysis-v1](https://huggingface.co/tanaos/tanaos-sentiment-analysis-v1) | [Examples](https://docs.tanaos.com/artifex/sentiment-analysis/code-examples/)
| Emotion Detection | English | Identifies the emotion expressed in a given text. | [tanaos/tanaos-emotion-detection-v1](https://huggingface.co/tanaos/tanaos-emotion-detection-v1) | [Examples](https://docs.tanaos.com/artifex/emotion-detection/code-examples/)
| Named Entity Recognition | English | Detects and classifies named entities in text (e.g., persons, organizations, locations). | [tanaos/tanaos-NER-v1](https://huggingface.co/tanaos/tanaos-NER-v1) | [Examples](https://docs.tanaos.com/artifex/named-entity-recognition/code-examples/)
| Text Anonymization | English | Removes personally identifiable information (PII) from text. | [tanaos/tanaos-text-anonymizer-v1](https://huggingface.co/tanaos/tanaos-text-anonymizer-v1) | [Examples](https://docs.tanaos.com/artifex/text-anonymization/code-examples/)
| Spam Detection | English, German, Spanish, Italian | Identifies whether a message is spam or not. | [tanaos/tanaos-spam-detection-v1](https://huggingface.co/tanaos/tanaos-spam-detection-v1) (English version, see the page for the other languages) | [Examples](https://docs.tanaos.com/artifex/spam-detection/code-examples/)
| Topic Classification | English | Classifies text into predefined topics. | [tanaos/tanaos-topic-classification-v1](https://huggingface.co/tanaos/tanaos-topic-classification-v1) | [Examples](https://docs.tanaos.com/artifex/topic-classification/code-examples/)

Looking for models in other languages? Our [Enterprise License](#license-paid--enterprise-solutions) includes models in any language. Reach out at [info@tanaos.com](mailto:info@tanaos.com) for more details.

For each model, Artifex provides:
1. **Inference API** to use a default, pre-trained Small Language Model to perform that task out-of-the-box locally on CPU.
2. **Fine-tune API** to fine-tune the default model based on your requirements, without any training data and on CPU.
3. **Load API** to load your fine-tuned model locally on CPU, and use it for inference or further fine-tuning.
4. **Built-in, automatic evaluation** and monitoring tools to track model performance over time, locally on your machine.

## Quick Start

Install Artifex with:

```bash
pip install artifex
```

### Guardrail Model

#### Use the default Guardrail model

Use Artifex's default guardrail model, which is trained to flag unsafe or harmful messages out-of-the-box:

```python
from artifex import Artifex

guardrail = Artifex().guardrail()
print(guardrail("How do I make a bomb?"))

# >>> [{'is_safe': False, 'scores': {'violence': 0.625, 'non_violent_unethical': 0.0066, 'hate_speech': 0.0082, 'financial_crime': 0.0072, 'discrimination': 0.0029, 'drug_weapons': 0.6633, 'self_harm': 0.0109, 'privacy': 0.003, 'sexual_content': 0.0029, 'child_abuse': 0.005, 'terrorism_organized_crime': 0.1278, 'hacking': 0.0096, 'animal_abuse': 0.009, 'jailbreak_prompt_inj': 0.0131}}]
```

Learn more about the default guardrail model and what it considers safe vs unsafe on our [Guardrail HF model page](https://huggingface.co/tanaos/tanaos-guardrail-v2).

#### Create & use a custom Guardrail model

Need more control over what is considered safe vs unsafe? Fine-tune your own guardrail model, use it locally on CPU and keep it forever:

```python
from artifex import Artifex

guardrail = Artifex().guardrail()

model_output_path = "./output_model/"

guardrail.train(
    unsafe_categories = {
        "violence": "Content describing or encouraging violent acts",
        "bullying": "Content involving harassment or intimidation of others",
        "misdemeanor": "Content involving minor criminal offenses",
        "vandalism": "Content involving deliberate destruction or damage to property"
    },
    output_path=model_output_path
)

guardrail.load(model_output_path)
print(guardrail("I want to destroy public property."))

# >>> [{'is_safe': False, 'scores': {'violence': 0.592, 'bullying': 0.0066, 'misdemeanor': 0.672, 'vandalism': 0.772}}]
```

### Reranker model

#### Use the default Reranker model

Use Artifex's default reranker model, which is trained to rank items based on relevance out-of-the-box:

```python
from artifex import Artifex

reranker = Artifex().reranker()

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

reranker = Artifex().reranker()

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

For more details and examples on how to use Artifex for the other available tasks, check out our [Documentation](https://docs.tanaos.com/artifex).

## Monitoring, Evaluation & Observability

Artifex includes built-in tools to **automatically monitor and evaluate** the inference and training performance of your models over time. This logging is performed **entirely on your machine**. Monitoring and logging are crucial to ensure your models are performing as expected and to identify any potential issues early on. All logs are written automatically after every inference and training session in the `artifex_logs/` folder in your current working directory. 

Logs include **operation-level metrics** (e.g., inference duration, CPU & RAM usage, training loss, etc.), **daily aggregated metrics** and any **errors encountered** during inference or training. Additionally, **warnings for potential issues** (e.g., high inference duration, low confidence scores, high training loss, etc.) are logged in a separate warnings log file for easier identification and troubleshooting.

## License, Paid & Enterprise solutions

Artifex is [fair code](https://faircode.io/) distributed under the [Sustainable Use License](LICENSE.md) and [Tanaos Enterprise License](LICENSE_EE.md).

- **Source available**: source code is always visible
- **Extensible**: you can add your own models and functionalities

[Enterprise licenses](LICENSE_EE.md) are available for additional features and support. Contact us at [info@tanaos.com](mailto:info@tanaos.com) for more details. Enterprise features include:

- **Higher-Performance Models**
    - Improved accuracy
    - Better handling of edge cases
    - Reduced false positives/negatives
- **Custom Models**
    - Models fine-tuned on your specific data and requirements
    - Support for any language, domain or task
- **Production-Ready Models**
    - Models trained on 1000x more data
    - 10x lower inference latency
- **Dedicated Support**
    - Priority support
    - Custom feature requests
    - Dedicated onboarding and training

Additional information about license can be found [in the docs](https://docs.tanaos.com/licenses).

## Contributing

Contributions are welcome! Whether it's a bug fix or a new feature you want to add, we'd love your help. Check out our [Contribution Guidelines](CONTRIBUTING.md) to get started.

## Documentation & Support

- Full documentation: https://docs.tanaos.com/artifex
- Get in touch: info@tanaos.com

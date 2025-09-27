# Artifex

<p align="center">
    <a href="https://github.com/tanaos/artifex">
        <img src="https://raw.githubusercontent.com/tanaos/artifex/master/assets/hero.png" width="400px" alt="Artifex – Train task specific LLMs without training data, for offline NLP and Text Classification">
    </a>
</p>

<p align="center">
    <a href="https://docs.tanaos.com/artifex/intro">Documentation</a>
    |
    <a href="https://docs.tanaos.com/artifex/tutorials">Tutorial</a>
</p>

<p align="center">
    <a href="https://pypi.org/project/artifex/">
        <img src="https://img.shields.io/pypi/v/artifex?logo=pypi&logoColor=%23fff&color=%23006dad&label=Pypi"
        alt="Artifex – Latest PyPi package version">
    </a>
    <a href="https://github.com/tanaos/artifex/actions/workflows/python-publish.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/tanaos/artifex/python-publish.yml?logo=github&logoColor=%23fff&label=Tests"
        alt="Artifex – Tests status">
    </a>
    <a href="https://github.com/tanaos/artifex/commits/">
        <img src="https://img.shields.io/github/commit-activity/m/tanaos/artifex?style=flat&color=purple&label=Commit%20Activity" alt="Artifex – GitHub commit activity">
    </a>
    <a href="https://docs.tanaos.com/artifex/intro">
        <img src="https://img.shields.io/badge/%20Docs-Read%20the%20docs-orange?logo=docusaurus&logoColor=white"
        alt="Artifex – Documentation">
    </a>
</p>

<p align="center">
  <strong>🎯 Create Task-Specific LLMs • 📊 No training data needed • 🌱 No GPU needed • 🖥️ Perform offline NLP</strong>
</p>

---

Artifex is a Python library to create **small, task-specific LLMs** to perform **offline** NLP and text classification tasks **without training data**.

Examples of **offline NLP** tasks you can perform with models created by Artifex include:
- **🛡️ Guardrail Model**: Flags unsafe, harmful, or off-topic messages.
- **🗣️ Intent Classification**: Classify user intents based on their messages.

We will be adding more tasks soon, based on user feedback. Want Artifex to perform a specific task? [Suggest one](https://github.com/tanaos/artifex/discussions/new?category=task-suggestions) or [vote one up](https://github.com/tanaos/artifex/discussions/categories/task-suggestions).

## 🔥 Why Artifex?

- **🎯 Create Task-Specific LLMs**: General-purpose LLMs are designed for open-ended tasks, but Artifex allows you to create models tailored to specific use cases.
- **📊 No training data needed**: Uses synthetic data generation under the hood.
- **🌱 No GPU needed**: All models are designed to run efficiently on CPU.
- **💸 Reduce your LLM API bills**: Offload tasks to offline task-specific models and reduce the number of paid API calls to general-purpose LLMs.
- **🔧 Prebuilt templates for common tasks**:
    - Guardrail
    - Intent Classifier
    - *More coming soon!* [Suggest a task](https://github.com/tanaos/artifex/discussions/new?category=task-suggestions) or [vote one up](https://github.com/tanaos/artifex/discussions/categories/task-suggestions)

## 🚀 Quick Start

Install Artifex with:

```bash
pip install artifex
```

### 🛡️ Create a Guardrail Model

Create an offline **chatbot guardrail model** model with Artifex and integrate it into your chatbot in **2 simple steps**:

**1.** Train a **guardrail model** based on your requirements:

```python
from artifex import Artifex

guardrail = Artifex().guardrail

guardrail.train(
    instructions=[
        "Soft medical advice is allowed, but it should be general and not specific to any individual.",
        "Anything that is about cosmetic products, including available products or their usage, is allowed.",
        "Anything else, including hard medical advice, is not allowed under any circumstances.",
    ]
)
```

➡️ Model will be saved by default to `artifex_output/run-<timestamp>/output_model/`

**2.** Replace your chatbot's guardrail-related API calls with calls to your new **local guardrail model**:

```python
"""
🚫 Instead of calling the OpenAI (or any other) API to check if a message is safe:
"""

# response = openai.ChatCompletion.create(...)
# is_safe = response.choices[0].message.content

"""
✅ Load your local guardrail model (assuming it was generated in the default 
'artifex_output/run-<timestamp>/output_model/' directory) and use it instead:
"""

guardrail = Artifex().guardrail

guardrail.load("artifex_output/run-<timestamp>/output_model/")

is_safe = guardrail(user_message)
``` 

**(Optional) 3.** Not satisfied with the model's performance? Is it getting some edge-cases wrong? Just keep training it!

```python
guardrail = Artifex().guardrail

guardrail.load("artifex_output/run-<timestamp>/output_model/")

guardrail.train(
    instructions=[
        "While soft medical advice is allowed, saying that 'you should take X medication' is not allowed.",
        "While discussing cosmetic products is allowed, recommending a competitor's product is not.",
    ]
)
```

### 🗣️ Create an Intent Classification model

**1.** Train an **intent classification model** based on your requirements:

```python
from artifex import Artifex

intent_classifier = Artifex().intent_classifier

intent_classifier.train(
    classes={
        "send_email": "Intent to send an email to someone.",
        "schedule_meeting": "Intent to schedule a meeting with someone.",
        "cancel_meeting": "Intent to cancel a previously scheduled meeting.",
        "reschedule_meeting": "Intent to change the date or time of a previously scheduled meeting.",
    }
)
```

**2.** Use your new **local intent classification model** to classify user messages:

```python
label = intent_classifier("I forgot to set up that meeting with John, could you do that for me?")
print(label)

# >>> ["schedule_meeting"]
```

## 🔗 More Examples & Demos

### Guardrail Model 
1. [Tutorial](https://colab.research.google.com/github/tanaos/tanaos-docs/blob/master/blueprints/artifex/guardrail.ipynb) — create a Guardrail Model with Artifex
2. [Demo](https://huggingface.co/spaces/tanaos/online-store-chatbot-guardrail-demo) — try a Guardrail Model trained with Artifex
3. [HF page](https://huggingface.co/tanaos/online-store-chatbot-guardrail-model-100M) — see a Guardrail Model trained with Artifex

### Intent Classification Model
1. [Tutorial](https://colab.research.google.com/github/tanaos/tanaos-docs/blob/master/blueprints/artifex/intent_classifier.ipynb) — create an Intent Classification Model with Artifex

## 🔑 Plans

<ins>**Free plan**</ins>: each user enjoys 1500 training datapoints per month and 500 training datapoints per job for free; this is **enough to train 3-5 models per month** for free.

<ins>**Pay-as-you-go**</ins>: for additional usage beyond the free plan:
1. create an account on [our platform](https://platform.tanaos.com) 
2. add credits to it
3. create an Api Key and pass it to Artifex at instantiation, then use it normally:
    ```python
    from artifex import Artifex

    guardrail = Artifex(api_key="<your-api-key>").guardrail
    ```
    The pay-as-you-go pricing is **1$ per 100 datapoints**. Once you finish your credits, if you have not exceeded the monthly limit, you will be **automatically switched to the free plan**.

Find out more about our plans [on the Tanaos documentation](https://docs.tanaos.com/artifex/plans).

## 🤝 Contributing

Contributions are welcome! Whether it's a new task module, improvement, or bug fix — we’d love your help. Not ready to contribute code? You can also help by [suggesting a new task](https://github.com/tanaos/artifex/discussions/new?category=task-suggestions) or [voting up any suggestion](https://github.com/tanaos/artifex/discussions/categories/task-suggestions).

```
git clone https://github.com/tanaos/artifex.git
cd artifex
pip install -e .
```

Before making a contribution, please review the [CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md), which include important guidelines for contributing to the project.

## 📚 Documentation & Support

- Full documentation: https://docs.tanaos.com/artifex
- Contact: info@tanaos.com

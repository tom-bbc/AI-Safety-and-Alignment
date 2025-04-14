# Intro to ML Safety

## Unit 2: Robustness

* **Robustness** = the ability of a system to handle adversarial attacks and endure once-in-a-century 'black swan' events.

### Adversarial Robustness

* *Imperceptible distorions* add carefully curated noise to an image/input to cause a model to generate different/incorrect outputs - modern models are robust against these attacks.
* *Perceptible distortions* add visible changes to an image/input that similarly throw off a model's outputs - models are not as robust to these changes.
* In future, if proxy models of human values are used to train agents, or detection models are used to identify undesirable agent behaviour, these must be adversarily robust such that they cannot be gamed/thrown off.

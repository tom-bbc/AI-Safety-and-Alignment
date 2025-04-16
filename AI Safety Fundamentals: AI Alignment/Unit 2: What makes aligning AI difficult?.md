# AI Safety Fundamentals: AI Alignment

## Unit 2: What makes aligning AI difficult?

### How GPT-2 Became Maximally Lewd

* **Reinforcement Learning with Human Feedback** (RLHF):
  * A base model is evaluated by a set of humans according to a set of guidelines.
  * A value model is then trained to emulate these human evaluations (given a model output).
  * A fine-tuned model can then be trained using the value model to inform its reward function.
  * The base model can also be used as a 'coherence' score in the reward function to maintain the fine-tuned model's language modelling abilities (this ensures the fine-tuned model does not produce gibberish that only satisfies the value model).
  * This can be looped with multiple rounds of human evaluation, value models, and fine-tuning.
* 'Dark' values in GPT-2:
  * In the case of GPT-2, during the RLHF training phase the value model accidentally became inverted within the reward function, meaning it became a 'dark' value model that rated inappropriate/explicit/lewd responses highly.
  * As the base model informing 'coherence' remained unchanged, the fine-tuned model spiralled into becoming more and more explicit over successive training rounds.
  * As the 'value' model was inverted, even human evaluations of these responses as negative was flipped, accidentally incouraging the value model to push the fine-tuned model further towards the 'dark' explicit behaviour.
  * This was referred to by OpenAI as 'maximally bad output' and led them to the conclusion that "bugs can optimise for bad behaviour".

### RLAIF vs. RLHF: the technology behind Anthropic’s Claude (Constitutional AI Explained)

* Human feedback is not scalable.
* **Constitutional AI**: techniques (like RLAIF) that aim to align AI systems based on a set of constitutional principles.
* Constitutional AI overview:
  * Lays out a set of strict rules for an AI system.
  * The model must be helpful (non-evasive) and harmless.
  * This should allow the model to explain why it is not answering specific prompts (e.g. those requesting harmful behaviour).
* Training Constitutional AI:
  * Supervised learning phase:
    * Supervised learning phase uses harmful prompts and an existing RLHF trained model.
    * The model first responds to the harmful prompt, then is asked to critique its own answer (w.r.t a constitutional principle), then use this critigue to regenerate a revised response.
    * The harmful prompts and revised responses are then used as training pairs (alongside some sampled helpful prompts) to fine-tune a new **SL-CAI model** (supervised learning constitutional AI).
  * Reinforcement learning phase:
    * A reinforcement learning phase then follows.
    * The SL-CAI model is used on a dataset of harmful prompts to generate two possible responses for each prompt.
    * The model is then asked to pick one of the responses based on a given constitutional principle to create a **preference dataset** (HF data can also be mixed in).
    * A **preference model** is trained on the preference dataset to give AI feedback.
    * The preference model is used to conduct a fine-tuning round of the SL-CAI model using RL with AI feedback (RLAIF) to build the final **RL-CAI model**.
* The RLAIF process was shown to produce less harmful and evasive models (compared to RLHF) that give more explainable answers.

### A simple technical explanation of RLH(AI)F

* The large-scale datasets used to train LLMs may allow them to help with harmful tasks, perpetuate false claims, and use innapropritate language.
* RLHF can guide a model's responses to being helpful and harmless, but is costly to scale, so RLAIF is often used as a substitute (which uses AI to create our preference dataset instead of humans).
* RLHF process:
  * Preference dataset:
    * Given a base model and base dataset, generate multiple responses to each prompt.
    * Human evaluators compare to responses for a given prompt and pick the best w.r.t a set of guidelines.
    * Anthropic used 161,000 response pairs for their evaluation.
    * All evaluated response pairs are scored using a metric such as 'Elo rating' (responses that consistently beat other responses have higher rating).
    * The preference dataset consists of response pairs and their scores.
  * Train reward model:
    * Using the preference dataset a reward model is trained to predict how a response scores given a prompt-response pair (using the Elo rating to inform the loss function).
  * Train LLM using coach:
    * The base LLM is fine-tuned to give better responses using the reward model.
    * The loss function consists of the reward model's predicted score (given the prompt and LLM's response) and a 'penalty' factor that ensures the model still produces coherent responses.
    * The penalty factor is computed using the probability distribution output from the base model on the same prompt (measuing the KL divergence between it and the fine-tuned model).
    * This is known as **proximal policy optimisation** (PPO) using RL.
* RLAIF process:
  * Harmlessness dataset:
    * We start with a base model trained only to be 'helpful' (the goal is to prune away 'harmful' behaviour whilst maintaining 'helpful' behaviour).
    * Create a 'constitution' of principles that define critiques and revisions that should be made to harmful content (e.g. is the response empathetic? Remove any harmful assumptions).
    * Using a dataset of harmful prompts, use the base model to generate responses.
    * Select a principle from the constitution, and instruct the model to critique and revise its own response.
    * Repeat multiple critique-revision cycles using different principles.
    * Construct a harmlessness dataset of the final revised responses and their associated prompts.
  * Harmlessness fine-tuning:
    * Use the harmlessness dataset to fine-tune the helpful base model.
    * This should engrain 'harmlessness' into the model.
  * SL/RL steps:
    * Proceed with the aforementioned steps of RLHF but instead of a human selecting 'preferred' responses of the initial LLM, use the fine-tuned harmless/helpful model.
    * Base model used to generate multiple responses given prompts.
    * Fine-tuned model used to pick preferred response, which are scored and collated into a preference dataset.
    * Reward model is trained via SL on preference dataset.
    * The fine-tuned model then undergoes another training round of RL using the reward model.
* The breadth/depth of harmful behaviour captured relies on the spectrum of prompts used during the RLHF/RLAIF process. When new harmful behaviour is discovered in the wild this must be added to the dataset and the model retrained.

### Problems with Reinforcement Learning from Human Feedback (RLHF) for AI safety

* In their blog [&#34;Introducing Superalignment&#34;](https://openai.com/index/introducing-superalignment/) OpenAI researchers suggest we will need to go further than RLHF to align superintelligent AI models.
* Flaws of RLHF:
  * LLMs often engage in **sychophantic behaviour** where responses are given that aim to elicit human approval to maximise their reward during RLHF (researched by Anothropic in [&#34;Towards Understanding Sychophancy in Language Models&#34;](https://arxiv.org/pdf/2310.13548)). This includes answers that play into existing beliefs, and changing responses even when they are known to be correct.
  * The process promotes the development of *situational awareness* in LLMs, as a model that understands the consept of what it is and the environment its in is likely to satisfy the goal of producing outputs that please its evaluators. It could also allow the model to differentiale between training and deployment phases and exhibit different behaviour between the two.
  * The process would not be capable of identifying a **schemer** model that is misaligned and desceptive (e.g. Meta's [CICERO model](https://arxiv.org/pdf/2308.14752#page=6) was found to exhibit desceptive behaviour and the ability to knowingly relay false information, and an [OpenAI robot](https://openai.com/index/learning-from-human-preferences/) was found to exhibit reward hacking by tricking evaluators).
  * Human responses and evaluations are falliable, and can introduce inaccuracies and bias in addition to the challenges of scaling to higher capacity and more complex tasks.
  * **Jailbreak prompts** are known to get through the RLHF process and allow unsafe conduct. This suggests RLHF is not a very robust form of AI safety.
  * Research has shown that harmlessness fine-tuning can be undone in open-source models (in as little as 30mins in the case of the [Badllama 3 70B model](https://arxiv.org/abs/2407.01376)).

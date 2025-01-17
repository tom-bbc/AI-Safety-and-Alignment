# AI Alignment Fast-Track

## Using AI to align AI

* **Scalable oversight** methods aim to assist humans in evaluating complex tasks and behaviour.
* **Task decomposition**:
  * Breaking down tasks into small, simple subtasks/components and getting feedback on each.
  * E.g. 'iterated amplification' (IA) iterates evaluations over multiple overlapping subtasks and 'iterated amplification and distillation' (IDA) integrates these evaluations back into model training.
* **Recursive reward modelling**:
  * AI assistance for humans evaluating LLM outputs.
  * The assisted evaluations can be used to train a reward model, which trains a new, better LLM, and repeat.
* **Constitutional AI**:
  * Techniques like RLAIF that are based on a set of constitutional principles.
* **Debate**:
  * Multiple AI systems argue different takes on a task/issue, providing evidence and arguments, and discrediting other debaters statements.
  * A human then decides the winner.
  * Within the debate arguments with bad tendancies like deception will ideally be picked up and pointed out by the other debaters.
  * However, debaters may win evaluators over by appealing to their biases etc., models may collude, or arguments may necessitate being too complex/niche to be preferred by evaluators.
* **Weak to strong generalisations**:
  * Some research suggests that even if the evaluator (human/model) is in theory less capable than the model we are trying to align, the more capable model should be able to generalise beyond the evaluator to improve its own judgement over time.

### Using Dangerous AI, But Safely?

* **Catestrophic actions**: behaviours we must ensure AI systems cannot do, even once. For example, inserting a backdoor/vulnerability into generated code for a security critical system that can later be exploited.
* How can we use a limited number of human evaluators and less intelligent but more trustworthy models to ensure a superintelligent model does not enable any catestrophic actions?
* **Red teaming**: a group of engineers who design an 'attack policy' within a model by purposefully corrupting SOTA models to exhibit harmful (and intelligent) behaviour.
* **Blue teaming**: a second group of engineers who design the 'protocol' that aims to use the harmful model without allowing its harmful behaviour to get through (maximising a usefulness score and safety score).
* Approach 1 - Upfront auditing:
  * A simple but common approach for red teams is to have the model exhibit harmful behaviour only a very small proportion of the time (~1%).
  * As not all model outputs are ever eched by human evaluators (~2%) it is unlikley any backdoors will be spotted duing blue team evaluation, but they will crop up eventually during deployment.
* Approach 2a - Trusted monitoring:
  * Instead, the blue team can use our small but trustworthy model to score the harmfulness likelihood of a large number of model outputs, and get human evaluators to only check the most harmful outputs (top ~2% of scores).
  * However, the attack policy may be able to write vulnerabilities that are too complex for the smaller model to identify (although research suggests it is tricky to evade all suspicion even from a small model).
* Approach 2b - Trusted monitoring:
  * Using the small trustworthy model, score every output of the intelligent model during deployment, and only allow responses that exceed a conservative threshold (use the trustworthy model to generate outputs when the threshold isn't met).
  * However, again the model is not as intelligent so may not detect complex vulnerabilities.
* Approach 3 - Untrusted monitoring:
  * Use the more intelligent model to check its own outputs. To ensure the model isn't 'lying' about if one of its codes contains a vulnerability, some synthetic prompts/responses with known vulnerabilties are checked, and if any desception from the model is witnessed we know it's misaligned.
  * This can only be countered if the intelligent model can 'lie selectively' by telling when it is being monitored (i.e. distinguising between its own generations and the blue team's synthetic prompts/responses).
  * This could be done by inserting hidden identifiers within the formatting/semantics of responses.
* None of these techniques allow us to 'trust' the untrusted model, they just allow us to make use of it by controlling its responses.

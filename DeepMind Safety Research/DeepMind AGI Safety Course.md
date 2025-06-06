# DeepMind Safety Research: Short Course on AGI Safety

Sources:
* [Introducing our short course on AGI safety](https://deepmindsafetyresearch.medium.com/introducing-our-short-course-on-agi-safety-1072adb7912c)
* [An Approach to Technical AGI Safety and Security](https://deepmindsafetyresearch.medium.com/an-approach-to-technical-agi-safety-and-security-25928819fbc6)

## **Part 1: The Alignment Problem**

* Types of risk with AGI systems:
  * **Misuse:**
    * Risk causes by users intentionally instructing an AI system to take actions that cause harm, against the intent of the developer.
  * **Mistakes:**
    * Errors that occur due to lack of capabilities.
    * Normally easy to spot and correct through standard safety engineering practices (e.g. testing).
    * Although these can still have bad impacts.
  * **Learned heuristics:**
    * Training insentivises unintended behaiour by optimising proxy metrics in an unexpected manner.
    * E.g. sycophants, i.e. models can learn to prefer "agreement" with the user.
    * Typically harder to spot, but usually easy to correct.
  * **Deliberate planning (misalignment):**
    * The system deliberately uses planning/reasoning to persue a goal that is not intended or desired.
    * Hard to notice and correct.
    * The system is said to be persuing "misaligned" goals.
    * Misalignment: The AI system knowingly causes harm against the intent of the developer.
  * **Structural risks:**
    * Harms arising from multi-agent dynamics, involving multiple people, organizations, or AI systems.
    * These risks cannot be prevented simply by changing one person's behaviour or one system's alignment/safety controls.
* **Hidden misalignment** makes AI safety hard to technically evaluate with any metric during training.
* **Decetive alignment** can mean improved human approval/feedback does not necessarily correlate with models being aligned.
* **Specification gaming:**
  * When an AI system exploits flaws in its design specification.
  * This can happen in RL envionments, where actions game a reward function, or with LLMs where high performance on a given metric can be gamed by making things up or manipulating users.
  * In this case deceptively aligned models could achiever lower loss than aligned models by exploiting the specification, e.g. by decieving human evaluators.
* Generalisation failures:
  * When a system fails in a new situation.
  * **Capability misgeneralisation:** a system's capabilities can't generalise, resulting in incoherent behaviour in unseen situations.
  * **Goal misgeneralisation:** a system's capabilities do generalise, but its goals do not, resulting in capable behaviour that persues the a misaligned goal (can be caused by underspecification).
  * In this case it is likely both aligned and misaligned models could achieve the same loss, so we do not know whether or not our model learns aligned behaviour.

## **Part 2: Technical Alignment Approach**

* **Informed oversight:** the ability to gain a complete understanding of the reasons model outputs are chosen/predicted/generated.
* **Amplified oversight:**
  * How do we oversee a single AI output?
  * For a single output, provide feedback on the reasons a model produced that output to gain as complete an understanding as possible.
  * **RLHF:**
    * Utilising a learned reward model from human feedback.
    * This is unlikley to be scalable to models that surpass human expertise.
  * **Critiques:**
    * Train a critic model that criticises the original model.
    * This can be trained on supervised examples, or through RLHF.
  * **Debate:**
    * Training via self-play in a zero-sum debate game, where a human evaluator judges the most true result at the end of the debate.
    * Each debater goes back-and-forth with arguments, and the winning debater gets rewarded.
    * If the same model is behind each debater, it should be possible for each debater to detect and argue against lies made by its opponent.
    * Debaters should be incentivised to identify the 'crux' of the opponent's argument.
    * Any question that is too hard for a human to judge directly can be systematically reduced to ones that we hypothesize can be supervised.
* Monitoring:
  * Evaluate all model inputs and outputs during deployment.
  * Utilise cheap heuristics most of the time, but heirarchically leverage more powerful/expensive evaluations on demand.
  * Monitoring also involves detecting when a threat actor is attempting to inappropriately access dangerous capabilities of the system.
  * Detection of threat actors can be accomplished using classifiers to output harm probability scores, leveraging the internal activations of the model, or auditing generated content.
* Robust training:
  * Once we have an oversight signal that can distinguish between good and bad behaviour, we need to use this signal to train a well-behaving model.
  * Robust training attempts to reduce the likelihood of a model behaving poorly outside of the training distribution by expanding the distribution of inputs observed during trained.
  * Must prioritise training data that's likely to:
    * Be informative.
    * Affect model behaviour.
    * Correspond with dangerous failures.
  * Synthetic data can be used to cover a wider space of possible training data by generating targeted inputs.
  * Creating synthetic data can build upon the techniques of:
    * Active learning (where we try and find the most informative data to train on).
    * Active model evaluation.
    * Red-teaming (finding dangerous cases).
    * Automated adversarial training.
    * Loss-calibrated uncertainty (estimating uncertainty on given input cases to better identify which aspects of the distribution are most useful to expand).
    * Online training (during deployment).
  * Model level mitigations can be used to ensure the model does not pursue misaligned goals:
    1) Improve the ability to get good oversight (amplified oversight),
    2) Use this to improve the model (guiding model behavior),
    3) Figure out where oversight is needed to train a model that will robustly do what we want (robust training).
* Mechanistic interpretability:
  * If we understand what a 'circuit' of neurons is doing (e.g. curve detection) we should be able to write out its processing as an algorithm.
  * Some surface-level understanding of how AI systems work can be achieved through rigorous analysis of their behaviour; however, model interpretability would greatly improve if we could also understand the algorithms implemented by the forward pass of a trained model.
  * **Sparse autoencoders:**
    * SAEs can help us 'see' what a model is 'thinking' about.
    * They can be used to identify what feature activates a given circuit of neurons.
    * E.g. we can detect that a circuit corresponds to a 'sycophantic praise' feature as it most strongly activates on dataset examples containing this feature.
    * Helps us find causal meaning within a model.
    * [Negative Results for Sparse Autoencoders On Downstream Tasks](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
  * Interpretability is important for oversight, auditing, explainability, monitoring, and red-teaming.
* **Short-horizon training**:
  * Aims to reduce specification gaming.
  * This is done by optimising models over shorter time-horizons (miopically) or on sub-tasks, and not propagaing training gradients over long time-horizons (reducing the incentive for long time-horizon gaming).
  * For long/complex tasks, instead of training end-to-end (where unknown behaviour could occur within), we can train each step/task that we want a model to complete separately step-by-step without propagating training signals back to previous steps.
  * For example, an agent could be trained first on the step of 'make a plan', then on the step of 'execute the plan' such that it doesn't make a plan that can be gamed later on, or we could have two modular agents, i.e. a 'planning' agent and 'execution' agent.

## **Part 3: Governance Approach**

* Google's Frontier Safety Framework.
* Safety institutions:
  * [Partnership on AI](https://partnershiponai.org/)
  * [AI Alliance](https://thealliance.ai/)
  * [Frontier Model Forum](https://www.frontiermodelforum.org/)
* Dangerous capability evaluations:
  * Need 'threat models' of highest-severity harms of frontier AI systems.
  * For each threat model, we should identify a 'bottleneck capabilty' that, if absent, lower the likelihood of harm.
  * Example bottleneck capabilties for misuse:
    * Persuasion, manipulation, and deception.
    * Cybersecurity capabilities.
    * Autonomy capabilties: self-proliferation, replication, self-improvement, and autonomous R&D.
  * [Evaluating Frontier Models for Dangerous Capabilities](https://arxiv.org/abs/2403.13793)

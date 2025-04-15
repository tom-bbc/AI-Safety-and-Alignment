# AI Alignment Fast-Track

## Final summary of course learnings

#### Key Definitions

* **Alignment**: an attempt to *align* the objectives and behaviour of AI systems with the intentions and values of its creators. Anthropic defines this as ensuring they  *"build safe, reliable, and steerable systems when those systems are starting to become as intelligent and as aware of their surroundings as their designers"* , and IBM as  *"encoding human values and goals into large language models to make them as helpful, safe, and reliable as possible"* [[Unit 1]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%201%3A%20Losing%20control%20to%20AI.md#what-is-ai-alignment).
* **Governance**: structures to deter the development/use of AI systems that could cause harm [[Unit 1]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%201%3A%20Losing%20control%20to%20AI.md#what-is-ai-alignment).
* **Resiliance**: ensuring other systems can cope with any negative/harmful behaviour/outcomes of AI systems [[Unit 1]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%201%3A%20Losing%20control%20to%20AI.md#what-is-ai-alignment).
* **Outer misalignment** (reward misspecification): the true goal of a user is not correctly mapped to the *proxy goal* that we actually input into the AI system [[Unit 1]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%201%3A%20Losing%20control%20to%20AI.md#what-is-ai-alignment). Note: Using proxies can result in misalignment [[Unit 1]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%201%3A%20Losing%20control%20to%20AI.md#what-failure-looks-like).
* **Inner misalignment** (goal misgeneralisation): the AI system has not correctly interpreted/actioned the true/proxy goal [[Unit 1]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%201%3A%20Losing%20control%20to%20AI.md#what-is-ai-alignment).
* **Influence-seeking behaviour**: systems that have a detailed enough understanding of their environment to be able to desceptively adapt their behaviour to better achieve certain goals [[Unit 1]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%201%3A%20Losing%20control%20to%20AI.md#what-failure-looks-like).
* Given 'seemingly good' performance, an AI model has the potential to be a [[Unit 1]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%201%3A%20Losing%20control%20to%20AI.md#why-ai-alignment-could-be-hard-with-modern-deep-learning):
  * **Saint**: genuinely performs as we expect.
  * **Sycophant**: satisfies short-term goals/tasks whilst disregarding long-term consequences.
  * **Schemer**: has a misaligned internal adgenda that seems to perform well during testing but is exploited to persue alternative goals later on.
  * Note: some research questions whether AI models can haveingrained long-term goals at all, and posit that saints/schemers is too anthropomorphic.

#### Reinforcement Learning with Human Feedback (RLHF)

* Overview:
  * RLHF can guide a model's responses to being helpful and harmless [[Unit 2]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%202%3A%20What%20makes%20aligning%20AI%20difficult%3F.md#a-simple-technical-explanation-of-rlhaif).
  * Human feedback is gathered about a base model (according to a set of guidelines), which is used to train a 'value model' (which emulates a human evaluator). The model can then be fine-tuned using the value model (plus a 'coherence' score from the unchanged base model) to inform its reward function [[Unit 2]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%202%3A%20What%20makes%20aligning%20AI%20difficult%3F.md#how-gpt-2-became-maximally-lewd).
* Training process [[Unit 2]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%202%3A%20What%20makes%20aligning%20AI%20difficult%3F.md#a-simple-technical-explanation-of-rlhaif):
  * Produce preference dataset using human feedback:
    * Use base model to create evaluation dataset of multiple responses per prompt.
    * Human evaluators pick the best prompt w.r.t a set of guidelines.
    * All prompt-response pairs are scored by 'Elo rating', producing a preference dataset of prompt-responses-scores.
  * Train value model to predict score of a given prompt-response pair.
  * Fine-tune the base model using a loss function that consists of the value model's predicted score and a 'penalty' factor that maintains coherency (computed as the KL divergence between the probability distribution output from the base model and fine-tuned model on the same prompt).
* Problems with RLHF [[Unit 2]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%202%3A%20What%20makes%20aligning%20AI%20difficult%3F.md#problems-with-reinforcement-learning-from-human-feedback-rlhf-for-ai-safety):
  * Scalability (hence the introduction of RLAIF).
  * LLMs often engage in sychophantic behaviour where responses are given that aim to elicit human approval to maximise their reward.
  * RLHF has been shown to not be capable of identifying schemer models that are capable of misaligned and desceptive behaviour.
  * Human evaluation can be falliable, and introduce inaccuracies and bias.
  * Not robust against jailbreak prompts.
  * Research suggests harmlessness fine-tuning can be undone in open-source models

#### Constitutional AI and Reinforcement Learning with AI Feedback (RLAIF)

* Overview:
  * **Constitutional AI**: techniques that aim to align AI systems based on a set of constitutional principles [[Unit 2]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%202%3A%20What%20makes%20aligning%20AI%20difficult%3F.md#rlaif-vs-rlhf-the-technology-behind-anthropics-claude-constitutional-ai-explained).
  * The constitutional principles should define rules that outline certain critiques/revisions that can be made to a model's responses to promote helpfulness and harmlessness [[Unit 2]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%202%3A%20What%20makes%20aligning%20AI%20difficult%3F.md#rlaif-vs-rlhf-the-technology-behind-anthropics-claude-constitutional-ai-explained).
  * Constitutional AI has been shown to produce models that are more explainable but less harmful and evasive [[Unit 2]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%202%3A%20What%20makes%20aligning%20AI%20difficult%3F.md#a-simple-technical-explanation-of-rlhaif).
  * **Reinforcement Learning with AI Feedback**: the training process through which we can train constitutional AI systems by replicating RLHF but with AI feedback instead of human evaluators [[Unit 2]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%202%3A%20What%20makes%20aligning%20AI%20difficult%3F.md#a-simple-technical-explanation-of-rlhaif).
* Training process [[Unit 2]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%202%3A%20What%20makes%20aligning%20AI%20difficult%3F.md#rlaif-vs-rlhf-the-technology-behind-anthropics-claude-constitutional-ai-explained):
  * Supervised learning phase:
    * A 'constitution' is created that defines critiques/revisions that should be made to harmful content.
    * An existing model that has been trained to be 'helpful' (e.g. through RLHF) generates responses over a set of 'harmful' prompts.
    * Select a principle from the constitution, and instruct the helpful model to critique and revise its own response based on that principle.
    * Repeat multiple critique-revision cycles using different principles.
    * A 'harmlessness' training dataset of SL samples is formed of harmful prompts and revised responses, plus a sample of helpful prompt-response pairs.
    * A new SL-CAI model (supervised learning constitutional AI) is trained by fine-tuning the helpful model on the harmlessness dataset.
  * Reinforcement learning phase:
    * We then conduct RLHF on the SL-CAI model, but instead of using human evaluators we use SL-CAI model itself.
    * The SL-CAI model (or the original helpful model in some cases) is used to generate a set of multiple responses for a (new) set of harmful prompts.
    * A preference dataset is constructed by using the SL-CAI model to pick one of the responses based on a given constitutional principle.
    * A preference model is trained via SL on the preference dataset.
    * The final RL-CAI model is trained by using the preference model to further fine-tune the SL-CAI model using RL.

#### Scalable Oversight

* **Scalable oversight**: methods that aim to assist humans in evaluating complex tasks and behaviour of AI systems [[Unit 3]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%203%3A%20Using%20AI%20to%20align%20AI.md#using-ai-to-align-ai).
* **Debate** [[Unit 3]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%203%3A%20Using%20AI%20to%20align%20AI.md#using-ai-to-align-ai):
  * Multiple AI systems argue different takes on a task/issue, providing evidence and arguments, and discrediting other debaters statements.
  * A winner is then decided either by the debaters coming to an aggreement or human evaluators.
  * Bad tendancies like deception are ideally picked up by the other debaters.
  * Debaters may win evaluators over by appealing to their biases, models may collude, or correct arguments may necessitate being too complex/niche to be preferred by evaluators.
* **Red teaming** [[Unit 3]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%203%3A%20Using%20AI%20to%20align%20AI.md#using-dangerous-ai-but-safely): a group of engineers who design an 'attack policy' by purposefully corrupting SOTA models to exhibit harmful behaviour.
* **Blue teaming** [[Unit 3]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%203%3A%20Using%20AI%20to%20align%20AI.md#using-dangerous-ai-but-safely): a second group of engineers who design the 'protocol' that aims to use the harmful model without allowing its harmful behaviour to get through (maximising a usefulness score and safety score).

#### Mechanistic Interpretability

* **Mechanistic Interpretability** [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#introduction-to-mechanistic-interpretability): the field of understanding the inner reasoning processes of neural networks to deduce how they produce their outputs. It is a crucial element of detecting unaligned model behaviours and unwanted biases, and for intervening to avoid failure modes caused by misalignment.
* Mechanistic interpretability research attempts to which parts of a network correspond to which neural 'features', and how this is mapped to the network's outputs.
* Improved interpretability could assist with model auditing, improved feedback, interpretable loss functions, forecasting discontinuities, and early-exiting of training [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#a-longlist-of-theories-of-impact-for-interpretability).

#### Neural Network Features

* **Features** are the fundamental unit of neural networks [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#an-introduction-to-circuits-blog). They can be seen as 'directions' in the vector space of neuron activations, i.e. the vector of activations of a set of neurons gives the 'feature' direction.
* Early network layers contain features that detect components like edges/curves, i.e. when a set of neurons are(n't) activated this corresponds to the positive/negative identification of a (edge/curve) component in the input. The features of later layers correspond to detecting higher-fidelity concepts, like a wheel.
* Features appear to exist in 'families' that detect the same feature but in different orentations, e.g. different directions/scales of curve.
* Proving a set of neurons is a 'curve detection feature' in a vision model [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#an-introduction-to-circuits-blog):
  * *Feature visualisation:* adjusting the input image such that the neurons are always activated - does the input always contain curves?
  * *Dataset examples*: find the images that cause the neurons to activate the most - do these contain curves?
  * *Synthetic examples*: introducing synthetic features (curves, no curves, other lines) to other images - do these cause the neurons to activate?
  * *Joint tuning*: does rotation change which the set of neurons is activated? Does one family of neurons gradually stop activating and another begin to activate?.
  * *Feature implementation*: inspecting the weighted connections between circuits (see below) and layers of neurons to identify the relevant feature.
  * *Feature use*: inspect the higher-fidelity features in later layers that emerge - do later derivative feature concepts contain curves?
  * *Handwritten circuits*: would a network purposely build to detect the feature have activations mimic that of the original network?

#### Neural Network Circuits

* **Circuits**: the relationship between a collection of neurons within a layer of the neural network that provides the mechanism by which knowledge is aquired [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#introduction-to-mechanistic-interpretability).
* Circuits are how we refer to the weighted subgraph of connections between neurons that encode meaning (i.e. a 'feature') that corresponds to tractable learned concepts of information.
* If we look at the circuits of a network, we can read meaning from the weights of connections between neurons and layers, showing us what concepts positive weights reinforce and negative weights inhibit.
* Circuits can be 'pure' (only represent a single feature) or 'polysemantic', and pure circuits can connect into polysemantic circuits in later laters.
* **Polysemantic circuits**:
  * Restricting a model containing 'N' neurons to only represent 'N' concepts would severely limit its understanding capabilities [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#lets-try-to-understand-ai-monosemanticity).
  * **Polysemanticity**: some single neurons can represent multiple different features (superposition). This makes it hard to disentanble harmful features from harmless ones [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#introduction-to-mechanistic-interpretability).
  * Through this arrangement a model can 'simulate' having much higher number of neurons required to represent many more concepts (sometimes known as 'superposition') [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#lets-try-to-understand-ai-monosemanticity).
  * For example, for a pair of neurons $A$ and $B$, neuron $A$ being activated at 0.5 AND neuron $B$ activated at 0 could signify the concept 'dog'. This can be extended to larger groups of neurons in higher dimensions [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#lets-try-to-understand-ai-monosemanticity).
  * Polysemantic neurons complicate the identification of features and circuits as there can be multiple different features that activate a given neuron/feature/circuit. This can make reasoning about a network difficult as these overlapping meanings need unfolding/disentangling [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#an-introduction-to-circuits-blog).

#### Sparse Autoencoders

* **Sparse autoencoders**: a network architecture for engraining monosemantic behaviour where neurons are constrained to only represent a single feature. This is done using a 'sparsity constraint' that encourages many of the neurons in a network to remain inactive and those that are active are 'specialised' to important features of the inputs [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#introduction-to-mechanistic-interpretability).
* This approach can be used to attempt to interpret feature representations in neurons [[Unit 4]](https://github.com/tom-bbc/AI-Safety-and-Alignment/blob/main/AI-Alignment-Fast-Track/Unit%204%3A%20Understanding%20how%20AI%20systems%20think.md#lets-try-to-understand-ai-monosemanticity).
* To interpret an LLM, the SAE is trained to predict the neuron activations of the LLM. This basically abstracts the 'neuron level' interpretability of the LLM to 'feature level' interpretations in the SAE, which makes the subjective interperation of the model's inner bevaviours/thinking much better than looking at actual singular neurons.
* The SAE can then be used to identify which of the LLMs neurons act together as a circuit, and what feature this circuit corresponds to.

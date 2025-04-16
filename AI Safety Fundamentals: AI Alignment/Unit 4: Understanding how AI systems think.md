# AI Safety Fundamentals: AI Alignment

## Unit 4: Understanding how AI systems think

### Introduction to Mechanistic Interpretability

* **Mechanistic Interpretability**: the field of understanding the inner reasoning processes of neural networks to deduce how they produce their outputs.
* Can be used to detect unaligned model behaviours and identify unwanted biases.
* This is crucial for intervening in these cases to avoid failure modes caused by misalignment through ['feature steering'](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) methods.
* **Circuits**: the relationship between a collection of features within a layer of the neural network that provides the mechanism by which knowledge is aquired.
* Some research suggests that networks tend to converge on similar feature representations and circuits.
* Mechanistic interpretability research attempts to which parts of a network correspond to which features and how this is mapped to the output.
* **Polysemanticity**: some single neurons can represent multiple different features (superposition).
* This makes it hard to disentanble harmful features from harmless ones.
* **Sparse autoencoders**: a network architecture for engraining monosemantic behaviour where neurons are constrained to only represent a single feature. This is done using a 'sparsity constraint' that encourages many of the neurons in a network to remain inactive and those that are active are 'specialised' to important features of the inputs.
* Anthropic scaled this method to interpreting LLMs to identify how millions of features are represented (["Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)). This work has also been used for 'feature steering' large models to weaken/strengthen certain features to influence the model's outputs.

### A Longlist of Theories of Impact for Interpretability

* Machine interperability could enable:
  * **Auditing**: checking for misalignment and detecting deceptive behaviour.
  * **Improved feedback**: reinforcing models to do the correct behaviours for the correct reasons.
  * **Interpretable loss functions**: ensure the system is learning to behave in an aligned manner by using interperability tools within the training process / loss function.
  * **Regulation**: more grounded, technically enforceable regulation surrounding AI safety and governance.
  * **Forecasting discontinuities**: prediction of the likelihood of alignment discontinuities that could arise in future from the training/deployment of larger/more complex AI systems.
  * **Early-exiting**: if we audit training for misalignment, we can halt training runs and start over / switch stregegy / architecture if misalignment starts to appear.

### Let's Try To Understand AI Monosemanticity

* The number of internal neurons (~100bn in GTP-4) and polysemanticity of those neurons (one neuron can respond to multiple input 'concepts'/genres of text) make it very challenging to interpret deep neural networks.
* Polysemanticity / superposition:
  * Some research (["Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html)) suggests this is because restricting a model containing $N$ neurons to only represent $N$ concepts would severely limit its understanding capabilities.
  * Pairs of neurons can also represent these concepts together: e.g. neuron $A$ being activated at $0.5$ AND neuron $B$ activated at $0$ signifies the concept 'dog'.
  * Anthropic's research found this can be extended to larger groups of neurons in higher dimensions, such as using 2 dimensions to represent 3 features in a triange-esc connection arrangement or 5 features in a pentagon arrangement, expanding to a 'square antiprism' to represent 8 features in 3 dimensions, etc... up to ~1000 dimensional space in complex LLMs.
  * Through this arrangement a model can 'simulate' having much higher number of neurons required to represent many more concepts (sometimes known as 'superposition').

![Polysemanticity in LLMs diagram](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F157938c0-cc3b-4bfa-8c21-7513581cca04_1433x1046.png)

* Sparse autoencoders:
  * Anthropic's paper ["Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"](https://transformer-circuits.pub/2023/monosemantic-features/index.html) uses 'sparse autoencoders' to attempt to interpret some of these feature representations in neurons.
  * The autoencoder was trained to predict activations of an LLM (that is simulating a high number of concepts in high dimensional space, as above).
  * From this, we can inspect how concepts are represented in features and where (i.e. which 'real' neurons acting together as a 'simulated' neuron), and how the activation of these neurons effect the LLM's output distribution.
  * Interestingly, whilst the LLM's neurons were polysemantic, the autoencoders remained monosemantic; i.e. each neuron represented a distict concept about how the LLM was representing its features.
  * Anthropic found that looking at interperability at this 'feature level' (i.e. 'simulated' neurons of a collection of actual neurons) made subjective interperation of the model's inner bevaviours/thinking much better than looking at actual singular neurons.
  * Feature representation was also inspected over different models trained identically but separatedly, and it was found these representations higly correlate.
  * However, Anthropics work focussed on interpreting only a small LLM (~16k features) and has not yet been scalable to frontier models.

### An Introduction to Circuits ([blog](https://distill.pub/2020/circuits/zoom-in/))

* The paper hypothesises: 'Features' are the fundamental unit of neural networks. Features are interconnected, forming 'circuits'. Analagous features/circuits form across different models and domains.
* Features:
  * Features can be seen as 'directions' in the vector space of neuron activations.
  * Early layers in a DNN contain features that detect components like edges/curves, whilst later layers features correspond to detecting higher-fidelity concepts like a wheel.
  * 'Curve detection neurons' are commonly found in vision models. OpenAI uses the example of the 'InceptionV1' model to explain these:
    * These curve detectors appear to exist in 'families' that detect the same curve feature but in different orentations.
    * The paper uses 7 arguments to 'prove' the behaviour of curve detection neurons, which can be used to interpret other features too:
    1) **Feature visualisation**: if we adjust/optimise the input image to always activate the given neurons, that input contains many curves (causal link).
    2) **Dataset examples**: the images in the test dataset that cause these neurons to fire the most contain many curves.
    3) **Synthetic examples**: introducing synthetic curves to input images gets these neurons to activate, but introducing straight lines etc. does not.
    4) **Joint tuning**: if we rotate a curve within an image, one family of neurons gradually stops activating and another begins to activate (hence families correspond to different orientations).
    5) **Feature implementation**: inspecting the weighted connections between circuits of neurons can help infer what feature they correspond to.
    6) **Feature use**: inspecting the higher-fidelity features later in the network that emerge from the features at the current earlier layer reveals those features to be ones that typically contain curves (suggesting these features are build up from curve features earlier in the network).
    7) **Handwritten circuits**: if we build our own network with known circuits corresponding to curve detection, its activations mimic that of the original network.
![Example of feature visualisation and dataset examples for a dog head detector](https://distill.pub/2020/circuits/zoom-in/images/dog-pose.png)
![Diagram showing how early layers' features can be built up to complex shapes later in the network](https://distill.pub/2020/circuits/zoom-in/images/curve-circuit.png)
  * These arguments can be brought together to interpret the meaning of given features and circuits.
  * The examples of 'high-low frequency detection' features and 'dog head detection' features are also given to show this interperation method can be used for more and less 'conceptual' features.
* Polysemantic neurons complicate this - optimising for each of the 7 arguments above may give multiple different features that activate a given neuron/circuit. This can make reasoning about a network difficult as these overlapping meanings need unfolding/disentangling.
* Circuits:
  * 'Circuits' are a weighted subgraph of connections between neurons in a neural network that seem to encode meaning that corresponds to learned facts.
  * Circuits often define tractable/meaningful concepts as their features with rich structures.
  * This can allow us to read meaning from the weights of connections between neurons and layers, showing us what concepts positive weights reinforce and negative weights inhibit.
  * E.g. the weights connecting an earlier curve detector to a full curve detector we may have stronger weights at the point in the curve where the orientation most fully aligns.
  * 'Pure' neurons/circuits that represent a single feature in earlier layers can feed into layer polysemantic neurons/circuits that are a 'superposition' of different features.
  * If we understand circuits, we should be able to make falsifiable predictions: i.e. how will a network's representations/outputs changed if the weights of a given circuit are edited? This could be used to break down statements about networks as a whole into a set of statements about the network's circuits.
* Universality:
  * It's commonly understood that all vision models trained on natural images tend to use their first layer as a 'Gabor filter' for texture analysis that identifies specific frequency content in the image in specific directions.
  * 'Universality' extrapolates this to posit that different trained networks learn highly correlated feature representations.
  * There is not yet strong evidence of this hypothesis, we only have anecdotal evidence that some low-level features seem to commonly form across different vison model architectures (e.g. curve detection features and high-low frequency detection features).

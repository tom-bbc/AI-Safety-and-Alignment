# Mechanistic Interpretability Quickstart Guide

## Resources

* Useful course / presentation: [Intro to Mechanistic Interpretability: TransformerLens & Induction Circuits](https://arena-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp)
* TransformerLens:
  * The TransformerLens library is a collection of features for mechanistic interpretability.
  * [TransformerLens Git Repo](https://github.com/TransformerLensOrg/TransformerLens)
  * [Transformer Lens Main Demo Notebook](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb#scrollTo=pPSgSaDOIQPG)
  * [Transformer Lens for Exploritary Analysis Demo](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb#scrollTo=b54lBNxYM_PS)
* CircuitsVis:
  * The CircuitsVis library lets you pass in tensors to an interactive visualization:
  * [CircuitsVis Git Repo](https://github.com/TransformerLensOrg/CircuitsVis)
  * [CircuitsVis example visualisations](https://transformerlensorg.github.io/CircuitsVis/?path=/story/activations-textneuronactivations--multiple-samples)
* A Mathematical Framework for Transformer Circuits:
  * [Anthropic blogpost](https://transformer-circuits.pub/2021/framework/index.html)
  * [Paper walkthrough video](https://www.youtube.com/watch?v=KV5gbOmHbjU)
  * [Explainer of key terms](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=aGu9fP1EG3hiVdq169cMOJId)

## Intro to Mechanistic Interpretability: TransformerLens & induction circuits

* **Induction circuits**:
  * A type of transformer circuit that can perform basic in-context learning.
  * (Strict) **Induction task**: the task of detecting and using repeated subsequence patterns within an input text; e.g. if we observe "James Bond" occurs earlier in a text, we likely want to use "Bond" as the next token to be predicted after the token "James" reoccurs.
  * More general forms of induction can involve tasks like checking whether the last $k$ tokens occur earlier in the input text.
  * **Previous token head**: an attention head who's behaviour is to (mainly) attend to the previous token.
  * **Induction head**: an attention head who's behaviour implements the induction task, i.e. attending to the token immediately after a previous occurance of the current token, and predicting the reuse of this token (by directly increasing the value of that token's logit). Typically if there are multiple different next tokens witnessed, these are all attended over diffusely.
  * **Anti-induction head**: implements the same induction tasks but suppresses the attended token.
  * **Induction circuit**: a circuit of two attention heads that implements induction behaviour (using K-composition), with the first head acting as a previous token head, and the second an induction head.
* TransformerLens Introduction:
  * Models can be loaded in as a `HookedTransformer`.
  * Overall architecture changes:
    * This architeture is slighly altered compared to the existing architecture of transformer models like Llama, but is computationally identical.
    * One of the main changes is that activations and weight matricies (the query, key, and value matrices) have two axes, `head_index` and `d_head`, instead of being flattened.
    * This means activations have shape `[batch, position, head_index, d_head]`, and weight matrices `[head_index, d_model, d_head] and W_O has shape [head_index, d_head, d_model]`.
  * Parameters and activations:
    * The **activations** of a model depend on a given input, e.g. the attention scores and patterns; we can use 'hooks' to access these values during a forward pass.
    * A single attention-only layer in this paradigm is a `TransformerBlock`; the architecture of each block is shown below.

![The architecture of a TransformerBlock](https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/small-merm.svg)

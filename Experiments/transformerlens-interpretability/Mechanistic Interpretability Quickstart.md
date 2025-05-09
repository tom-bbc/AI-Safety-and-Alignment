# Mechanistic Interpretability Quickstart Guide

## Resources

* Useful course / presentation: [Intro to Mechanistic Interpretability: TransformerLens &amp; Induction Circuits](https://arena-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp)
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
  * Generalised induction heads can generalise from one observation that token `B` follows token `A`, to predict that token `B` will follow `A` after future occurrences of `A`, even if these two tokens had never appeared together in the model's training data.
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
    * <img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/small-merm.svg" alt="Attention-only architecture of a TransformerBlock" width="200"/>
    * A full `TransformerBlock` layer combines this attention-only block with biases, layernorms, and MLPs, as shown below.
    * <img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/full-merm.svg" alt="Full architecture of a TransformerBlock" width="200"/>
    * In TransformerLens, you can inspect weight matrices across the whole model with `model.W_Q` or by layer `model.blocks[0].attn.W_Q` (which gives all query weights for attention heads in layer 0).
    * We can similarly inspect the embeddings, unembeddings, and positional embeddings using `model.W_E`, `model.W_U`, and `model.W_pos` respectively.
    * And the same pattern can be used for bias matrices, e.g. `model.b_Q` for all query biases.
    * MLP layers can be inspected using `model.W_in` and `model.W_out`.
  * The model's tokeniser can be accessed through `model.tokenizer`. We can tokenize with `model.to_tokens()` and decode with `model.to_string()` or `model.to_str_tokens()`.
* Caching activation value:
  * We can generate logits and cache the utilised activation values on a given input with `logits, cache = model.run_with_cache(text)`.
  * Each key in the returned cache object corresponds to a single activation within the full model. These can be indexed by later using the key pattern `cache["pattern", 0]`.
* Visualising attention heads:
  * Research suggests that some aspects of a model are 'intrinsically interpretable' --- such as the input tokens, the output logits, and the attention patterns --- whereas others --- such the residual stream, keys, queries, values --- may not be possible to interpret and are instead compressed intermediate states.
  * We can use the CircuitsVis library (above) to visualise the attention pattern of all attention heads in each layer of a given model on a given input sequence.
  * To do this, use the `attention.attention_patterns` function that takes input of the attention head patterns and string tokens of the input sequence.
  * For models with causal attention (attention can only look backwards) this fill plot a series of triangular charts of the attention patterns.
  * There are three basic patterns that appear within the visualised attention patterns:
    * Previous token heads - main attention patterns are on the preceeding token.
    * Current token heads - attention mainly on the current token.
    * First token heads - mainly attend to the first token in the sequence. The first token is often used as the 'resting' position for heads that only sometimes activate, as attention probabilities must sum to 1.
* Induction heads:
  * **Induction head circuit:** the induction circuit consists of a previous token head in layer 0 and an induction head in layer 1, where the induction head learns to attend to the token immediately after copies of the current token via K-Composition with the previous token head. Hence, induction heads cannot form in a single layer model.
  * Induction heads seem to be responsible for the vast majority of **in-context learning** - the ability to use far back tokens in the context to predict the next token. This is a significant way in which transformers outperform older architectures like RNNs or LSTMs.
  * They develop fairly suddenly in a phase change (from about 2B to 4B training tokens) and are responsible for a significant loss decrease, so much so that there's a visible bump in the loss curve when they develop.
  * This core circuit seems to be used in a bunch of more sophisticated settings, such as translation or few-shot learning.
  *

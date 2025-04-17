# Negative Results for Sparse Autoencoders On Downstream Tasks and Deprioritising SAE Research

Sources:
* [Negative Results for Sparse Autoencoders On Downstream Tasks and Deprioritising SAE Research | Medium](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
* [Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research | AI Alignment Forum](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/sae-progress-update-2-draft#Dataset_debugging_with_SAEs)

### Overview of findings around SAEs use in interpretability

* It is evident that when using SAEs for interpreting model behaviour, results are not always compelling for all sequences; for a given arbitrary sentence, the latents that are activated do not seem to always correspond to a crisp explanation.
* The hope of interpretability research using SAEs is that SAE latents capture some canonical set of true concepts inside a model.
* Researchers found that the standard interpretability approach using SAEs (computing the average interpretability of a uniformly sampled SAE latent) can give to misleading results.
* This is because SAEs don't penalise models with a latents that are high-frequency but low interpretability.
* Other issues with SAEs are highlighted:
  * They do not have a complete vocabulary of all concepts.
  * Concepts can be represented in noisy ways (e.g. small activations don't seem interpretable).
  * Seemingly interpretable latents have many false negatives.
* However, the researchers argue that if attempting to gain a better understanding of what goals and deceptive alignment look like within a model, we do not necessarily require complete visibility of all of the 'true features' of a model, but can instead make do with a decent approximation to the models computation.

### Experimenting with using SAEs for OOD probing

* Researchers investigated the utility of SAEs to detect out-of-distribution user prompts with harmful intent (specifically using the presence of different jailbreaks, with new jailbreaks as the OOD set).
* They found SAEs underperformed linear probes, which seem to work "near perfectly" even on the OOD set.
* The same result was found when fine-tuning pre-trained SAEs specifically on chat data; although this did close the performance gap.
* Specifically, they found "$k$-sparse SAE probes" can fit the training set for moderate $k \approx 20$, and successfully generalise to an in-distribution test set, but show distinctly worse performance on the OOD set.
* They state their key kinding is that the probing results show current SAEs do not find the 'concepts' required to be useful on an important task (detecting harmful intent), but a linear probe can find a useful direction.

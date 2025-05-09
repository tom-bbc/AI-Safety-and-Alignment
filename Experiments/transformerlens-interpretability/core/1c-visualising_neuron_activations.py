import os
import webbrowser

import circuitsvis
import circuitsvis.activations
import einops
import torch
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils

################################################################################
# Main process
################################################################################


def main(model_name: str, input_text: str, device: str) -> None:
    ################################################################################
    # Loading the model
    ################################################################################

    model = HookedTransformer.from_pretrained(model_name, device=device)
    n_layers = model.cfg.n_layers

    ################################################################################
    # Inspecting activation values
    ################################################################################

    token_ids = model.to_tokens(input_text)
    token_strings = model.to_str_tokens(input_text)

    # Generating logits and caching the utilised activation values on a given input
    output_logits, cached_activations = model.run_with_cache(
        token_ids, remove_batch_dim=True
    )

    # Get neuron activations for all layers
    neuron_activations = [
        cached_activations["post", layer_idx] for layer_idx in range(n_layers)
    ]
    neuron_activations = torch.stack(neuron_activations, dim=1)

    # html_page = circuitsvis.activations.text_neuron_activations(
    #     tokens=token_strings, activations=neuron_activations
    # )

    # Reshape activations to match sequence dimentions
    neuron_activations = tl_utils.to_numpy(
        einops.rearrange(
            neuron_activations,
            "seq layers neurons -> 1 layers seq neurons",
        )
    )

    # Visualise largest activations on sequence for each neuron and layer
    html_page = circuitsvis.topk_tokens.topk_tokens(
        tokens=[token_strings],
        activations=neuron_activations,
        max_k=10,
        first_dimension_name="Layer",
        third_dimension_name="Neuron",
        first_dimension_labels=list(range(n_layers)),
    )

    # Visualise neuron activations on set of HTML pages
    html_page_filepath = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "outputs",
        "neuron_activations_all_layers.html",
    )

    with open(html_page_filepath, "w") as fp:
        fp.write(str(html_page))

    print(f" * Visualised neuron activations saved to: {html_page_filepath}")
    webbrowser.open("file://" + html_page_filepath)


if __name__ == "__main__":
    model_name = "gpt2-small"
    device = "mps"

    example_text = "HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"  # noqa

    main(model_name, example_text, device)

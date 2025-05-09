import os
import webbrowser

import circuitsvis
from transformer_lens import HookedTransformer


def main(model_name: str, input_text: str, device: str) -> None:
    ################################################################################
    # Loading the model
    ################################################################################

    model = HookedTransformer.from_pretrained(model_name, device=device)

    ################################################################################
    # Inspecting activation values
    ################################################################################

    token_ids = model.to_tokens(input_text)
    token_strings = model.to_str_tokens(input_text)

    # Generating logits and caching the utilised activation values on a given input
    output_logits, cached_activations = model.run_with_cache(
        token_ids, remove_batch_dim=True
    )

    # Visualise attention patterns on set of HTML pages
    # n_layers = model.cfg.n_layers
    layer_idx = 0

    # for layer_idx in range(n_layers):
    cached_activations_layer_idx = cached_activations["pattern", layer_idx]

    html_page = circuitsvis.attention.attention_patterns(
        attention=cached_activations_layer_idx,
        tokens=token_strings,
    )

    html_page_filepath = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "outputs",
        f"attention_head_activations_layer_{layer_idx}.html",
    )

    with open(html_page_filepath, "w") as fp:
        fp.write(str(html_page))

    print(f"\n * Visualised attention head activations saved to: {html_page_filepath}")
    webbrowser.open("file://" + html_page_filepath)


if __name__ == "__main__":
    # model_name = "gpt2-small"
    model_name = "meta-llama/Llama-3.2-1B"
    device = "mps"

    example_text = "HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"  # noqa
    # example_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."  # noqa

    main(model_name, example_text, device)

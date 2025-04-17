import circuitsvis
import circuitsvis.attention
import torch
from einops import einsum
from transformer_lens import HookedTransformer


def print_section_header(title: str) -> None:
    print("\n" + "#" * 80)
    print(f"# {title}")
    print("#" * 80)


def main() -> None:
    ################################################################################
    # Loading the model
    ################################################################################

    model_name = "gpt2-small"
    device = "mps"
    model = HookedTransformer.from_pretrained(model_name, device=device)

    ################################################################################
    # Inspecting the model
    ################################################################################

    print_section_header("Inspecting the model")

    model_config = model.cfg
    embedding_matrices = model.W_E
    query_matrices = model.W_Q
    query_matrices_layer_0 = model.blocks[0].attn.W_Q
    mlp_output_layer = model.W_out

    print(f"\n * Model config: \n\n{model_config}")
    print(f"\n * Model embedding matrices: \n\n{embedding_matrices}")
    print(f"\n * Model query matrices: \n\n{query_matrices}")
    print(
        "\n * Model query matrices for attn heads in layer 0: \n\n",
        {query_matrices_layer_0},
    )
    print(f"\n * Model output MLP layer: \n\n{mlp_output_layer}")

    ################################################################################
    # The model's tokenizer
    ################################################################################

    print_section_header("The model's tokenizer")

    example_text_input = "HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"  # noqa

    # Access the tokenizer
    print(f"\n * Tokenizer: \n\n{model.tokenizer}")

    # Tokenize a text sequence
    encoded_tokens = model.to_tokens(example_text_input)
    print(f"\n * Encoded tokens from string: \n\n{encoded_tokens}")

    decoded_text = model.to_string(encoded_tokens)
    print(f"\n * Decoded string from tokens: \n\n{decoded_text}")

    ################################################################################
    # Running the model
    ################################################################################

    print_section_header("Running the model")

    # Generate output loss on text sequence
    output_loss = model(example_text_input, return_type="loss")
    print(f"\n * Model output loss on example text input: {output_loss}")

    # Generate output logits on text sequence
    output_logits = model(example_text_input, return_type="logits")
    print(f"\n * Model output logits on example text input: \n\n{output_logits}")

    predicted_tokens = output_logits.argmax(dim=-1).squeeze()
    predicted_text = model.to_str_tokens(predicted_tokens)
    print(f"\n * Model prediction on example text input: \n\n{predicted_text}")

    correctly_predicted_tokens = (
        predicted_tokens[:-1] == model.to_tokens(example_text_input).squeeze()[1:]
    )
    correctly_predicted_tokens = predicted_tokens[:-1][correctly_predicted_tokens]
    correctly_predicted_text = model.to_str_tokens(correctly_predicted_tokens)
    print(
        "\n * Correct predictions for next token prediction task: \n\n",
        correctly_predicted_text,
    )

    ################################################################################
    # Inspecting activation values
    ################################################################################

    print_section_header("Inspecting activation values")

    example_text_2 = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."  # noqa
    tokenized_text_2 = model.to_tokens(example_text_2)

    # Generating logits and caching the utilised activation values on a given input
    output_logits, cached_activations = model.run_with_cache(
        tokenized_text_2, remove_batch_dim=True
    )

    cached_activations_layer_0 = cached_activations["pattern", 0]
    print(
        "\n * Activations of layer 0 on example text input: \n\n",
        cached_activations_layer_0,
    )

    # Comparing cached activations to manually calculated activation values
    cached_query_vector_layer_0 = cached_activations["q", 0]
    cached_key_vector_layer_0 = cached_activations["k", 0]

    seq, nhead, headsize = cached_query_vector_layer_0.shape
    attention_scores_layer_0 = einsum(
        cached_query_vector_layer_0,
        cached_key_vector_layer_0,
        "seqQ n h, seqK n h -> n seqQ seqK",
    )

    attention_mask = torch.triu(
        torch.ones((seq, seq), dtype=torch.bool), diagonal=1
    ).to(device)
    attention_scores_layer_0 = attention_scores_layer_0.masked_fill(
        attention_mask, -1e9
    )

    manual_activations_layer_0 = (attention_scores_layer_0 / headsize**0.5).softmax(-1)
    check_match = torch.equal(cached_activations_layer_0, manual_activations_layer_0)

    print(
        "\n * Manually calculated activations of layer 0 on example text input: \n\n",
        manual_activations_layer_0,
    )
    print(f"\n * Manual and cached activations match: {check_match}")

    # Visualising attention patterns
    example_text_string_tokens = model.to_str_tokens(example_text_2)

    html_page = circuitsvis.attention.attention_patterns(
        attention=cached_activations_layer_0,
        tokens=example_text_string_tokens,
    )

    with open("attn_heads.html", "w") as f:
        f.write(str(html_page))


if __name__ == "__main__":
    main()
    print()

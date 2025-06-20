import os
import webbrowser

import circuitsvis
import circuitsvis.attention
import torch
from eindex import eindex
from huggingface_hub import hf_hub_download
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig
from utils import generate_repeated_tokens, print_section_header

################################################################################
# Main process
################################################################################


def main() -> None:
    ################################################################################
    # Loading the model
    ################################################################################

    device = "mps"
    display_html = False

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Define a toy attention-only model with no MLP layers, LayerNorms, or biases.
    config = HookedTransformerConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=50278,
        attention_dir="causal",
        attn_only=True,
        tokenizer_name="EleutherAI/gpt-neox-20b",
        seed=398,
        use_attn_result=True,
        normalization_type=None,
        positional_embedding_type="shortformer",
    )

    model = HookedTransformer(config)

    # We can initialise a model by downloading its weights from the HF Hub
    model_name = "callummcdougall/attn_only_2L_half"
    model_weights_path = hf_hub_download(model_name, filename="attn_only_2L_half.pth")

    pretrained_weights = torch.load(
        model_weights_path, map_location=device, weights_only=True
    )
    model.load_state_dict(pretrained_weights)

    ################################################################################
    # Inspecting activation values
    ################################################################################

    input_text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."  # noqa
    token_strings = model.to_str_tokens(input_text)

    logits, cached_activations = model.run_with_cache(input_text, remove_batch_dim=True)

    if display_html:
        print_section_header("Visualising attention patterns as HTML page")

        cached_activations_layer_0 = cached_activations["pattern", 0]
        html_page = circuitsvis.attention.attention_patterns(
            tokens=token_strings, attention=cached_activations_layer_0
        )
        html_page = str(html_page)

        html_page_filepath = os.path.join(
            os.getcwd(), "../outputs/attention_head_activations_layer_0.html"
        )

        with open(html_page_filepath, "w") as fp:
            fp.write(html_page)

        print(
            f"\n * Visualised attention head activations saved to: {html_page_filepath}"
        )
        webbrowser.open("file://" + html_page_filepath)

    ################################################################################
    # Detecting current, previous, and first token heads
    ################################################################################

    print_section_header("Detecting current, previous, and first token heads")

    n_layers = config.n_layers
    n_heads = config.n_heads

    # Detect current token heads
    current_token_heads = detect_current_token_heads(
        cached_activations, n_layers, n_heads
    )
    print(
        f"\n * Heads attending to the current token: {', '.join(current_token_heads)}"
    )

    # Detect previous token heads
    previous_token_heads = detect_previous_token_heads(
        cached_activations, n_layers, n_heads
    )
    print(
        f" * Heads attending to the previous token: {', '.join(previous_token_heads)}"
    )

    # Detect first token heads
    first_token_heads = detect_first_token_heads(cached_activations, n_layers, n_heads)
    print(f" * Heads attending to the first token: {', '.join(first_token_heads)}")

    ################################################################################
    # Plotting per-token loss on repeated sequence
    ################################################################################

    print_section_header("Plotting per-token loss on repeated sequence")

    sequence_length = 50
    batch_size = 1

    random_repeated_token_sequence = generate_repeated_tokens(
        model.cfg.d_vocab, sequence_length, batch_size, device
    )

    output_logits, cached_activations = model.run_with_cache(
        random_repeated_token_sequence
    )
    cached_activations.remove_batch_dim()
    random_repeated_token_strings = model.to_str_tokens(random_repeated_token_sequence)

    model.reset_hooks()

    # Get log probabilities
    log_probs = output_logits.log_softmax(dim=-1)
    log_probs = eindex(log_probs, random_repeated_token_sequence, "b s [b s+1]")
    log_probs = log_probs.flatten()
    log_probs = log_probs.cpu().detach().numpy()

    # Plot per-token loss
    initial_sequence_loss = log_probs[:sequence_length].mean()
    repeated_sequence_loss = log_probs[sequence_length:].mean()

    print(f"\n * Loss on initial part of sequence: {initial_sequence_loss:.2f}")
    print(f" * Loss on repeated part of sequence: {repeated_sequence_loss:.2f}")

    # Visualise attention patterns in second layer to identifiy induction heads
    cached_activations_layer_1 = cached_activations["pattern", 1]

    html_page = circuitsvis.attention.attention_patterns(
        attention=cached_activations_layer_1,
        tokens=random_repeated_token_strings,
    )

    html_page_filepath = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "outputs",
        "induction_head_activations.html",
    )

    with open(html_page_filepath, "w") as fp:
        fp.write(str(html_page))

    print(f"\n * Visualised attention head activations saved to: {html_page_filepath}")
    webbrowser.open("file://" + html_page_filepath)

    ################################################################################
    # Using attention patterns to detect induction heads
    ################################################################################

    print_section_header("Using attention patterns to detect induction heads")

    induction_heads = detect_induction_heads(
        cached_activations,
        n_layers,
        n_heads,
        sequence_length,
    )

    print(f"\n * Induction heads: {', '.join(induction_heads)}")


################################################################################
# Detecting current, previous, and first token heads
################################################################################


def detect_current_token_heads(
    cache: ActivationCache,
    n_layers: int,
    n_heads: int,
    attention_score_threshold: float = 0.4,
) -> list[str]:
    """Returns a list of indices "layer.head" detected as being current-token heads"""

    head_indices = []

    for layer in range(n_layers):
        for head in range(n_heads):
            # Get attention pattern for each layer and head
            attention_pattern = cache["pattern", layer][head]

            # Take average of attention scores along diagonal of attention matrix
            # i.e. scores related to the current token
            attention_score = attention_pattern.diagonal().mean()

            if attention_score >= attention_score_threshold:
                head_indices.append(f"{layer}.{head}")

    return head_indices


def detect_previous_token_heads(
    cache: ActivationCache,
    n_layers: int,
    n_heads: int,
    attention_score_threshold: float = 0.4,
) -> list[str]:
    """Returns a list of indices "layer.head" detected as being previous-token heads"""

    head_indices = []

    for layer in range(n_layers):
        for head in range(n_heads):
            # Get attention pattern for each layer and head
            attention_pattern = cache["pattern", layer][head]

            # Take average of attention scores along the first sub-diagonal
            # i.e. scores related to the previous token
            attention_score = attention_pattern.diagonal(-1).mean()

            if attention_score >= attention_score_threshold:
                head_indices.append(f"{layer}.{head}")

    return head_indices


def detect_first_token_heads(
    cache: ActivationCache,
    n_layers: int,
    n_heads: int,
    attention_score_threshold: float = 0.4,
) -> list[str]:
    """Returns a list of indices "layer.head" detected as being first-token heads"""

    head_indices = []

    for layer in range(n_layers):
        for head in range(n_heads):
            # Get attention pattern for each layer and head
            attention_pattern = cache["pattern", layer][head]

            # Take average of attention scores along the first row
            # i.e. scores related to the first token
            attention_score = attention_pattern[:, 0].mean()

            if attention_score >= attention_score_threshold:
                head_indices.append(f"{layer}.{head}")

    return head_indices


################################################################################
# An induction-head detector
################################################################################


def detect_induction_heads(
    cache: ActivationCache,
    n_layers: int,
    n_heads: int,
    repeated_seq_len: int,
    attention_score_threshold: float = 0.4,
) -> list[str]:
    """Returns a list of indices "layer.head" detected as being induction heads"""

    # In our repeated sequence, tokens reoccur at `repeated_seq_len` intervals
    # So the offset to check for attended tokens should be -(repeated_seq_len - 1)
    offset = -(repeated_seq_len - 1)
    head_indices = []

    # Cycle through each head in each layer
    for layer in range(n_layers):
        for head in range(n_heads):
            # Get attention pattern for each layer and head
            attention_pattern = cache["pattern", layer][head]

            # Check for attended tokens with an offset of -(repeated_seq_len - 1)
            attention_score = attention_pattern.diagonal(offset).mean()

            if attention_score > attention_score_threshold:
                head_indices.append(f"{layer}.{head}")

    return head_indices


if __name__ == "__main__":
    main()
    print()

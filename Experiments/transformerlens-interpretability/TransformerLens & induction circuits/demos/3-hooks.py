import einops
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from transformer_lens.hook_points import HookPoint
from utils import generate_repeated_tokens, print_section_header

################################################################################
# Main process
################################################################################


def main(model_name: str, input_text: str, device: str) -> None:
    ################################################################################
    # Loading the model
    ################################################################################

    model = HookedTransformer.from_pretrained(model_name, device=device)
    token_ids = model.to_tokens(input_text)

    ################################################################################
    # Running with hooks
    ################################################################################

    print_section_header("Running with hooks")

    # Baseline loss
    baseline_loss = model(
        token_ids,
        return_type="loss",
    )
    print(f"\n * Baseline loss: {float(baseline_loss):.4f}")

    # Check loss to see how hooks on activation values affect model loss
    # Hoocks are specified as a tuple of (name of specific hook, hook function)
    block_0_hook_name = tl_utils.get_act_name("pattern", 0)
    block_1_hook_name = tl_utils.get_act_name("pattern", 1)

    hooked_loss = model.run_with_hooks(
        token_ids,
        return_type="loss",
        fwd_hooks=[
            (block_0_hook_name, basic_hook_function),
            (block_1_hook_name, basic_hook_function),
        ],
    )

    print(f" * Hooked loss: {float(hooked_loss):.4f}")

    # Remove all hooks at end of run
    model.reset_hooks()

    ################################################################################
    # Calculate induction scores with hooks (equivalent to ActivationCache)
    ################################################################################

    print_section_header("Calculate induction scores with hooks")

    # Generate random sequence of tokens
    sequence_length = 50
    batch_size = 10

    random_repeated_token_sequence = generate_repeated_tokens(
        model.cfg.d_vocab, sequence_length, batch_size, device
    )

    # Tensor to store the induction score for each head
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    induction_scores = torch.zeros((n_layers, n_heads), device=model.cfg.device)

    # Initialise induction scoring hook function
    induction_score_hook = InductionScoreHook(sequence_length, induction_scores)

    # Run the hook function over all attention patterns in the model
    print("\n * Attention head induction scores:")

    model.run_with_hooks(
        random_repeated_token_sequence,
        return_type=None,
        fwd_hooks=[
            (lambda name: name.endswith("pattern"), induction_score_hook.compute)
        ],
    )

    print(
        f"\n * Thresholded induction scores ({n_layers}, {n_heads}): \n",
        induction_scores,
    )

    # Find 3D indices of non-zero elements in the induction_scores tensor
    non_zero_indices = induction_scores.nonzero()
    print(
        "\n * Heads with significant induction scores (layer, head): \n",
        non_zero_indices,
    )

    # Remove all hooks at end of run
    model.reset_hooks()


################################################################################
# Basic hook function
################################################################################


def basic_hook_function(attn_pattern: Tensor, hook: HookPoint) -> Tensor:
    """Hook function used to edit activation values."""
    attn_pattern = attn_pattern * 0.75

    return attn_pattern


################################################################################
# Hook to calculate induction scores
################################################################################


class InductionScoreHook:
    def __init__(self, sequence_length: int, induction_scores: Tensor):
        self.sequence_length = sequence_length
        self.induction_scores = induction_scores

    def compute(self, pattern: Tensor, hook: HookPoint):
        """
        Calculates the induction score for layers and heads in the transformer.

        Scores stored in the [layer, head] position of `induction_scores` tensor.
        """

        # Find the attention score for each repeated token
        offset = -(self.sequence_length - 1)
        attention_pattern_diagonal = pattern.diagonal(dim1=-2, dim2=-1, offset=offset)

        # Compute the average attention score for heads in each layer
        # averaged over the batch dimension
        average_score_of_heads = einops.reduce(
            attention_pattern_diagonal,
            "batch head_index position -> head_index",
            "mean",
        )

        average_score_of_heads = [round(float(s), 4) for s in average_score_of_heads]

        print(f"   * Layer {hook.layer()}:", average_score_of_heads)

        thresholded_scores = torch.tensor(
            list(map(lambda s: 0 if s < 0.4 else s, average_score_of_heads))
        )

        self.induction_scores[hook.layer(), :] = thresholded_scores


if __name__ == "__main__":
    device = "mps"
    model_name = "gpt2-small"
    # model_name = "meta-llama/Llama-3.2-1B"

    example_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."  # noqa

    main(model_name, example_text, device)
    print()

import torch

################################################################################
# Print header of new output section
################################################################################


def print_section_header(title: str) -> None:
    print("\n" + "#" * 80)
    print(f"# {title}")
    print("#" * 80)


################################################################################
# Generate sequence of repeated random tokens
################################################################################


def generate_repeated_tokens(
    vocab, seq_len: int, batch_size: int, device: str
) -> torch.Tensor:
    """Generates a sequence of repeated random tokens"""
    torch.manual_seed(42)

    repeating_tokens = torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.int64)
    full_sequence = torch.cat([repeating_tokens, repeating_tokens], dim=-1).to(device)

    return full_sequence

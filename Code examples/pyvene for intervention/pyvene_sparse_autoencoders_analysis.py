# LATENT ANALYSIS WITH SPARSE AUTOENCODERS
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pyvene import CollectIntervention, IntervenableModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Define sparse autoencoder architecture
class JumpReLUSAECollectIntervention(CollectIntervention):
    """Collect activations"""

    def __init__(self, **kwargs):
        # Note that we initialise to zeros as we're loading pre-trained weights.
        super().__init__(**kwargs, keep_last_dim=True)
        self.W_enc = nn.Parameter(
            torch.zeros(self.embed_dim, kwargs["low_rank_dimension"])
        )
        self.W_dec = nn.Parameter(
            torch.zeros(kwargs["low_rank_dimension"], self.embed_dim)
        )
        self.threshold = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_enc = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_dec = nn.Parameter(torch.zeros(self.embed_dim))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def forward(self, base, source=None, subspaces=None):
        acts = self.encode(base)

        return acts


if __name__ == "__main__":
    # Set up
    native_device = "mps"
    device = "cuda" if torch.cuda.is_available() else native_device
    device = torch.device(device)

    torch.set_grad_enabled(False)

    # Load model and tokenizer
    gemma_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    print("\n * Model & tokenizer loaded.")

    # Define prompt sequence
    prompt = "Would you be able to travel through time using a wormhole?"
    print(f" * Input prompt: {prompt}")

    # Tokenize prompt
    tokenized_prompt = tokenizer.encode(
        prompt, return_tensors="pt", add_special_tokens=True
    ).to(device)
    print(f" * Tokenized prompt: {tokenized_prompt}")

    # Generate predictions using model
    predicted_tokens = gemma_model.generate(
        input_ids=tokenized_prompt, max_new_tokens=50
    )
    predictions = tokenizer.decode(predicted_tokens[0])
    print(f" * Presicted response: {predictions[0]}")

    # Load sparse autoencoder model as Pyvene intervention for analyzing latents
    # Get pre-trained SAE parameters from HF
    LAYER = 20
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename=f"layer_{LAYER}/width_16k/average_l0_71/params.npz",
        force_download=False,
    )
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}

    # Create SAE instance with these parameters
    sparse_autoencoder = JumpReLUSAECollectIntervention(
        embed_dim=params["W_enc"].shape[0], low_rank_dimension=params["W_enc"].shape[1]
    )
    sparse_autoencoder.load_state_dict(pt_params, strict=False)
    sparse_autoencoder.to(device)

    # Connect the SAE to the model
    # (add the intervention to the model computation graph via the config)
    intervened_model = IntervenableModel(
        {
            "component": f"model.layers[{LAYER}].output",
            "intervention": sparse_autoencoder,
        },
        model=gemma_model,
    )

    # Generate predictions on prompt using intervened model
    predictions_from_intervened_model = intervened_model.forward(
        {"input_ids": tokenized_prompt}, return_dict=True
    )
    intervened_activations = predictions_from_intervened_model.collected_activations[0]

    print("\n * Actiavtions of model collected using sparse autoencoder:", end="\n\n")
    print((intervened_activations > 1).sum(-1), end="\n\n")
    print(intervened_activations.max(-1), end="\n\n")

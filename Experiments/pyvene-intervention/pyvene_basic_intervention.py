# Set up
import pandas as pd
import pyvene
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    geom_tile,
    ggplot,
    scale_y_log10,
    theme,
)
from pyvene import (
    IntervenableConfig,
    IntervenableModel,
    RepresentationConfig,
    VanillaIntervention,
    embed_to_distrib,
    format_token,
    top_vals,
)


# Patching on position-aligned tokens
def simple_position_config(model_type, component, layer):
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,  # layer
                component,  # component
                "pos",  # intervention unit
                1,  # max number of unit
            ),
        ],
        intervention_types=VanillaIntervention,
    )
    return config


if __name__ == "__main__":
    # Initialise pre-trained model
    config, tokenizer, gpt2_model = pyvene.create_gpt2()
    print("\n * Model & tokenizer loaded.")

    # Print the probability distribution generated for the token vocab on two sentences
    base = "The capital of Spain is"
    tokenized_base = tokenizer(base, return_tensors="pt")
    print(" * Base sequence tokenized.")

    predictions = gpt2_model(**tokenized_base)
    print(" * Predictions generated on base sequence.")

    distribution = embed_to_distrib(
        gpt2_model, predictions.last_hidden_state, logits=False
    )

    print("\n * Predicted token distributions:")
    print("   * Base sentence:")
    print(f"     * Input text: {base}")
    print("     * Output dist:", end="\n\n")
    top_vals(tokenizer, distribution[0][-1], n=10)

    source = "The capital of Italy is"
    tokenized_source = tokenizer(source, return_tensors="pt")

    predictions = gpt2_model(**tokenized_source)
    distribution = embed_to_distrib(
        gpt2_model, predictions.last_hidden_state, logits=False
    )

    print("\n   * Source sentence:")
    print(f"     * Input text: {source}")
    print("     * Output dist:", end="\n\n")
    top_vals(tokenizer, distribution[0][-1], n=10)

    # Patching on position-aligned tokens
    # Encode inputs
    base = tokenizer(base, return_tensors="pt")
    sources = [tokenizer(source, return_tensors="pt")]

    # Encode labels
    tokens = tokenizer.encode(" Madrid Rome")

    # Generate "position config" for each layer in model and position in input sequences
    print(
        "\n * Intervene in model layers over source/base input sequences:", end="\n\n"
    )
    data = []

    for layer_i in range(gpt2_model.config.n_layer):
        # Analyse tokens generated for base sequence
        position_config = simple_position_config(
            type(gpt2_model), "mlp_output", layer_i
        )
        intervened_model = IntervenableModel(position_config, gpt2_model)

        for position_i in range(len(base.input_ids[0])):
            _, counterfactual_outputs = intervened_model(
                base, sources, {"sources->base": position_i}
            )

            distribution = embed_to_distrib(
                gpt2_model, counterfactual_outputs.last_hidden_state, logits=False
            )

            for token in tokens:
                data.append(
                    {
                        "token": format_token(tokenizer, token),
                        "probability": float(distribution[0][-1][token]),
                        "model_layer": f"f{layer_i}",
                        "input_position": position_i,
                        "type": "mlp_output",
                    }
                )

        # Analyse tokens generated for source sequence
        position_config = simple_position_config(
            type(gpt2_model), "attention_input", layer_i
        )
        intervened_model = IntervenableModel(position_config, gpt2_model)

        for position_i in range(len(base.input_ids[0])):
            _, counterfactual_outputs = intervened_model(
                base, sources, {"sources->base": position_i}
            )

            distribution = embed_to_distrib(
                gpt2_model, counterfactual_outputs.last_hidden_state, logits=False
            )

            for token in tokens:
                data.append(
                    {
                        "token": format_token(tokenizer, token),
                        "probability": float(distribution[0][-1][token]),
                        "model_layer": f"a{layer_i}",
                        "input_position": position_i,
                        "type": "attention_input",
                    }
                )

    # Format analysis into graphs
    df = pd.DataFrame(data)

    df["model_layer"] = df["model_layer"].astype("category")
    df["token"] = df["token"].astype("category")

    nodes = []
    for layer in range(gpt2_model.config.n_layer - 1, -1, -1):
        nodes.append(f"f{layer}")
        nodes.append(f"a{layer}")

    df["model_layer"] = pd.Categorical(
        df["model_layer"], categories=nodes[::-1], ordered=True
    )

    # Predicted label token at each model layer and position in the input sequence
    print(
        "\n * Generated graph: 'Predicted output token at each model layer and input position'."
    )
    graph = (
        ggplot(df)
        + geom_tile(
            aes(
                x="input_position",
                y="model_layer",
                fill="probability",
                color="probability",
            )
        )
        + facet_wrap("~token")
        + theme(
            axis_text_x=element_text(rotation=90),
            plot_title="Predicted output token probability at each model layer and input position",
        )
    )
    graph.save(
        "outputs/Prediction-probability-over-layers-and-positions.png", verbose=False
    )

    # Probability of the label token at each model layer
    filtered = df
    filtered = filtered[filtered["input_position"] == 4]

    print(
        " * Generated graph: 'Probability of the label token at each model layer'.",
        end="\n\n",
    )
    graph = (
        ggplot(filtered)
        + geom_bar(aes(x="model_layer", y="probability", fill="token"), stat="identity")
        + theme(
            axis_text_x=element_text(rotation=90),
            legend_position="none",
            plot_title="Probability of the label token at each model layer",
        )
        + scale_y_log10()
        + facet_wrap("~token", ncol=1)
    )
    graph.save("outputs/Label-probability-over-layers.png", verbose=False)

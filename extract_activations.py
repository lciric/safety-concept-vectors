"""
Extract residual stream activations from stories and compute concept directions.

For each story:
  1. Tokenize
  2. Forward pass with activation caching
  3. Extract residual stream at 3 positions: last token, middle token, mean

For each concept:
  direction = mean(concept_activations) - mean(neutral_activations)

Usage:
    python extract_activations.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
    python extract_activations.py --model meta-llama/Llama-3.3-70B-Instruct --device cuda --use-nnsight
"""

import torch
import numpy as np
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def load_stories(stories_dir):
    """Load all story JSON files. Returns dict: concept_name -> list of story texts."""
    stories = {}
    for f in sorted(Path(stories_dir).glob("*.json")):
        concept = f.stem  # e.g. "eval-awareness", "neutral"
        with open(f) as fh:
            data = json.load(fh)
        texts = [s["text"] for s in data if s.get("text")]
        stories[concept] = texts
        print(f"  Loaded {len(texts)} stories for '{concept}'")
    return stories


def extract_with_transformerlens(model, texts, batch_size=1, max_length=512):
    """
    Extract residual stream activations using TransformerLens.
    Returns dict with 3 extraction modes, each of shape (n_texts, n_layers, d_model).
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    all_last = []
    all_middle = []
    all_mean = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i:i+batch_size]

        for text in batch_texts:
            # Tokenize with truncation
            tokens = model.to_tokens(text, prepend_bos=True)
            if tokens.shape[1] > max_length:
                tokens = tokens[:, :max_length]

            seq_len = tokens.shape[1]
            mid_pos = seq_len // 2
            last_pos = seq_len - 1

            # Forward pass with cache
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: "resid_post" in name
                )

            # Extract at each layer
            last_token_acts = []
            middle_token_acts = []
            mean_acts = []

            for layer in range(n_layers):
                resid = cache["resid_post", layer]  # (1, seq_len, d_model)

                last_token_acts.append(resid[0, last_pos, :].cpu().float().numpy())
                middle_token_acts.append(resid[0, mid_pos, :].cpu().float().numpy())
                mean_acts.append(resid[0, :, :].mean(dim=0).cpu().float().numpy())

            all_last.append(np.stack(last_token_acts))      # (n_layers, d_model)
            all_middle.append(np.stack(middle_token_acts))
            all_mean.append(np.stack(mean_acts))

            # Free cache memory
            del cache
            torch.cuda.empty_cache()

    return {
        "last_token": np.stack(all_last),    # (n_texts, n_layers, d_model)
        "middle_token": np.stack(all_middle),
        "mean_token": np.stack(all_mean),
    }


def extract_with_nnsight(model_name, texts, device="cuda", batch_size=1, max_length=512):
    """
    Extract residual stream activations using nnsight (for large models).
    Returns dict with 3 extraction modes, each of shape (n_texts, n_layers, d_model).
    """
    from nnsight import LanguageModel
    import transformers

    print(f"Loading {model_name} via nnsight...")
    model = LanguageModel(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    all_last = []
    all_middle = []
    all_mean = []

    for i in tqdm(range(len(texts)), desc="Extracting (nnsight)"):
        tokens = tokenizer(
            texts[i],
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)

        seq_len = tokens["input_ids"].shape[1]
        mid_pos = seq_len // 2
        last_pos = seq_len - 1

        last_token_acts = []
        middle_token_acts = []
        mean_acts = []

        with model.trace(tokens["input_ids"]):
            for layer in range(n_layers):
                resid = model.model.layers[layer].output[0].save()

        # Process saved activations
        for layer in range(n_layers):
            r = resid[layer].value  # (1, seq_len, d_model)
            last_token_acts.append(r[0, last_pos, :].cpu().float().numpy())
            middle_token_acts.append(r[0, mid_pos, :].cpu().float().numpy())
            mean_acts.append(r[0, :, :].mean(dim=0).cpu().float().numpy())

        all_last.append(np.stack(last_token_acts))
        all_middle.append(np.stack(middle_token_acts))
        all_mean.append(np.stack(mean_acts))

        torch.cuda.empty_cache()

    return {
        "last_token": np.stack(all_last),
        "middle_token": np.stack(all_middle),
        "mean_token": np.stack(all_mean),
    }


def compute_directions(concept_activations, neutral_activations):
    """
    Compute concept direction at each layer.
    direction[layer] = mean(concept[layer]) - mean(neutral[layer])
    Normalized to unit length.

    Args:
        concept_activations: (n_concept, n_layers, d_model)
        neutral_activations: (n_neutral, n_layers, d_model)
    Returns:
        directions: (n_layers, d_model) — unit vectors
        raw_directions: (n_layers, d_model) — unnormalized (for magnitude analysis)
    """
    concept_mean = concept_activations.mean(axis=0)  # (n_layers, d_model)
    neutral_mean = neutral_activations.mean(axis=0)

    raw_directions = concept_mean - neutral_mean
    norms = np.linalg.norm(raw_directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # avoid division by zero
    directions = raw_directions / norms

    return directions, raw_directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stories-dir", type=str, default="stories")
    parser.add_argument("--output-dir", type=str, default="activations")
    parser.add_argument("--use-nnsight", action="store_true", help="Use nnsight instead of TransformerLens")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--extraction-mode", type=str, default="all",
                        choices=["last_token", "middle_token", "mean_token", "all"])
    args = parser.parse_args()

    # Load stories
    print(f"\nLoading stories from {args.stories_dir}/")
    stories = load_stories(args.stories_dir)

    if "neutral" not in stories:
        print("ERROR: No neutral.json found. Generate it first with --neutral.")
        return

    concepts = [c for c in stories.keys() if c != "neutral"]
    print(f"\nFound {len(concepts)} concepts + neutral ({len(stories['neutral'])} stories)")

    # Load model and extract
    if args.use_nnsight:
        print(f"\nUsing nnsight for {args.model}")
        # Extract neutral
        print("\nExtracting neutral corpus...")
        neutral_acts = extract_with_nnsight(args.model, stories["neutral"],
                                             args.device, max_length=args.max_length)

        # Extract each concept
        concept_acts = {}
        for concept in concepts:
            print(f"\nExtracting {concept} ({len(stories[concept])} stories)...")
            concept_acts[concept] = extract_with_nnsight(
                args.model, stories[concept],
                args.device, max_length=args.max_length
            )
    else:
        print(f"\nLoading {args.model} via TransformerLens...")
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(
            args.model,
            device=args.device,
            dtype=torch.float16
        )
        print(f"  Loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

        # Extract neutral
        print(f"\nExtracting neutral corpus ({len(stories['neutral'])} stories)...")
        neutral_acts = extract_with_transformerlens(model, stories["neutral"],
                                                     max_length=args.max_length)

        # Extract each concept
        concept_acts = {}
        for concept in concepts:
            print(f"\nExtracting {concept} ({len(stories[concept])} stories)...")
            concept_acts[concept] = extract_with_transformerlens(
                model, stories[concept], max_length=args.max_length
            )

    # Compute directions for each extraction mode
    modes = ["last_token", "middle_token", "mean_token"] if args.extraction_mode == "all" else [args.extraction_mode]

    model_name_clean = args.model.split("/")[-1]
    output_base = os.path.join(args.output_dir, model_name_clean)

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Computing directions ({mode})")
        print(f"{'='*60}")

        mode_dir = os.path.join(output_base, mode)
        os.makedirs(mode_dir, exist_ok=True)

        neutral_data = neutral_acts[mode]  # (n_neutral, n_layers, d_model)

        # Save neutral activations
        np.save(os.path.join(mode_dir, "neutral_activations.npy"), neutral_data)
        print(f"  Neutral: {neutral_data.shape}")

        all_directions = {}

        for concept in concepts:
            concept_data = concept_acts[concept][mode]  # (n_concept, n_layers, d_model)

            # Save concept activations
            np.save(os.path.join(mode_dir, f"{concept}_activations.npy"), concept_data)

            # Compute direction
            directions, raw_directions = compute_directions(concept_data, neutral_data)

            all_directions[concept] = directions
            np.save(os.path.join(mode_dir, f"{concept}_direction.npy"), directions)

            # Magnitude of raw direction per layer (how "strong" the concept signal is)
            magnitudes = np.linalg.norm(raw_directions, axis=1)

            print(f"  {concept}: activations {concept_data.shape}, "
                  f"direction magnitude range [{magnitudes.min():.2f}, {magnitudes.max():.2f}]")

        # Save all directions in one file for convenience
        np.savez(
            os.path.join(mode_dir, "all_directions.npz"),
            **all_directions
        )

        # Compute cosine similarity matrix between all concept directions at each layer
        n_concepts = len(concepts)
        n_layers = list(all_directions.values())[0].shape[0]

        # At best layer (we'll determine this during probing, for now use middle layer)
        mid_layer = n_layers // 2
        similarity_matrix = np.zeros((n_concepts, n_concepts))
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                d1 = all_directions[c1][mid_layer]
                d2 = all_directions[c2][mid_layer]
                similarity_matrix[i, j] = np.dot(d1, d2)

        np.save(os.path.join(mode_dir, "concept_similarity_matrix.npy"), similarity_matrix)

        # Save concept names for reference
        with open(os.path.join(mode_dir, "concept_names.json"), "w") as f:
            json.dump(concepts, f)

        print(f"\n  Saved to {mode_dir}/")

    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"  Stories: {args.stories_dir}/")
    print(f"  Activations: {output_base}/")
    print(f"  Modes: {modes}")
    print(f"  Concepts: {len(concepts)}")
    print(f"  Neutral stories: {len(stories['neutral'])}")
    n_layers = list(concept_acts.values())[0]["last_token"].shape[1]
    d_model = list(concept_acts.values())[0]["last_token"].shape[2]
    print(f"  Layers: {n_layers}, d_model: {d_model}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

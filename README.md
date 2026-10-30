# Whisper-small Hindi ASR: Fine-tuning on FLEURS

Fine-tuning OpenAI's Whisper-small for Hindi speech recognition using the Google FLEURS dataset, and comparing zero-shot vs. fine-tuned word error rate (WER).

## Results

Evaluated on the FLEURS Hindi (`hi_in`) test set, 418 utterances.

| Model | WER |
|---|---|
| Whisper-small (zero-shot) | 66.10% |
| Whisper-small (fine-tuned, 1,000 steps) | 26.56% |

An absolute reduction of 39.54 points (59.8% relative).

## Setup

- **Model:** `openai/whisper-small`
- **Dataset:** `google/fleurs`, config `hi_in`
- **Training:** 1,000 steps, batch size 8 with gradient accumulation of 2, learning rate 1e-5, fp16, gradient checkpointing
- **Hardware:** single Tesla T4 GPU (Kaggle)
- **Metric:** WER via `jiwer`, with text normalisation (lowercasing, punctuation and danda removal) applied to both references and predictions
- **Libraries:** Hugging Face Transformers, Datasets, PyTorch, jiwer

## Error analysis

The notebook saves a per-utterance comparison (`error_analysis.csv`) and prints the worst fine-tuned utterances. The remaining errors are mostly English loanword spellings and proper nouns.

## How to run

1. Open `whisper-small-hindi-finetune.ipynb` on Kaggle (GPU, Internet on) or Colab (T4).
2. Run the cells top to bottom. The language and number of training steps are set in the config cell.
3. The final cell launches an optional Gradio demo for microphone or file input.

## Notes

- The notebook is set up for Hindi (`hi`). The language is configurable in the config cell, but only Hindi was run and reported here.
- Results come from a single short run (1,000 steps) and are not tuned.

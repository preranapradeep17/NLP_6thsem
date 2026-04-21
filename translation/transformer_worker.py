import json
import os
import sys
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from transformers import MarianMTModel, MarianTokenizer

from transformer_cache import resolve_local_hf_snapshot


def main():
    source_lang = sys.argv[1]
    target_lang = sys.argv[2]
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    model_path = resolve_local_hf_snapshot(model_name)

    try:
        tokenizer = MarianTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            local_files_only=model_path != model_name,
        )
        model = MarianMTModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            local_files_only=model_path != model_name,
        )
        print(json.dumps({"status": "ready"}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        raise

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
            text = payload["text"]
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            outputs = model.generate(**inputs)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(json.dumps({"translation": translation}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()

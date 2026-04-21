import json
import os
import subprocess
import sys

from transformers import MarianMTModel, MarianTokenizer

from translation.transformer_cache import resolve_local_hf_snapshot


class TransformerTranslator:
    def __init__(self, source_lang="en", target_lang="fr"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        self.model_path = resolve_local_hf_snapshot(self.model_name)
        self.worker = None

        print(f"⏳ Loading Transformer Model: {self.model_name}...")

        if "tensorflow" in sys.modules:
            self._start_worker()
        else:
            self.tokenizer = MarianTokenizer.from_pretrained(
                self.model_path,
                use_fast=False,
                local_files_only=self.model_path != self.model_name,
            )
            self.model = MarianMTModel.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=False,
                local_files_only=self.model_path != self.model_name,
            )

        print("✅ Transformer Translation Model Loaded!")

    def _start_worker(self):
        worker_path = os.path.join(os.path.dirname(__file__), "transformer_worker.py")
        self.worker = subprocess.Popen(
            [sys.executable, "-u", worker_path, self.source_lang, self.target_lang],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        ready_line = self.worker.stdout.readline().strip()
        if not ready_line:
            raise RuntimeError("Transformer worker failed to start.")

        payload = json.loads(ready_line)
        if payload.get("status") != "ready":
            raise RuntimeError(payload.get("error", "Transformer worker failed to load."))

    def translate(self, text):
        if self.worker is not None:
            return self._translate_via_worker(text)

        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _translate_via_worker(self, text):
        if self.worker.poll() is not None:
            raise RuntimeError("Transformer worker exited unexpectedly.")

        request = json.dumps({"text": text})
        self.worker.stdin.write(request + "\n")
        self.worker.stdin.flush()

        response_line = self.worker.stdout.readline().strip()
        if not response_line:
            raise RuntimeError("Transformer worker returned no output.")

        payload = json.loads(response_line)
        if "error" in payload:
            raise RuntimeError(payload["error"])

        return payload["translation"]

    def close(self):
        if self.worker is None or self.worker.poll() is not None:
            return

        self.worker.terminate()
        try:
            self.worker.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.worker.kill()

    def __del__(self):
        self.close()

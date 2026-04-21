import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

warnings.filterwarnings("ignore")

from translation.seq2seq import DLTranslator


def main():
    if len(sys.argv) < 2:
        raise SystemExit("No text provided.")

    translator = DLTranslator()
    print(translator.translate(sys.argv[1]))


if __name__ == "__main__":
    main()

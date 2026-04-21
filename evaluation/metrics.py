from sklearn.metrics import accuracy_score, f1_score


def evaluate_sentiment(y_true, y_pred):
    """
    Calculates evaluation metrics for sentiment analysis.

    Parameters:
    y_true : list -> actual labels (0 or 1)
    y_pred : list -> predicted labels (0 or 1)

    Returns:
    dict -> Accuracy and F1 Score
    """

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        "Accuracy": round(accuracy, 4),
        "F1-Score": round(f1, 4)
    }


# 🔜 Future Extensions (for your project report / viva)

def evaluate_translation():
    """
    Placeholder for BLEU Score implementation.
    """
    return "BLEU evaluation coming soon"


def evaluate_summarization():
    """
    Placeholder for ROUGE Score implementation.
    """
    return "ROUGE evaluation coming soon"


# 🔍 Quick test
if __name__ == "__main__":
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]

    results = evaluate_sentiment(y_true, y_pred)

    print("\n📊 Evaluation Results")
    print(results)
import sys

from torchtext.data.metrics import bleu_score

from src.interpret import interpret

# sys.path.insert(0, "/Users/mcciupek/Documents/42/AI/Hand2Text/src")


if __name__ == "__main__":
    pred = ["My full pytorch test"]
    y = ["My full pytorch test"]
    score = interpret(pred, y)
    print(score)

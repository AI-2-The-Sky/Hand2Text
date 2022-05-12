import sys

from torchtext.data.metrics import bleu_score

from src.interpret import interpret

if __name__ == "__main__":
    pred = ["My full pytorch test"]
    y = ["My full pytorch test"]
    score = interpret(pred, y)
    print(score)

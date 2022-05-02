import sys

# adding Folder_2 to the system path
sys.path.insert(0, "/Users/mcciupek/Documents/42/AI/Hand2Text/src")

from torchtext.data.metrics import bleu_score

from interpret import interpret

if __name__ == "__main__":
    pred = ["My full pytorch test"]
    y = ["My full pytorch test"]
    score = interpret(pred, y)
    print(score)

from models.components.BlueScore import BleuScore


def tokenize(strlist: list):
    return [s.split() for s in strlist]


def interpret(pred: list, target: list):
    """
    Calculate the BLEU score of the output of the model
    Input:
        pred: the output of the model
        target: the correct sentence
    Output:
        BLEU score of the model
    """
    score = BleuScore()
    pred = tokenize(pred)
    target = tokenize(target)
    return score(pred, [target]).item()

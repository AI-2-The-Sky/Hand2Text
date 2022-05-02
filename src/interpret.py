from models.components.BlueScore import BleuScore


def interpret(pred: list, target: list) -> float:
    """
    Calculate the BLEU score of the output of the model
    Input:
        output: the output of the model
    Output:
        the BLEU score of the model
    """
    score = BleuScore()
    return score(pred, [elem for elem in target])


if __name__ == "__main__":
    pred = [["My", "full", "pytorch", "test"], ["Another", "Sentence"]]
    y = [[["My", "full", "pytorch", "test"], ["Completely", "Different"]], [["No", "Match"]]]
    score = interpret(pred, y)
    print(score)

    pred = [["My", "full", "pytorch", "test"]]
    y = [[["My", "full", "pytorch", "test"]]]
    score = interpret(pred, y)
    print(score)

    pred = ["Hello", "world"]
    y = ["Hello", "world"]
    pred = [["Hello", "world", "how", "are", "you"]]
    y = [[["Hello", "world", "how", "are", "you"]]]
    score = interpret(pred, y)
    print(score)

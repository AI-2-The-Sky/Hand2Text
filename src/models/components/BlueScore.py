import torch
from torchmetrics import Metric
from torchtext.data.metrics import bleu_score


class BleuScore(Metric):
    """Compute Bleu score of a translated text compared to one or more references corpus."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # self.add_state("score", default=torch.tensor(0))

    def update(self, candidate_corpus, references_corpus, max_n=4, weights=[0.0, 1.0, 0.0, 0.0]):
        # candidate_corpus = [['My', 'full', 'pytorch', 'test']] –> an iterable of candidate translations.
        # references_corpus = [[['My', 'full', 'pytorch', 'test']]] –> an iterable of iterables of reference translations
        self.candidate_corpus = [candidate_corpus]
        self.reference_corpus = [[references_corpus]]
        self.max_n = max_n
        self.weights = weights

    def compute(self):
        """Compute Bleu score.

        Return:
                Bleu score: tensor
        """
        result_score = bleu_score(
            self.candidate_corpus, self.reference_corpus, self.max_n, self.weights
        )

        return torch.tensor(result_score)

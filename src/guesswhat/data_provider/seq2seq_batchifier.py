import numpy as np
import collections

from generic.data_provider.batchifier import AbstractBatchifier, BatchifierSplitMode, batchifier_split_helper
from generic.data_provider.nlp_utils import padder


class QuestionerBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, status=list(), split_mode=BatchifierSplitMode.NoSplit, **kwargs):
        self.tokenizer = tokenizer
        self.sources = sources
        self.status = status
        self.split_mode = split_mode
        self.kwargs = kwargs

    def split(self, games):
        return batchifier_split_helper(games, split_mode=self.split_mode)

    def filter(self, games):

        if len(self.status) > 0:
            games = [g for g in games if g.status in self.status]

        return games

    def apply(self, games):

        batch = collections.defaultdict(list)
        batch_size = len(games)

        for i, game in enumerate(games):

            batch['raw'].append(game)

            # Encode question answers
            q_tokens = [self.tokenizer.encode(q) for q in game.questions]
            a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

            # Flatten questions/answers except the last one
            dialogue = [self.tokenizer.start_token]  # Add start token
            for q_tok, a_tok in zip(q_tokens[:-1], a_tokens[:-1]):
                dialogue += q_tok
                dialogue += a_tok

            dialogue += [self.tokenizer.stop_dialogue]  # Add STOP token

            batch["dialogue"].append(dialogue)
            batch["question"].append(q_tokens[-1])

            # image
            img = game.image.get_image()
            if img is not None:
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

        # Pad dialogue tokens tokens
        batch['dialogue'], batch['seq_length_dialogue'] = padder(batch['dialogues'], padding_symbol=self.tokenizer.padding_token)
        batch['question'], batch['seq_length_question'] = padder(batch['question'], padding_symbol=self.tokenizer.padding_token)

        return batch





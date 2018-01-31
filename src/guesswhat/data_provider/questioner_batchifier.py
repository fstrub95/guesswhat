import numpy as np
import collections

from generic.data_provider.batchifier import AbstractBatchifier, BatchifierSplitMode, batchifier_split_helper

from generic.data_provider.image_preprocessors import get_spatial_feat
from generic.data_provider.nlp_utils import padder, padder_3d


class Seq2SeqBatchifier(AbstractBatchifier):

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

            dialogue += [self.tokenizer.stop_token]  # Add STOP token

            if game.is_full_dialogue:
                dialogue += [self.tokenizer.stop_dialogue]

            batch["dialogue"].append(dialogue)
            batch["question"].append(q_tokens[-1])

            # image
            img = game.image.get_image()
            if img is not None:
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

        # Pad dialogue tokens tokens
        batch['dialogue'], batch['seq_length_dialogue'] = padder(batch['dialogue'], padding_symbol=self.tokenizer.padding_token)
        batch['question'], batch['seq_length_question'] = padder(batch['question'], padding_symbol=self.tokenizer.padding_token)

        return batch


class LSTMBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, status=list(), **kwargs):
        self.tokenizer = tokenizer
        self.sources = sources
        self.status = status
        self.kwargs = kwargs

    def filter(self, games):
        if len(self.status) > 0:
            return [g for g in games if g.status in self.status]
        else:
            return games

    def apply(self, games):

        batch = collections.defaultdict(list)
        batch_size = len(games)

        all_answer_indices = []
        for i, game in enumerate(games):

            batch['raw'].append(game)

            # Flattened question answers
            q_tokens = [self.tokenizer.encode(q) for q in game.questions]
            a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

            tokens = [self.tokenizer.start_token]  # Add start token
            answer_indices = []
            cur_index = 0
            for q_tok, a_tok in zip(q_tokens, a_tokens):
                tokens += q_tok
                tokens += a_tok

                # Compute index of answer in the full dialogue
                answer_indices += [cur_index + len(q_tok) + 1]
                cur_index = answer_indices[-1]

            tokens += [self.tokenizer.stop_dialogue]  # Add STOP token

            batch["dialogues"].append(tokens)
            all_answer_indices.append(answer_indices)

            # Object embedding
            obj_spats, obj_cats = [], []
            for index, obj in enumerate(game.objects):
                spatial = get_spatial_feat(obj.bbox, game.image.width, game.image.height)
                category = obj.category_id

                if obj.id == game.object_id:
                    batch['targets_category'].append(category)
                    batch['targets_spatial'].append(spatial)
                    batch['targets_index'].append(index)

                obj_spats.append(spatial)
                obj_cats.append(category)
            batch['obj_spats'].append(obj_spats)
            batch['obj_cats'].append(obj_cats)

            # image
            img = game.image.get_image()
            if img is not None:
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

        # Pad dialogue tokens tokens
        batch['dialogues'], batch['seq_length'] = padder(batch['dialogues'], padding_symbol=self.tokenizer.padding_token)
        seq_length = batch['seq_length']
        max_length = max(seq_length)

        # Compute the token mask
        batch['padding_mask'] = np.ones((batch_size, max_length), dtype=np.float32)
        for i in range(batch_size):
            batch['padding_mask'][i, (seq_length[i] + 1):] = 0.

        # Compute the answer mask
        batch['answer_mask'] = np.ones((batch_size, max_length), dtype=np.float32)
        for i in range(batch_size):
            batch['answer_mask'][i, all_answer_indices[i]] = 0.

        # Pad objects
        batch['obj_spats'], obj_length = padder_3d(batch['obj_spats'])
        batch['obj_cats'], obj_length = padder(batch['obj_cats'])


        batch['num_obj'] = obj_length

        # Compute the object mask
        # max_objects = max(obj_length)
        # batch['obj_mask'] = np.zeros((batch_size, max_objects), dtype=np.float32)
        # for i in range(batch_size):
        #     batch['obj_mask'][i, :obj_length[i]] = 1.0

        return batch

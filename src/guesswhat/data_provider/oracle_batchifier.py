import numpy as np
import collections
from PIL import Image
import copy

from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import get_spatial_feat, resize_image
from generic.data_provider.nlp_utils import padder, padder_3d
from itertools import chain


class OracleBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, split_question=True, glove=None, ignore_NA=False, status=list()):
        self.tokenizer = tokenizer
        self.sources = sources
        self.status = status
        self.split_question = split_question
        self.ignore_NA = ignore_NA
        self.glove = glove

    def filter(self, games):

        if len(self.status) > 0:
            games = [g for g in games if g.status in self.status]

        if self.ignore_NA:
            games = [g for g in games if g.answers[-1] != "N/A"]

        return games


    def apply(self, games):
        sources = self.sources
        tokenizer = self.tokenizer
        batch = collections.defaultdict(list)

        for i, game in enumerate(games):
            batch['raw'].append(game)

            image = game.image

            if 'question' in sources:
                questions = []
                for q, a in zip(game.questions[:-1], game.answers[:-1]):
                    questions.append(tokenizer.encode(q))
                    questions.append(tokenizer.encode(a, is_answer=True))
                questions.append(tokenizer.encode(game.questions[-1]))
                batch['question'].append(list(chain.from_iterable(questions)))

            if 'glove' in self.sources:
                questions = []
                for q, a in zip(game.questions[:-1], game.answers[:-1]):
                    questions.append(self.tokenizer.tokenize_question(q))
                    questions.append([self.tokenizer.format_answer(a)])
                questions.append(tokenizer.tokenize_question(game.questions[-1]))
                questions = list(chain.from_iterable(questions))
                glove_vectors = self.glove.get_embeddings(questions)
                batch['glove'].append(glove_vectors)

            if 'answer' in sources:
                batch['answer'].append(tokenizer.encode_oracle_answer(game.answers[-1], sparse=False))

            if 'category' in sources:
                batch['category'].append(game.object.category_id)

            if 'spatial' in sources:
                spat_feat = get_spatial_feat(game.object.bbox, image.width, image.height)
                batch['spatial'].append(spat_feat)

            if 'crop' in sources:
                batch['crop'].append(game.object.get_crop())

            if 'image' in sources:
                batch['image'].append(image.get_image())

            if 'mask' in sources:
                assert "image" in batch, "mask input require the image source"
                mask = game.object.get_mask()
                mask = mask.astype(np.float32)
                ft_width, ft_height = batch['image'][-1].shape[1],\
                                     batch['image'][-1].shape[0] # Use the image feature size (not the original img size)

                mask = resize_image(Image.fromarray(mask), height=ft_height, width=ft_width)
                batch['mask'].append(np.array(mask))


        # pad the questions
        if 'question' in sources:
            batch['question'], batch['seq_length'] = padder(batch['question'], padding_symbol=tokenizer.word2i['<padding>'])

        if 'glove' in sources:
            batch['glove'], _ = padder_3d(batch['glove'])

        return batch




import numpy as np
import re

from guesswhat.models.qgen.qgen_sampling_wrapper import QGenSamplingWrapper
from guesswhat.models.qgen.qgen_beamsearch_wrapper import QGenBSWrapper
from generic.tf_utils.evaluator import Evaluator

from guesswhat.models.looper.tools import list_to_padded_tokens


class QGenWrapperDecoder(object):
    def __init__(self, qgen, tokenizer, max_tokens, k_best):

        self.sampling_out = qgen.create_sampling_graph(start_token=tokenizer.start_token,
                                                       stop_token=tokenizer.stop_token,
                                                       max_tokens=max_tokens)

        self.greedy_out = qgen.create_greedy_graph(start_token=tokenizer.start_token,
                                                   stop_token=tokenizer.stop_token,
                                                   max_tokens=max_tokens)

        self.beam_out = qgen.create_beam_graph(start_token=tokenizer.start_token,
                                               stop_token=tokenizer.stop_token,
                                               max_tokens=max_tokens,
                                               k_best=k_best)

        self.qgen = qgen
        self.evaluator = None

        self.tokenizer = tokenizer

        self.prev_dialogues = None

    def initialize(self, sess):
        sources = self.qgen.get_sources(sess)
        self.evaluator = Evaluator(sources, self.qgen.scope_name,
                                   network=self.qgen, tokenizer=self.tokenizer)

    def reset(self, batch_size):
        self.prev_dialogues = None
        # TODO Potential optimization -> store encoder state

    def sample_next_question(self, sess, prev_answers, game_data, mode):

        if mode == "sampling":
            output = self.sampling_out
        elif mode == "greedy":
            output = self.greedy_out
        elif mode == "beam_search":
            output = self.beam_out

        else:
            assert False, "Invalid sampling mode: {}".format(mode)

        # Retrieve the history of question/answer pairs
        # TODO Potential optimization -> store encoder state
        if self.prev_dialogues is None:
            self.prev_dialogues = prev_answers
        else:
            self.prev_dialogues = [d + a for d, a in zip(self.prev_dialogues, prev_answers)]


        dialogue, seq_length = list_to_padded_tokens(self.prev_dialogues, self.tokenizer)

        # Prepare batch of data
        game_data["dialogue"] = dialogue
        game_data["seq_length_dialogue"] = seq_length
        game_data["is_training"] = False

        # sample the next questions
        assert self.tokenizer.padding_token == 0
        next_padded_questions, next_question_seq_length = self.evaluator.execute(sess, output=output, batch=game_data)

        # unpad the question
        next_questions = [list(q[:sq]) for q, sq in zip(next_padded_questions, next_question_seq_length)]
        self.prev_dialogues = [pq + q for pq, q in zip(self.prev_dialogues, next_questions)]

        return next_padded_questions, next_questions, next_question_seq_length


# This is very ugly code that must be refactored.
# To avoid breaking future code, we hide the implementation behind this Decorator
# Implementation of sampling was updated for speed reason while eam search rely ion legacy code
# Therefore, their internal implementation differs. that iw why we put a wrapper to hide technical detail in the looper


class QGenWrapperLSTM(object):
    def __init__(self, qgen, tokenizer, max_length, k_best):

        self.sampling_wrapper = QGenSamplingWrapper(qgen, tokenizer, max_length)
        self.bs_wrapper = QGenBSWrapper(qgen, tokenizer, max_length, k_best)
        self.qgen = qgen

    def initialize(self, sess):
        self.sampling_wrapper.initialize(sess)
        self.bs_wrapper.initialize(sess)

    def reset(self, batch_size):
        self.sampling_wrapper.reset(batch_size)
        self.bs_wrapper.reset(batch_size)

    def sample_next_question(self, sess, prev_answers, game_data, mode):

        if mode == "sampling":
            return self.sampling_wrapper.sample_next_question(sess, prev_answers, game_data, greedy=False)
        elif mode == "greedy":
            return self.sampling_wrapper.sample_next_question(sess, prev_answers, game_data, greedy=True)
        elif mode == "beam_search":
            return self.bs_wrapper.sample_next_question(sess, prev_answers, game_data)
        else:
            assert False, "Invalid samppling mode: {}".format(mode)


class QGenUserWrapper(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def initialize(self, sess):
        pass

    def reset(self, batch_size):
        pass

    def sample_next_question(self, _, prev_answers, game_data, **__):

        if prev_answers[0] == self.tokenizer.start_token:
            print("Type the character '(S)top' when you want to guess the object")
        else:
            print("A :", self.tokenizer.decode(prev_answers[0]))

        print()
        while True:
            question = input('Q: ')
            if question != "":
                break

        # Stop the dialogue
        if question == "S" or question == "Stop":
            tokens = [self.tokenizer.stop_dialogue]

        # Stop the question (add stop token)
        else:
            question = re.sub('\?', '', question) # remove question tags if exist
            question +=  " ?"
            tokens = self.tokenizer.apply(question)

        return [tokens], np.array([tokens]), [len(tokens)]

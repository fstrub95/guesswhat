import argparse
import logging
import os
import collections

import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump
from generic.utils.thread_pool import create_cpu_pool
from generic.data_provider.image_loader import get_img_builder
from generic.data_provider.nlp_utils import GloveEmbeddings

from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.questioner_batchifier import LSTMBatchifier
from guesswhat.data_provider.seq2seq_batchifier import Seq2SeqBatchifier

from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer

from guesswhat.models.qgen.qgen_lstm_network import QGenNetworkLSTM
from guesswhat.models.qgen.qgen_decoder_network import QGenNetworkDecoder

if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('QGen network baseline!')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-config", type=str, help='Config file')
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")
    parser.add_argument("-glove_file", type=str, default="glove_dict.pkl", help="Glove file name")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-continue_exp", type=bool, default=False, help="Continue previously started experiment?")
    parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")
    parser.add_argument("-no_games_to_load", type=int, help="Number of games to use during training Default : all")

    args = parser.parse_args()
    config, exp_identifier, save_path = load_config(args.config, args.exp_dir, args)
    logger = logging.getLogger()

    ###############################
    #  LOAD DATA
    #############################

    # Load image
    logger.info('Loading images..')
    image_loader = get_img_builder(config['model']['image'], args.img_dir)
    crop_loader = None  # get_img_loader(config, 'crop', args.image_dir)

    # Load data
    logger.info('Loading data..')
    trainset = Dataset(args.data_dir, "train", image_loader, crop_loader, args.no_games_to_load)
    validset = Dataset(args.data_dir, "valid", image_loader, crop_loader, args.no_games_to_load)
    testset = Dataset(args.data_dir, "test", image_loader, crop_loader, args.no_games_to_load)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file))

    # Load glove
    glove = None
    if config["model"]["dialogue"]['glove']:
        logger.info('Loading glove..')
        glove = GloveEmbeddings(args.glove_file)

    # Build Network
    logger.info('Building network..')
    # network = QGenNetworkLSTM(config["model"], num_words=tokenizer.no_words, policy_gradient=False)
    network = QGenNetworkDecoder(config["model"], num_words=tokenizer.no_words, policy_gradient=False)

    # Build Optimizer
    logger.info('Building optimizer..')
    optimizer, outputs = create_optimizer(network, config["optimizer"])

    ###############################
    #  START TRAINING
    #############################

    # Load config
    batch_size = config['optimizer']['batch_size']
    no_epoch = config["optimizer"]["no_epoch"]

    # Store experiments
    data = dict()
    data["hash_id"] = exp_identifier
    data["config"] = config
    data["args"] = args
    data["loss"] = collections.defaultdict(list)

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()

    # CPU/GPU option
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sources = network.get_sources(sess)
        logger.info("Sources: " + ', '.join(sources))

        sess.run(tf.global_variables_initializer())
        start_epoch = load_checkpoint(sess, saver, args, save_path)

        # Create training tools
        evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
        # batchifier = LSTMBatchifier(tokenizer, sources, status=('success',))
        batchifier = Seq2SeqBatchifier(tokenizer, sources, status=('success',), )


        from guesswhat.train.eval_listener import QGenListener
        idx, _ = network.create_greedy_graph(start_token=tokenizer.start_token, stop_token=tokenizer.stop_dialogue, max_tokens=10)
        listener = QGenListener(require=idx)

        best_val_loss = 1e5
        for t in range(0, config['optimizer']['no_epoch']):

            logger.info('Epoch {}..'.format(t + 1))

            # Create cpu pools (at each iteration otherwise threads may become zombie - python bug)
            cpu_pool = create_cpu_pool(args.no_thread, use_process=False)

            train_iterator = Iterator(trainset,
                                      batch_size=batch_size, pool=cpu_pool,
                                      batchifier=batchifier,
                                      shuffle=True)
            [train_loss, _] = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer])

            valid_iterator = Iterator(validset, pool=cpu_pool,
                                      batch_size=batch_size*2,
                                      batchifier=batchifier,
                                      shuffle=False)
            [valid_loss] = evaluator.process(sess, valid_iterator, outputs=outputs, listener=listener)

            question_tokens = listener.results
            for qt in question_tokens:
                logger.info(tokenizer.decode(qt))


            logger.info("Training loss: {}".format(train_loss))
            logger.info("Validation loss: {}".format(valid_loss))

            data["loss"]["train"].append(train_loss)
            data["loss"]["valid"].append(valid_loss)

            if valid_loss < best_val_loss:
                best_train_loss = train_loss
                best_val_loss = valid_loss
                saver.save(sess, save_path.format('params.ckpt'))
                logger.info("QGen checkpoint saved...")

                data["ckpt_epoch"] = t

                pickle_dump(data, save_path.format('status.pkl'))

        # Load early stopping
        saver.restore(sess, save_path.format('params.ckpt'))
        cpu_pool = create_cpu_pool(args.no_thread, use_process=False)
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size*2,
                                 batchifier=batchifier,
                                 shuffle=True)
        [test_loss, _] = evaluator.process(sess, test_iterator, outputs)

        logger.info("Testing loss: {}".format(test_loss))

        data["loss"]["test"] = test_loss
        pickle_dump(data, save_path.format('status.pkl'))
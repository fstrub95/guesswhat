import argparse
import logging
import os
import collections

import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint, create_resnet_saver
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump
from generic.utils.thread_pool import create_cpu_pool
from generic.data_provider.image_loader import get_img_builder
from generic.data_provider.nlp_utils import GloveEmbeddings

from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.questioner_batchifier import LSTMBatchifier
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.models.guesser.guesser_baseline import GuesserNetwork
from guesswhat.train.eval_listener import ProfilerListener


if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('Guesser network baseline!')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-crop_dir", type=str, help='Directory with crops')
    parser.add_argument("-config", type=str, help="Configuration file")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")
    parser.add_argument("-glove_file", type=str, default="glove_dict.pkl", help="Glove file name")
    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-continue_exp", type=bool, default=False, help="Continue previously started experiment?")
    parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")
    parser.add_argument("-no_games_to_load", type=int, help="No games to use during training Default : all")

    args = parser.parse_args()
    config, exp_identifier, save_path = load_config(args.config, args.exp_dir, args)
    logger = logging.getLogger()

    ###############################
    #  LOAD DATA
    #############################

    # Load image and crops
    logger.info('Loading images..')
    image_builder, crop_builder = None, None
    use_resnet = False
    if config["model"]['inputs'].get('image', False):
        logger.info('Loading images..')
        image_builder = get_img_builder(config['model']['image'], args.img_dir)
        use_resnet = image_builder.is_raw_image()

    if config["model"]['inputs'].get('crop', False):
        logger.info('Loading crops..')
        crop_builder = get_img_builder(config['model']['crop'], args.crop_dir, is_crop=True)
        use_resnet = crop_builder.is_raw_image()

    # Load data
    logger.info('Loading data..')
    trainset = Dataset(args.data_dir, "train", image_builder, crop_builder, args.no_games_to_load)
    validset = Dataset(args.data_dir, "valid", image_builder, crop_builder, args.no_games_to_load)
    testset = Dataset(args.data_dir, "test", image_builder, crop_builder, args.no_games_to_load)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(args.dict_file)

    # Load glove
    glove = None
    if config["model"]["question"]['glove']:
        logger.info('Loading glove..')
        glove = GloveEmbeddings(args.glove_file)

    # Build Network
    logger.info('Building network..')

    #network = GuesserNetwork(config['model'], no_words=tokenizer.no_words)

    from guesswhat.models.oracle.oracle_film import FiLM_Oracle
    network = FiLM_Oracle(config['model'], no_words=tokenizer.no_words, no_answers=2)

    # Build Optimizer
    logger.info('Building optimizer..')
    optimizer, outputs = create_optimizer(network, config["optimizer"])

    ###############################
    #  START  TRAINING
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
    data["error"] = collections.defaultdict(list)

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()

    # Retrieve only resnet variabes
    if use_resnet:
        resnet_saver = create_resnet_saver([network])

    # CPU/GPU option
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    builder = tf.profiler.ProfileOptionBuilder
    opts = builder(builder.time_and_memory()).order_by('micros').build()

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

        sources = network.get_sources(sess)
        logger.info("Sources: " + ', '.join(sources))

        sess.run(tf.global_variables_initializer())
        start_epoch = load_checkpoint(sess, saver, args, save_path)

        best_val_err = 0
        best_train_err = None

        # create training tools
        evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
        #batchifier = LSTMBatchifier(tokenizer, sources, glove=glove, status=('success',))

        from guesswhat.train.eval_listener import GuesserAccuracyListener
        accuracy_computation = GuesserAccuracyListener(require=network.softmax)

        from guesswhat.data_provider.guesser_batchifier_oracle_like import OracleGuesserBatchifier
        batchifier = OracleGuesserBatchifier(tokenizer, sources, glove=glove, status=('success',))



        for t in range(start_epoch, no_epoch):

            # Create cpu pools (at each iteration otherwise threads may become zombie - python bug)
            cpu_pool = create_cpu_pool(args.no_thread, use_process=False)

            logger.info('Epoch {}..'.format(t + 1))

            train_iterator = Iterator(trainset,
                                      batch_size=batch_size, pool=cpu_pool,
                                      batchifier=batchifier,
                                      shuffle=True)
            train_loss, train_accuracy = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer], listener=accuracy_computation)
            train_accuracy2 = accuracy_computation.evaluate()

            valid_iterator = Iterator(validset, pool=cpu_pool,
                                      batch_size=batch_size*2,
                                      batchifier=batchifier,
                                      shuffle=False)
            valid_loss, valid_accuracy = evaluator.process(sess, valid_iterator, outputs=outputs, listener=accuracy_computation)
            valid_accuracy2 = accuracy_computation.evaluate()


            logger.info("Training loss: {}".format(train_loss))
            logger.info("Training error: {}".format(1-train_accuracy2))
            logger.info("Training error (Fake): {}".format(1-train_accuracy))
            logger.info("Validation loss: {}".format(valid_loss))
            logger.info("Validation error: {}".format(1-valid_accuracy2))
            logger.info("Training error (Fake): {}".format(1-train_accuracy))

            data["loss"]["train"].append(train_loss)
            data["loss"]["valid"].append(valid_loss)
            data["error"]["train"].append(1-train_accuracy)
            data["error"]["valid"].append(1-valid_accuracy)

            if valid_accuracy > best_val_err:
                best_train_err = train_accuracy
                best_val_err = valid_accuracy
                saver.save(sess, save_path.format('params.ckpt'))
                logger.info("Guesser checkpoint saved...")

                data["ckpt_epoch"] = t

                pickle_dump(data, save_path.format('status.pkl'))

        # Load early stopping
        saver.restore(sess, save_path.format('params.ckpt'))
        cpu_pool = create_cpu_pool(args.no_thread, use_process=False)
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size,
                                 batchifier=batchifier,
                                 shuffle=True)

        [test_loss, test_accuracy] = evaluator.process(sess, test_iterator, outputs, listener=accuracy_computation)
        test_accuracy2 = accuracy_computation.evaluate()

        logger.info("Testing loss: {}".format(test_loss))
        logger.info("Testing error: {}".format(1-test_accuracy2))
        logger.info("Testing error (Fake): {}".format(1-test_accuracy))

        data["loss"]["test"] = test_loss
        data["loss"]["error"] = (1-test_accuracy)
        pickle_dump(data, save_path.format('status.pkl'))
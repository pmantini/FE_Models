import argparse
import json
import inspect

import models

from Hyper_Setup import log_file

import logging

logging.basicConfig(filename=log_file, format='%(filename)s:%(lineno)s %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(log_file)

formatter = logging.Formatter('%(filename)s:%(lineno)s %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
fh.setFormatter(formatter)


def get_classes_in_module(modulename):
    classes = dir(modulename)

    list_model = []
    for c in classes:
        if not c.startswith('__'):
            list_model.append(c)

    return list_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help='Name of the Model',
        default='points')
    parser.add_argument(
        '-t',
        '--task',
        type=str,
        help='task = ["train", "predict"]'
        )
    parser.add_argument(
        '-a',
        '--model-arg',
        type=json.loads,
        help='A dictinary of arguments',
        default={})

    args = parser.parse_args()

    #Get list of models
    list_models = get_classes_in_module(models)

    #Check if the model exists
    if args.model not in list_models:
        logging.error("Model %s is invalied" , args.model)
        logging.error("Available Models: %s, EXITING!", str(list_models))
        exit()

    #pick the model from args
    for name, obj in inspect.getmembers(models):
        if inspect.isclass(obj):
            if name == args.model:
                model = obj


    model_obj = model(args)

    #Get list of arguments
    model_arg_available = model_obj.get_args()

    #Check if requireguments are avaialble
    required_args_check_pass = True
    for arg in model_arg_available["required"]:
        if arg not in args.model_arg.keys():
            logging.error("%s is a required arguement", arg)
            required_args_check_pass = False

    if not required_args_check_pass:
        logging.error("Required arguments check failed, Exiting")
        exit()

    #Warn if optional args do not match
    for argument in args.model_arg.keys():
        if argument not in model_arg_available["required"] and argument not in model_arg_available["optional"]:
            logging.warning("%s is a not a valid optional arguement", argument)




    if args.task == "train_eval":
        logging.info("Begin_training")
        model_obj.do_init(args.model_arg)
        model_obj.do_train_and_eval()
    elif args.task == "eval":
        logging.info("Model Evaluation")
        model_obj.do_init(args.model_arg)
        model_obj.do_eval()
    elif args.task == "pred":
        logging.info("Model Prediction")
        model_obj.do_init(args.model_arg)
        model_obj.do_pred()

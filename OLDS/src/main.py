#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import random
from accelerate import Accelerator
from transformers import set_seed
from args import parse_args
from time import sleep
from others.logging import get_logger
from models.train import train
from models.test import test_all
from models.predict import predict_all
# from models.predict import predict
from others.build_omission_data import build_omission_data
from others.build_omission_data import build_omission_label, group_omission_data
from seq2seq.post_edit import post_edit_train, post_edit_evaluate


def main():
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Initialize the logger
    logger = get_logger(args.log_file, accelerator)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    if args.do_train:
        train(args, logger, accelerator)
    if args.do_test:
        test_all(args, logger, accelerator)
    if args.do_predict:
        predict_all(args, logger, accelerator)
    if args.do_process_train:
        build_omission_data(args, logger, accelerator)
    if args.do_process:
        group_omission_data(args)
        sleep(10)
        build_omission_label(args)
    if args.do_edit_train:
        post_edit_train(args, logger, accelerator)
    if args.do_edit_test:
        post_edit_evaluate(args, logger, accelerator)

if __name__ == "__main__":
    main()

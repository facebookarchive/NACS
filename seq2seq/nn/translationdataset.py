# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torchtext import data, datasets
import os


class FactoredTranslationDataset(data.Dataset):
    """Translation Dataset"""

    urls = []
    name = ''
    dirname = ''

    @staticmethod
    def sort_key(x):
        return lambda x: len(x.src)

    def __init__(self, path, exts, fields, max_length=0, **kwargs):
        """Create a FactoredTranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data in each language.
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            raise ValueError("You should provide fields as a list of ('name', field) tuples")

        # src_path = os.path.expanduser(path + exts[0])
        # trg_path = os.path.expanduser(path + exts[1])
        #
        # # find src_tags and trg_tags fields in the fields list
        # if len(fields) > 2 and fields[2][0] == 'src_tags':
        #     src_tags_path = os.path.expanduser(path + exts[2])
        #     src_tags_field = fields[2][1]
        # else:
        #     src_tags_path = None
        #     src_tags_field = None
        #
        # if len(fields) > 2 and fields[2][0] == 'trg_tags':
        #     trg_tags_path = os.path.expanduser(path + exts[2])
        #     trg_tags_field = fields[2][1]
        # elif len(fields) == 4 and fields[3][0] == 'trg_tags':
        #     trg_tags_path = os.path.expanduser(path + exts[3])
        #     trg_tags_field = fields[3][1]
        # else:
        #     trg_tags_path = None
        #     trg_tags_field = None

        examples = []

        paths = [os.path.expanduser(path + x) for x in exts]
        files = [open(p, mode='r', encoding='utf-8') for p in paths]

        while True:
            lines = [f.readline() for f in files]

            if lines[0] != '' and lines[1] != '':
                lines = [line.strip() for line in lines]

                # if len(lines[0]) == 0 or len(lines[1]) == 0:
                #     continue

                example = data.Example.fromlist(lines, fields)

                # if max_length > 0:
                #     example.src = example.src[:max_length]
                #     example.trg = example.trg[:max_length]

                examples.append(example)
            else:
                break

        for f in files:
            f.close()

        # if factored_input:
        #     with open(src_path) as src_file, open(trg_path) as trg_file, open(src_tags_path) as src_factor_file:
        #         for src_line, trg_line, src_factor_line in zip(src_file, trg_file, src_factor_file):
        #             src_line, trg_line, src_factor_line = src_line.strip(), trg_line.strip(), src_factor_line.strip()
        #             if src_line != '' and trg_line != '':
        #                 examples.append(data.Example.fromlist([src_line, trg_line, src_factor_line], fields))

        # else:
        # with open(src_path) as src_file, open(trg_path) as trg_file:
        #
        #     src_tags_file = open(src_tags_path) if src_tags_path is not None else None
        #     trg_tags_file = open(trg_tags_path) if trg_tags_path is not None else None
        #
        #     for src_line, trg_line in zip(src_file, trg_file):
        #
        #         src_tags_line = src_tags_file.readline().strip() if src_tags_file is not None else None
        #         trg_tags_line = trg_tags_file.readline().strip() if trg_tags_file is not None else None
        #
        #         src_line, trg_line = src_line.strip(), trg_line.strip()
        #
        #         if src_line != '' and trg_line != '':
        #
        #             [src_line, trg_line]
        #
        #             examples.append(data.Example.fromlist(lines, fields))

        super(FactoredTranslationDataset, self).__init__(examples, fields, **kwargs)

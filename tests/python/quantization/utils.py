import mxnet as mx

import logging

class FilterList(logging.Filter):
    def __init__(self, default=False, allows=[], disables=[]):
        self.rules = {}
        self._internal_filter_rule = "_internal_filter_rule"

        self.rules[self._internal_filter_rule] = default
        for name in allows:
            splits = name.split(".")
            rules = self.rules
            for split in splits:
                if split not in rules:
                    rules[split] = {}
                rules = rules[split]

            rules[self._internal_filter_rule] = True

        for name in disables:
            splits = name.split(".")
            rules = self.rules
            for split in splits:
                if split not in rules:
                    rules[split] = {}
                rules = rules[split]

            rules[self._internal_filter_rule] = False

    def filter(self, record):
        splits = record.name.split(".")
        rules = self.rules

        rv = rules[self._internal_filter_rule]
        for split in splits:
            if split not in rules:
                return rv
            else:
                rules = rules[split]
                if self._internal_filter_rule in rules:
                    rv = rules[self._internal_filter_rule]

        return rv

def load_parameters(graph, params, prefix=None, ctx=None):
    params_dict = graph.collect_params()
    params_dict.initialize(ctx=ctx)
    for name in params_dict:
        split_name, uniq_name = name.split("_"), []
        [uniq_name.append(sname) for sname in split_name if sname not in uniq_name]
        param_name = "_".join(uniq_name)
        param_name = param_name[len(prefix):] if prefix else param_name
        params_dict[name].set_data(params[param_name])

def load_dataset():
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}
    std_args = {'std_r': rgb_std[0], 'std_g': rgb_std[1], 'std_b': rgb_std[2]}

    return mx.io.ImageRecordIter(path_imgrec="./data/val_256_q90.rec",
                                label_width=1,
                                preprocess_threads=60,
                                batch_size=10,
                                data_shape=(3, 224, 224),
                                label_name="softmax_label",
                                rand_crop=False,
                                rand_mirror=False,
                                shuffle=True,
                                shuffle_chunk_seed=3982304,
                                seed=48564309,
                                **mean_args,
                                **std_args)


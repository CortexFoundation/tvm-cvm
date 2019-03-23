import mxnet as mx

import logging

class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super(ColoredFormatter, self).__init__(fmt, datefmt, style)

        self.log_colors = {
            "DEBUG": "\033[38;5;111m",
            "INFO": "\033[38;5;47m",
            "WARNING": "\033[38;5;178m",
            "ERROR": "\033[38;5;196m",
            "CRITICAL": "\033[30;48;5;196m",
            "DEFAULT": "\033[38;5;15m",
            "RESET": "\033[0m"
        }

    def format(self, record):
        log_color = self.get_color(record.levelname)
        message = super(ColoredFormatter, self).format(record)
        message = log_color + message + self.log_colors["RESET"]
        return message

    def get_color(self, level_name):
        lname = level_name if level_name in self.log_colors else "DEFAULT"
        return self.log_colors[lname]


class FilterList(logging.Filter):
    """ Filter with logging module

        Filter rules as below:
            {allow|disable log name} > level no > keywords >
            {inheritance from parent log name} > by default filter
        TODO:
    """
    def __init__(self, default=False, allows=[], disables=[],
            keywords=[], log_level=logging.INFO):
        self.rules = {}
        self._internal_filter_rule = "_internal_filter_rule"
        self.log_level = log_level
        self.keywords = keywords

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
        rules = self.rules
        rv = rules[self._internal_filter_rule]

        splits = record.name.split(".")
        for split in splits:
            if split in rules:
                rules = rules[split]
                if self._internal_filter_rule in rules:
                    rv = rules[self._internal_filter_rule]
            else:
                if record.levelno >= self.log_level:
                    return True

                for keyword in self.keywords:
                    if keyword in record.getMessage():
                        return True

                return rv

        return rv

def load_parameters(graph, params, prefix=None, ctx=None):
    params_dict = graph.collect_params()
    params_dict.initialize(ctx=ctx)
    for name in params_dict:
        split_name, uniq_name = name.split("_"), []
        [uniq_name.append(sname) for sname in split_name if sname not in uniq_name]
        param_name = "_".join(uniq_name)
        param_name = param_name[len(prefix):] if prefix else param_name
        assert param_name in params or name in params, \
            "param name(%s) with origin(%s) not exits"%(param_name, name)
        data = params[name] if name in params else params[param_name]
        params_dict[name].set_data(data)

    return params_dict

def load_dataset(batch_size=10):
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}
    std_args = {'std_r': rgb_std[0], 'std_g': rgb_std[1], 'std_b': rgb_std[2]}

    return mx.io.ImageRecordIter(path_imgrec="./data/val_256_q90.rec",
                                label_width=1,
                                preprocess_threads=60,
                                batch_size=batch_size,
                                data_shape=(3, 224, 224),
                                label_name="softmax_label",
                                rand_crop=False,
                                rand_mirror=False,
                                shuffle=True,
                                shuffle_chunk_seed=3982304,
                                seed=48564309,
                                **mean_args,
                                **std_args)


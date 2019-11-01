from mxnet import ndarray as nd
import math
import numpy as np

from sym_utils import *

def _get_opt(out, lambd):
    absmax = out.abs().max().asscalar()
    if lambd is None:
        return absmax
    mean = nd.mean(out).asscalar()
    std = nd.norm(out - mean).asscalar() / math.sqrt(np.product(out.shape))
    alpha = abs(mean) + lambd * std
    if alpha < 0.95 * absmax:
        print ("[", mean, std, "]", alpha, absmax)
        return alpha
    return absmax

def sym_calibrate(symbol, **kwargs):
    logger = logging.getLogger('log.mrt.calibrate')
    _, deps = topo_sort(symbol, logger=logger, with_deps=True)
    old_ths = self.th_dict if self.th_dict else {}
    th_dict, out_cache, data = {}, {}, kwargs['data']
    def _impl(op, params, **kwargs):
        name, op_name = op.attr('name'), op.attr('op_name')
        deps, logger = kwargs['deps'], kwargs['logger']
        if op_name == 'null':
            out = kwargs['data'][name] if is_inputs(name, params) \
                  else params[name]
        elif childs is None:
            out = get_nd_op(op_name)(**attr)
        else:
            cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
            nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
            out = get_nd_op(op_name)(*nd_inputs, **attr)
            for n, _ in cinfos:
                assert n in deps
                deps[n].remove(name)
                if len(deps[n]) == 0:
                    del out_cache[n]
        out = [out] if len(sym) == 1 else out
        out_cache[name] = [o.as_in_context(ctx) for o in out]
        opts = float(_get_opt(out[0], lambd))
        if name in old_ths:
            th_dict[name] = max(old_ths[name], opts)
        else:
            th_dict[name] = opts
            p = logger.debug if opts < 30 else logger.warn
            p("collect symbol %-40s out_shape=%-20s th_dict: (%s)",
                    name, [o.shape for o in out], th_dict[name])

    topo_visit_transformer(symbol, params, _impl, data=data, deps=deps)
    out_cache.clear()
    return th_dict


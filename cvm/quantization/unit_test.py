import unittest

import mxnet as mx
import nnvm

import transformer as tfm
import sym_utils as sutils

def graph_equal(src, des):
    if src.attr('op_name') != des.attr('op_name'):
        return src, des
    if sutils.get_entry_id(src) != sutils.get_entry_id(des):
        return src, des
    if src.list_attr() != des.list_attr():
        return src, des
    src_childs = sutils.sym_iter(src.get_children())
    des_childs = sutils.sym_iter(des.get_children())
    if src_childs is None:
        if des_childs is None:
            return None, None
        else:
            return src, des
    if len(src_childs) != len(des_childs):
        return src, des
    for i, op in enumerate(src_childs):
        r1, r2 = graph_equal(op, des_childs[i])
        if r1 is not None:
            return r1, r2
    return None, None

def summary(sym, err_op=None):
    _s = ""
    for op in sutils.topo_sort(sym):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sutils.sym_iter(op.get_children()), op.list_attr()
        prefix = "Op " if op_name == "null" else "Var"
        if (err_op is not None) and (name == err_op.attr('name')):
            prefix = "> " + prefix
        _s += "%5s:%-10s, Name=%-15s, Attr=%-40s" \
               % (prefix, op_name, name, attr)
        if childs is not None:
            cinfos = ["%s(%d)" % (c.attr('name'), sutils.get_entry_id(c)) \
                         for c in childs]
            _s += ", Inputs=%s" % ", ".join(cinfos)
        _s += "\n"
    return _s


class TfmTest(unittest.TestCase):
    def _collect_params(self, symbol):
        params = {}
        for op in sutils.topo_sort(symbol):
            if sutils.is_var(op, params):
                _, shp, _ = op.infer_shape()
                params[op.attr('name')] = mx.nd.uniform(-1, 1, shp[0])
        return params

    def _assert_equal(self, op, des, passes=[]):
        if isinstance(passes, str):
            passes = [passes]
        params = self._collect_params(op)
        for p in passes:
            op, params = getattr(tfm, p)(op, params)

        r1, r2 = graph_equal(op, des)
        _s ="Graph Not Equal\n" + "-" * 20 +"\n"
        _s += summary(op, r1) + "-" * 20 + "\n"
        _s += summary(des, r2)
        self.assertIsNone(r1, _s)

    def _assert_error(self, op, passes):
        if isinstance(passes, str):
            passes = [passes]
        params = self._collect_params(op)
        with self.assertRaises(AssertionError):
            for p in passes:
                op, params = getattr(tfm, p)(op, params)

class TestNull(TfmTest):
    op = mx.sym.var("data", shape=[1, 3, 2, 1])

    def test_validate(self):
        self._assert_equal(self.op, self.op, "validate")

    def test_rewrite(self):
        self._assert_equal(self.op, self.op, "rewrite")

    def test_fuse_transpose(self):
        self._assert_equal(self.op, self.op, "fuse_transpose")

class TestTranspose(TfmTest):
    op = mx.sym.transpose(
            mx.sym.transpose(TestNull.op, axes=[0, 2, 3, 1]),
            axes=[3, 2, 1, 0])

    def test_validate(self):
        self._assert_equal(self.op, self.op, "validate")

        op1 = mx.sym.transpose(TestNull.op, axes=[])
        des = mx.sym.transpose(TestNull.op, axes=[3, 2, 1, 0])
        self._assert_equal(op1, des, "validate")

        op2 = mx.sym.transpose(TestNull.op)
        self._assert_equal(op2, des, "validate")

    def test_rewrite(self):
        self._assert_equal(self.op, self.op, "rewrite")

    def test_fuse_transpose(self):
        des = mx.sym.transpose(TestNull.op, axes=[1, 3, 2, 0])
        self._assert_equal(self.op, des, "fuse_transpose")


class TestElemwiseAdd(TfmTest):
    def test_fuse_transpose(self):
        x = mx.sym.var("x", shape=(1,2,3,4))
        op1 = mx.sym.transpose(x, axes=[1, 3, 2, 0])
        op2 = mx.sym.transpose(x, axes=[1, 3, 2, 0])
        op = mx.symbol.elemwise_add(op1, op2)

        des = mx.sym.transpose(
                mx.sym.elemwise_add(x, x),
                axes=[1, 3, 2, 0])

        self._assert_equal(op, des, "fuse_transpose")


class TestConcat(TfmTest):
    def test_fuse_transpose(self):
        x = mx.sym.var('x', shape=(1, 3, 244, 244))
        op1 = mx.sym.transpose(x, axes=[3, 2, 1, 0])
        op2 = mx.sym.transpose(x, axes=[])
        op = mx.sym.Concat(op1, op2, dim=1)

        des = mx.sym.transpose(mx.sym.concat(x, x, dim=2), axes=[3,2,1,0])

        self._assert_equal(op, des,
                ["validate", "fuse_transpose"])


class TestFuseMultiplyInputs(TfmTest):
    def test_fmi(self):
        d1 = mx.sym.var('d1', shape=(1, 2, 3))
        d2 = mx.sym.var('d2', shape=(4, 2))
        d3 = mx.sym.var('d3', shape=(2, 3))
        op = mx.sym.Group([d1, d2, d3])

        data = mx.sym.var('data', shape=(20,))
        s1 = mx.sym.slice(data, begin=(0,), end=(6,))
        r1 = mx.sym.reshape(s1, shape=(1, 2, 3))
        s2 = mx.sym.slice(data, begin=(6,), end=(14,))
        r2 = mx.sym.reshape(s2, shape=(4, 2))
        s3 = mx.sym.slice(data, begin=(14,), end=(20,))
        r3 = mx.sym.reshape(s3, shape=(2, 3))
        des = mx.sym.Group([r1, r2, r3])

        self._assert_equal(op, des)



import sys
if __name__ == "__main__":
    unittest.main(argv=sys.argv, verbosity=5)

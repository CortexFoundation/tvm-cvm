import unittest

import mxnet as mx
import nnvm

import transformer as tfm
import sym_utils as sutils

class TfmTest(unittest.TestCase):
    def _collect_params(self, symbol):
        params = {}
        for op in sutils.topo_sort(symbol):
            if sutils.is_var(op, params):
                _, shp, _ = op.infer_shape()
                params[op.attr('name')] = mx.nd.uniform(-1, 1, shp[0])
        return params

    def _assert_equal(self, src, tgt):
        if src is None:
            self.assertIsNone(tgt)
            return
        elif isinstance(src, list):
            self.assertTrue(isinstance(tgt, list))
            self.assertEqual(len(src), len(tgt))
            for i in range(len(src)):
                self._assert_equal(src[i], tgt[i])
            return
        op_name = src.attr('op_name')
        self.assertEqual(op_name, tgt.attr('op_name'))
        src_eid, tgt_eid = sutils.get_entry_id(src), sutils.get_entry_id(tgt)
        self.assertEqual(src_eid, tgt_eid)
        src_attr, tgt_attr = src.list_attr(), tgt.list_attr()
        self.assertEqual(src_attr, tgt_attr)
        self._assert_equal(sutils.sym_iter(src.get_children()),
                           sutils.sym_iter(tgt.get_children()))

    def _assert(self, op, pass_t, tgt):
        params = self._collect_params(op)
        src, _ = getattr(tfm, pass_t)(op, params)
        self._assert_equal(src, tgt)

    def _assert_error(self, op, pass_t):
        params = self._collect_params(op)
        with self.assertRaises(AssertionError):
            getattr(tfm, pass_t)(op, params)

class TestNull(TfmTest):
    op = mx.sym.var("data", shape=[1, 3, 2, 1])

    def test_validate(self):
        self._assert(self.op, "validate", self.op)

    def test_rewrite(self):
        self._assert(self.op, "rewrite", self.op)

    def test_fuse_transpose(self):
        self._assert(self.op, "fuse_transpose", self.op)

    # def test_compile(self):
        # self._assert(self.op, "compile",
                # nnvm.sym.Variable("tgt", shape=[1, 3, 2, 1]))

class TestTranspose(TfmTest):
    op = mx.sym.transpose(
            mx.sym.transpose(TestNull.op, axes=[0, 2, 3, 1]),
            axes=[3, 2, 1, 0])

    def test_validate(self):
        self._assert(self.op, "validate", self.op)

        op1 = mx.sym.transpose(TestNull.op, axes=[])
        tgt = mx.sym.transpose(TestNull.op, axes=[3, 2, 1, 0])
        self._assert(op1, "validate", tgt)

        op2 = mx.sym.transpose(TestNull.op)
        self._assert(op2, "validate", tgt)

    def test_rewrite(self):
        self._assert(self.op, "rewrite", self.op)

    #def test_fuse_transpose(self):
        #tgt = mx.sym.transpose(TestNull.op, axes=[1, 3, 0, 2])
        #self._assert(self.op, "fuse_transpose", tgt)


class TestElemwiseAdd(TfmTest):
    def test_fuse_transpose(self):
        x = mx.sym.var("x", shape=(1,2,3,4))
        op1 = mx.sym.transpose(x, axes=[1, 3, 2, 0])
        op2 = mx.sym.transpose(x, axes=[1, 3, 2, 0])
        op = mx.symbol.elemwise_add(op1, op2)

        tgt = mx.sym.transpose(
                mx.sym.elemwise_add(x, x),
                axes=[1, 3, 2, 0])
        
        self._assert(op, "fuse_transpose", tgt)


        pass


class TestConcat(TfmTest):
    def test_fuse_transpose(self):
        x = mx.sym.var('x', shape=(1, 3, 244, 244))
        op1 = mx.sym.transpose(x, axes=[3, 2, 1, 0])
        op2 = mx.sym.transpose(x, axes=[3, 2, 1, 0])
        op = mx.sym.Concat(op1, op2, dim=1)

        tgt = mx.sym.transpose(mx.sym.concat(x, x, dim=2), axes=[3,2,1,0])

        self._assert(op, "fuse_transpose", tgt)

        pass

if __name__ == "__main__":
    unittest.main(verbosity=5)

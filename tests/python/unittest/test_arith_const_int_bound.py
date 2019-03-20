import tvm

def test_dtype_bound():
    analyzer = tvm.arith.Analyzer()

    x = tvm.var("x", dtype="int64")
    bd = analyzer.const_int_bound(x)
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == bd.POS_INF

    x = tvm.var("x", dtype="int8")
    bd = analyzer.const_int_bound(x)
    assert bd.min_value == -128
    assert bd.max_value == 127

    x = tvm.var("x", dtype="uint8")
    bd = analyzer.const_int_bound(x)
    assert bd.min_value == 0
    assert bd.max_value == 255


def test_cast_bound():
    analyzer = tvm.arith.Analyzer()
    x = tvm.var("x", dtype="int8")
    bd = analyzer.const_int_bound((x % 3).astype("uint32"))
    assert bd.min_value == 0
    assert bd.max_value == 2

    bd = analyzer.const_int_bound(
        (x % 3).astype("float32").astype("int32"))
    assert bd.min_value == -2
    assert bd.max_value == 2


def test_add_sub_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x", "int64"), tvm.var("y", "int64")
    bd = analyzer.const_int_bound(x + y)
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == bd.POS_INF

    analyzer.update(x, tvm.arith.ConstIntBound(0, 4))
    analyzer.update(y, tvm.arith.ConstIntBound(1, 10))
    bd = analyzer.const_int_bound(x + y)
    assert bd.min_value == 1
    assert bd.max_value == 14

    bd = analyzer.const_int_bound(x - y)
    assert bd.min_value == -10
    assert bd.max_value == 3

    analyzer.update(x, tvm.arith.ConstIntBound(0, bd.POS_INF), override=True)
    bd = analyzer.const_int_bound(x - y)
    assert bd.min_value == -10
    assert bd.max_value == bd.POS_INF

    bd = analyzer.const_int_bound(1 - x)
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == 1


def test_mul_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x"), tvm.var("y")

    analyzer.update(x, tvm.arith.ConstIntBound(-2, 4))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))
    bd = analyzer.const_int_bound(x * y + 20)
    assert bd.min_value == 0
    assert bd.max_value == 60

    analyzer.update(x, tvm.arith.ConstIntBound(-3, 4), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(-8, 2), override=True)
    bd = analyzer.const_int_bound(x * y)
    assert bd.min_value == -32
    assert bd.max_value == 24

    analyzer.update(x, tvm.arith.ConstIntBound(bd.NEG_INF, 4), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(-8, 2), override=True)
    bd = analyzer.const_int_bound(x * y)
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == bd.POS_INF


def test_div_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x"), tvm.var("y")

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 4))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))
    bd = analyzer.const_int_bound(x / y)
    assert bd.min_value == -2

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 4), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(-2, 0), override=True)
    bd = analyzer.const_int_bound(x / y)
    assert bd.min_value == -4
    assert bd.max_value == 9

    analyzer.update(x, tvm.arith.ConstIntBound(bd.NEG_INF, 4), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(-2, 1), override=True)
    bd = analyzer.const_int_bound(x / y)
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == bd.POS_INF


def test_mod_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x"), tvm.var("y")

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 4))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))
    bd = analyzer.const_int_bound(x % y)
    assert bd.min_value == -9
    assert bd.max_value == 4

    analyzer.update(x, tvm.arith.ConstIntBound(bd.NEG_INF, bd.POS_INF), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10), override=True)
    bd = analyzer.const_int_bound(x % y)
    assert bd.min_value == -9
    assert bd.max_value == 9

    analyzer.update(x, tvm.arith.ConstIntBound(1, bd.POS_INF), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10), override=True)
    bd = analyzer.const_int_bound(x % y)
    assert bd.min_value == 0
    assert bd.max_value == 9


def test_min_max_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x"), tvm.var("y")

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 11))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))
    bd = analyzer.const_int_bound(tvm.min(x, y))
    assert bd.min_value == -9
    assert bd.max_value == 10

    analyzer.update(x, tvm.arith.ConstIntBound(bd.NEG_INF, bd.POS_INF), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10), override=True)
    bd = analyzer.const_int_bound(tvm.min(x, y))
    assert bd.min_value == bd.NEG_INF
    assert bd.max_value == 10

    bd = analyzer.const_int_bound(tvm.max(x, y))
    assert bd.min_value == 4
    assert bd.max_value == bd.POS_INF

    analyzer.update(x, tvm.arith.ConstIntBound(1, bd.POS_INF), override=True)
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10), override=True)
    bd = analyzer.const_int_bound(tvm.max(x, y))
    assert bd.min_value == 4
    assert bd.max_value == bd.POS_INF


def test_select_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x"), tvm.var("y")

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 11))
    analyzer.update(y, tvm.arith.ConstIntBound(4, 10))

    bd = analyzer.const_int_bound(
        tvm.expr.Select(x > 1, (y < 0).astype("int32"), y + 1))
    assert bd.min_value == 0
    assert bd.max_value == 11


def test_shift_and_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x"), tvm.var("y")

    analyzer.update(x, tvm.arith.ConstIntBound(-9, 11))
    analyzer.update(y, tvm.arith.ConstIntBound(2, 10))

    bd = analyzer.const_int_bound(x >> y)
    assert bd.min_value == -3
    assert bd.max_value == 2

    bd = analyzer.const_int_bound(x & y)
    assert bd.min_value == 0
    assert bd.max_value == 10

    analyzer.update(x, tvm.arith.ConstIntBound(10, 11), override=True)
    bd = analyzer.const_int_bound(x & y)
    assert bd.min_value == 0
    assert bd.max_value == 10


def test_mix_index_bound():
    analyzer = tvm.arith.Analyzer()
    x, y = tvm.var("x"), tvm.var("y")
    analyzer.update(x, tvm.arith.ConstIntBound(0, 24 - 1))
    analyzer.update(y, tvm.arith.ConstIntBound(0, 3 - 1))
    bd = analyzer.const_int_bound((x % 8) + (x / 8) * 8)
    assert bd.min_value == 0
    assert bd.max_value == 24 - 1

    bd = analyzer.const_int_bound(y + x * 3)
    assert bd.min_value == 0
    assert bd.max_value == 24 * 3 - 1

    bd = analyzer.const_int_bound((x % 7) + (x / 7) * 7)
    assert bd.min_value == 0
    assert bd.max_value == (23 // 7) * 7 + 6


if __name__ == "__main__":
    test_dtype_bound()
    test_cast_bound()
    test_add_sub_bound()
    test_mul_bound()
    test_div_bound()
    test_mod_bound()
    test_min_max_bound()
    test_select_bound()
    test_shift_and_bound()
    test_mix_index_bound()

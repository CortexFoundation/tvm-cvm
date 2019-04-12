"""
Get Started with NNVM
=====================
**Author**: `Tianqi Chen <https://tqchen.github.io/>`_

This article is an introductory tutorial to workflow in NNVM.
"""
import nnvm.compiler
import nnvm.symbol as sym

######################################################################
# Declare Computation
# -------------------
# We start by describing our need using computational graph.
# Most deep learning frameworks use computation graph to describe
# their computation. In this example, we directly use
# NNVM's API to construct the computational graph.
#
# .. note::
#
#   In a typical deep learning compilation workflow,
#   we can get the models from :any:`nnvm.frontend`
#
# The following code snippet describes :math:`z = x + \sqrt{y}`
# and creates a nnvm graph from the description.
# We can print out the graph ir to check the graph content.

x = sym.Variable("data")
y = sym.dense(x, units=16, use_bias=False)
z = sym.clip(y, a_max = 100, a_min = 0)
z = sym.dense(z, units=10, use_bias=False)
z = sym.clip(z, a_max = 100, a_min = 0)
compute_graph = nnvm.graph.create(z)

shape = (1, 28)
with nnvm.compiler.build_config(opt_level=0):
    deploy_graph, lib, params = nnvm.compiler.build(
            compute_graph, target="cuda", shape={"data": shape}, dtype="int32")

with open('/tmp/start_cuda.json', "w") as fout:
    fout.write(deploy_graph.json())
exit()


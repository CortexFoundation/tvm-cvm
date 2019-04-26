"""Util to invoke clang in the system."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import subprocess

from .._ffi.base import py_str
from .. import codegen
from . import util


def find_clang(required=True):
    """Find clang in system.

    Parameters
    ----------
    required : bool
        Whether it is required,
        runtime error will be raised if the compiler is required.

    Returns
    -------
    valid_list : list of str
        List of possible paths.

    Note
    ----
    This function will first search clang that
    matches the major llvm version that built with tvm
    """
    cc_list = []
    if hasattr(codegen, "llvm_version_major"):
        cc_list += ["clang-%d.0" % codegen.llvm_version_major()]
    cc_list += ["clang"]
    cc_list += ["clang.exe"]
    valid_list = [util.which(x) for x in cc_list]
    valid_list = [x for x in valid_list if x]
    if not valid_list and required:
        raise RuntimeError(
            "cannot find clang, candidates are: " + str(cc_list))
    return valid_list


def create_llvm(inputs,
                output=None,
                options=None,
                cc=None):
    """Create llvm text ir.

    Parameters
    ----------
    inputs : list of str
        List of input files name or code source.

    output : str, optional
        Output file, if it is none
        a temporary file is created

    options : list
        The list of additional options string.

    cc : str, optional
        The clang compiler, if not specified,
        we will try to guess the matched clang version.

    Returns
    -------
    code : str
        The generated llvm text IR.
    """
    cc = cc if cc else find_clang()[0]
    cmd = [cc]
    cmd += ["-S", "-emit-llvm"]
    temp = util.tempdir()
    output = output if output else temp.relpath("output.ll")
    inputs = [inputs] if isinstance(inputs, str) else inputs
    input_files = []
    for i, code in enumerate(inputs):
        if util.is_source_path(code):
            input_files.append(code)
        else:
            temp_path = temp.relpath("input%d.cc" % i)
            with open(temp_path, "w") as output_file:
                output_file.write(code)
            input_files.append(temp_path)
    if options:
        cmd += options
    cmd += ["-o", output]
    cmd += input_files
    print(cmd)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    return open(output).read()

import io
from typing import List

import pytest
import sh


def run_command(command: List[str]):
    """Default method for executing shell commands with pytest."""
    msg = None
    try:
        _stdout = io.StringIO()
        sh.python(command, _out=_stdout)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
        str_stdout = _stdout.getvalue()
        print(str_stdout)
    if msg:
        pytest.fail(msg=msg)

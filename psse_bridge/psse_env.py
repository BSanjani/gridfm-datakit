# psse_bridge/psse_env.py

import sys
import os


def init_psse(psse_version=35):
    """
    Initialize PSS/E Python environment.
    Call ONCE before any psspy usage.
    """

    if psse_version == 35:
        PSSE_ROOT = r"C:\Program Files\PTI\PSSE35\35.2"
        PYTHON_PATH = os.path.join(PSSE_ROOT, "PSSPY39")
        BIN_PATH = os.path.join(PSSE_ROOT, "PSSBIN")
    else:
        raise RuntimeError("Unsupported PSS/E version")

    if PYTHON_PATH not in sys.path:
        sys.path.append(PYTHON_PATH)
    if BIN_PATH not in sys.path:
        sys.path.append(BIN_PATH)

    import redirect
    redirect.psse2py()

    import psspy
    psspy.psseinit(10000)

    return psspy

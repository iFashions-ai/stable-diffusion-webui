import sys
from modules.cmd_args import get_argv

argv = get_argv() or sys.argv
# this will break any attempt to import xformers which will prevent stability diffusion repo from trying to use it
if "--xformers" not in "".join(argv):
    sys.modules["xformers"] = None

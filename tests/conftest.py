import os
import tempfile

# spacepy bootstraps a config file at ~/.spacepy/spacepy.rc the first time it's
# imported. Its bootstrap code has a race condition: if multiple processes hit
# a missing/stale rc file at the same time, spacepy can raise
# `UnboundLocalError: cannot access local variable 'nextsec'` while trying to
# write it (see spacepy/__init__.py:_write_defaults). Under `pytest-xdist`,
# every worker imports spacepy independently at roughly the same time, so they
# race over that single shared file.
#
# Give each xdist worker its own spacepy config directory so there is no
# shared file to race over. Serial runs (no xdist, or `python -m unittest`)
# leave SPACEPY unset and get spacepy's normal default location.
_worker_id = os.environ.get("PYTEST_XDIST_WORKER")
if _worker_id is not None and "SPACEPY" not in os.environ:
    os.environ["SPACEPY"] = os.path.join(tempfile.gettempdir(), f"spacepy-{_worker_id}")

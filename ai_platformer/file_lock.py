"""Nonblocking exclusive locks released when the owning stream closes."""

import os


def lock_exclusive(stream):
    """Lock an open binary file; callers keep it open for the protected work."""
    if os.name == "nt":
        import msvcrt

        # Windows byte-range locks may cover bytes beyond EOF. Avoid writing
        # before acquiring the lock: another process may already own byte zero.
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

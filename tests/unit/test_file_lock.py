"""Exercise OS locks across processes, including release after close."""
import subprocess
import sys

from ai_platformer.file_lock import lock_exclusive


def test_competing_process_is_rejected_then_can_acquire(tmp_path):
    path = tmp_path / "output.lock"
    code = """
import sys
from ai_platformer.file_lock import lock_exclusive
with open(sys.argv[1], 'a+b') as stream:
    try:
        lock_exclusive(stream)
    except OSError:
        sys.exit(7)
"""
    def contender():
        return subprocess.run([sys.executable, "-c", code, str(path)],
                              capture_output=True, text=True, timeout=15)

    with path.open("a+b") as stream:
        lock_exclusive(stream)
        result = contender()
        assert result.returncode == 7, result.stderr
    result = contender()
    assert result.returncode == 0, result.stderr

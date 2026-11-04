"""One cancellable background check per play session; no worker UI mutations."""

from concurrent.futures import Future, CancelledError
from threading import Event, Thread
from types import SimpleNamespace

from ai_platformer.content.play_session import PlayAdmissionError


class SuggestionJob:
    def __init__(self, snapshot):
        self.future = Future()
        self.cancelled = Event()
        self.thread = Thread(target=self._run, args=(snapshot,), daemon=True,
                             name="player-candidate-check")
        self.thread.start()

    def _run(self, snapshot):
        try:
            # Optional generation dependencies also load off the UI thread.
            from .candidates import prepare_offer
            result = prepare_offer(snapshot, cancelled=self.cancelled.is_set)
            if self.cancelled.is_set():
                raise CancelledError()
            self.future.set_result(result)
        except Exception as error:
            self.future.set_exception(error)

    def cancel(self):
        self.cancelled.set()


def start_suggestion(session):
    previous = getattr(session, "suggestion_job", None)
    if previous is not None and not previous.future.done():
        raise PlayAdmissionError("上次检查正在结束，请稍后再试。")
    try:
        source = session.records.freeze_profile_source(session.prepared) if session.records else None
    except (OSError, ValueError, RuntimeError) as error:
        raise PlayAdmissionError("游玩记录暂不可用，仍可重试原关。") from error
    # This snapshot holds no live session, recorder, store or Pygame object.
    snapshot = SimpleNamespace(prepared=session.prepared, records=source)
    try:
        job = SuggestionJob(snapshot)
    except RuntimeError as error:
        raise PlayAdmissionError("后台检查无法启动，仍可重试原关。") from error
    session.suggestion_job = job
    return job

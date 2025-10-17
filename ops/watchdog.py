import multiprocessing as mp
import traceback


class TimeoutError(Exception):
    pass


def run_with_timeout(fn, kwargs=None, timeout_s=120, name="job"):
    kwargs = kwargs or {}
    q = mp.Queue()

    def _target(q, kwargs):
        try:
            res = fn(**kwargs)
            q.put(("ok", res))
        except Exception as e:
            q.put(("err", f"{e}\n{traceback.format_exc()}"))

    p = mp.Process(target=_target, args=(q, kwargs), name=name, daemon=True)
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(2)
        raise TimeoutError(f"{name} exceeded {timeout_s}s and was killed")
    if q.empty():
        raise RuntimeError(f"{name} died without result")
    status, payload = q.get()
    if status == "err":
        raise RuntimeError(payload)
    return payload

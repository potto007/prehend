"""repair_doubled_calls: collapse a doubled empty-call '()()' (the .lower()() decode
stutter -> TypeError 'str' object is not callable) down to '()' BEFORE exec, so the
known corruption runs as the intended code instead of erroring. Opt-in (default off)."""

from prehend.environments.local_repl import LocalREPL


def test_repair_collapses_stutter_no_error():
    repl = LocalREPL(repair_doubled_calls=True)
    r = repl.execute_code("y = 'AB'.lower()()")   # would TypeError without repair
    assert not (r.stderr or "").strip(), f"unexpected stderr: {r.stderr!r}"
    r2 = repl.execute_code("print(y)")
    assert "ab" in r2.stdout


def test_default_does_not_repair():
    repl = LocalREPL()  # default off -> legacy behavior
    r = repl.execute_code("'AB'.lower()()")
    assert "not callable" in (r.stderr or "")


def test_single_call_untouched():
    repl = LocalREPL(repair_doubled_calls=True)
    r = repl.execute_code("z = 'AB'.lower()")  # a single () must be left alone
    assert not (r.stderr or "").strip()
    r2 = repl.execute_code("print(z)")
    assert "ab" in r2.stdout

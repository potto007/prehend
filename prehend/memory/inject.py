"""prehend injection: render retrieved experiences as a ``<Memory_Block>``.

The block is meant to be prepended to the orchestrator's ``root_prompt`` (the
text the top-level model attends to directly), NOT buried in the offloaded REPL
context variable, so guidance is actually read before the model writes programs.

Each entry carries a ``polarity``: ``positive`` is a guiding-path recipe to
follow, ``negative`` is a guard-rule to apply when its condition fires.
"""
from __future__ import annotations

_XML_ESCAPE = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})


def _escape(text: str) -> str:
    return str(text).translate(_XML_ESCAPE)


def render_memory_block(entries: list[dict]) -> str:
    """Render retrieved experiences as a ``<Memory_Block>`` string.

    Returns ``""`` when there are no entries, so the no-memory path stays
    byte-identical to running without the memory layer.
    """
    if not entries:
        return ""

    lines = [
        "<Memory_Block>",
        "  <Guidance>Lessons distilled from similar past problems. "
        "Follow positive insights when they apply; apply negative guard rules "
        "when their condition fires. Ignore any entry that does not fit.</Guidance>",
    ]
    for entry in entries:
        polarity = _escape(entry.get("polarity", "positive"))
        lines.append(f'  <Entry polarity="{polarity}">')
        insight = entry.get("key_insight", "")
        if insight:
            lines.append(f"    <Insight>{_escape(insight)}</Insight>")
        findings = entry.get("findings") or []
        if findings:
            lines.append("    <Findings>")
            for f in findings:
                lines.append(f"      - {_escape(f)}")
            lines.append("    </Findings>")
        cautions = entry.get("cautions") or []
        if cautions:
            lines.append("    <Cautions>")
            for c in cautions:
                lines.append(f"      - {_escape(c)}")
            lines.append("    </Cautions>")
        lines.append("  </Entry>")
    lines.append("  <OptOut>If an entry above does not actually fit this "
                 "problem, ignore it.</OptOut>")
    lines.append("</Memory_Block>")
    return "\n".join(lines)

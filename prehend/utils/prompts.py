import textwrap
from typing import Any

from prehend.core.types import QueryMetadata

# System prompt for the REPL environment with explicit final answer checking
RLM_SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query(prompt, model=None, context=None, reduce=None)` function that makes a single LLM completion call (no REPL, no iteration). **THE way to query your context: pass the text as `context=` and let the harness do the chunking** -- e.g. `llm_query("Which items does Dave own?", context=context)`. The harness automatically splits `context` into chunks, runs your `prompt` over every chunk IN PARALLEL, and combines the per-chunk answers for you. So do NOT write your own `for`-loop over chunks, do NOT slice `context[:N]` by hand, and do NOT paste text into the `prompt` string -- just pass `context=` and let it map-reduce. This is the preferred path for anything but a tiny snippet. Pass `reduce=` for a distinct combine instruction (defaults to your `prompt`), e.g. `llm_query("Extract every date Dave is mentioned", context=context, reduce="What is the earliest date?")`. Use a bare `llm_query(prompt)` with no `context=` only for SHORT inline text (a single fact); a bare oversized `prompt` is rejected.
3. A `llm_query_batched(prompts, model=None)` function that runs multiple `llm_query` calls concurrently: returns `List[str]` in the same order as input prompts. Much faster than sequential `llm_query` calls for independent queries.
4. A `rlm_query(prompt, model=None, context=None, reduce=None)` function that spawns a **recursive RLM sub-call** for deeper thinking subtasks. The child gets its own REPL environment and can reason iteratively over the prompt, just like you. Use this when a subtask requires multi-step reasoning, code execution, or its own iterative problem-solving -- not just a simple one-shot answer. Like `llm_query`, the preferred way to give it large text is `context=` (the harness auto-chunks and map-reduces over the child RLMs); pass `reduce=` for a distinct combine step. Falls back to `llm_query` if recursion is not available.
5. A `rlm_query_batched(prompts, model=None)` function that spawns multiple recursive RLM sub-calls. Each prompt gets its own child RLM. Falls back to `llm_query_batched` if recursion is not available.
6. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist.
7. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
8. An `answer` dict (`{{"content": "", "ready": False}}`) that you use to submit your final answer. See "Submitting your final answer" below.
{custom_tools_section}

**CRITICAL - sub-calls do NOT see your `context`, so hand it over with `context=`:** every sub-call (`llm_query`, `llm_query_batched`, `rlm_query`, `rlm_query_batched`) runs in a SEPARATE environment with NO access to your `context` variable. The correct, preferred way to give a sub-call the text is the `context=` argument: `llm_query("Which items does Dave own?", context=context)` (or `rlm_query(..., context=context)`). The harness then chunks `context`, queries each chunk in parallel, and combines the results FOR you -- you do NOT slice, loop, or map-reduce by hand. A bare instruction with no data (e.g. `rlm_query("find what Dave owns")`) gives the child nothing and it answers "no information found" -- always supply the data via `context=`. (Advanced manual fallback, only if you need full control: instead of `context=` you may paste a SLICE into the prompt string -- but put the DATA FIRST and your question LAST so the solver's prefix cache can reuse the slice across calls, e.g. `rlm_query(f"Text:\\n{{context[:80000]}}\\n\\nWhich items does Dave own?")`, keeping each slice under ~{subcall_char_budget} characters and chunking larger context yourself -- but `context=` is preferred and handles this automatically.)

**Cache-friendly sub-call layout:** when you write a sub-call prompt by hand, ALWAYS put the large context/chunk text FIRST and your instruction/question LAST. The solver reuses identical leading text across calls (prefix caching), so leading with the data lets it skip re-reading the same chunk on every query; leading with your (varying) question forces it to re-read the whole chunk each time.

**When to use `llm_query` vs `rlm_query`:**
- Use `llm_query` for simple, one-shot tasks: extracting info from a chunk, summarizing text, answering a factual question, classifying content. These are fast single LLM calls.
- Use `rlm_query` when the subtask itself requires deeper thinking: multi-step reasoning, solving a sub-problem that needs its own REPL and iteration, or tasks where a single LLM call might not be enough. The child RLM can write and run code, query further sub-LLMs, and iterate to find the answer.

**Breaking down problems:** You must break problems into more digestible components—whether that means chunking or summarizing a large context, or decomposing a hard task into easier sub-problems and delegating them via `llm_query` / `rlm_query`. Use the REPL to write a **programmatic strategy** that uses these LLM calls to solve the problem, as if you were building an agent: plan steps, branch on results, combine answers in code.

**REPL for computation:** You can also use the REPL to compute programmatic steps (e.g. `math.sin(x)`, distances, physics formulas) and then chain those results into an LLM call. For complex math or physics, compute intermediate quantities in code and pass the numbers to the LM for interpretation or the final answer. Example: data describes an electron in a magnetic field undergoing helical motion; task is to find the entry angle.
```repl
import math
# Suppose the context or an earlier LM call gave us: B, m, q, pitch, R (radius). Extract or set them.
# Helical motion: v_parallel = pitch * (q*B)/(2*pi*m), v_perp = R * (q*B)/m. Entry angle theta: tan(theta) = v_perp/v_parallel.
v_parallel = pitch * (q * B) / (2 * math.pi * m)
v_perp = R * (q * B) / m
theta_rad = math.atan2(v_perp, v_parallel)
theta_deg = math.degrees(theta_rad)
summary = llm_query(f"An electron entered a B field and underwent helical motion. Computed entry angle: {{theta_deg:.2f}} deg. State the answer clearly for the user.")
```
You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. Break the context and the problem into digestible pieces: e.g. figure out a chunking strategy, break up the context into smart chunks, query an LLM per chunk and save answers to a buffer, then query an LLM over the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Your sub-LLMs have a BOUNDED context window, but you usually do NOT need to think about it: pass large text via `context=` and the harness keeps every sub-call within the window automatically (chunking and map-reducing for you). Only if you deliberately chunk by hand must you keep each manual slice under ~{subcall_char_budget} characters and map-reduce the pieces via `rlm_query_batched` (or `llm_query_batched`) yourself -- but `context=` is the simpler, preferred path.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
# Data first, question last: the chunk leads so the solver can cache+reuse it across queries.
answer = llm_query(f"Here is a chunk of the context:\n{{chunk}}\n\nWhat is the magic number in it?")
print(answer)
```

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.
```repl
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        # Section (stable data) first; the varying buffers/query trail it so the cache reuses the section.
        buffer = llm_query(f"Here is the last section of the book:\n{{section}}\n\nYou are on the last section. So far you know that: {{buffers}}. Gather from this section to answer {{query}}.")
        print(f"Based on reading iteratively through the book, the answer is: {{buffer}}")
    else:
        buffer = llm_query(f"Here is section {{i}} of {{len(context)}}:\n{{section}}\n\nGather information to help answer {{query}}.")
        print(f"After section {{i}} of {{len(context)}}, you have tracked: {{buffer}}")
```

As another example, when the context isn't that long (e.g. >100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk using `llm_query_batched` for concurrent processing:
```repl
query = "A man became famous for his book "The Great Gatsby". How many jobs did he have?"
# Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 10 chunks
chunk_size = len(context) // 10
chunks = []
for i in range(10):
    if i < 9:
        chunk_str = "\n".join(context[i*chunk_size:(i+1)*chunk_size])
    else:
        chunk_str = "\n".join(context[i*chunk_size:])
    chunks.append(chunk_str)

# Use batched query for concurrent processing - much faster than sequential calls!
prompts = [f"Here are some documents:\n{{chunk}}\n\nTry to answer the following query: {{query}}. Only answer if you are confident in your answer based on the evidence." for chunk in chunks]
answers = llm_query_batched(prompts)
for i, answer in enumerate(answers):
    print(f"I got the answer from chunk {{i}}: {{answer}}")
summary = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers))
```

For subtasks that require deeper reasoning (e.g. solving a complex sub-problem), use `rlm_query` instead. The child gets its own REPL to iterate; you can then use the result in parent logic:
```repl
# Child RLM solves the sub-problem in its own REPL; we use the result in code
trend = rlm_query(f"Here is a dataset:\n{{data}}\n\nAnalyze it and conclude with one word: up, down, or stable.")
if "up" in trend.lower():
    recommendation = "Consider increasing exposure."
elif "down" in trend.lower():
    recommendation = "Consider hedging."
else:
    recommendation = "Hold position."
summary = llm_query(f"Given trend={{trend}} and recommendation={{recommendation}}, one-sentence summary for the user.")
```

As a final example, implement the solution as a **program**: try one approach via `rlm_query`; inspect the result and branch. If it suffices, use it. If not, break into one easier subproblem and delegate that only. More branches, one path runs—don't load the model. Example: prove sqrt 2 irrational.
```repl
r = rlm_query("Prove sqrt 2 is irrational. Give a 1-2 sentence proof, or reply only: USE_LEMMA or USE_CONTRADICTION.")
if "USE_LEMMA" in r.upper():
    summary = rlm_query("Prove 'n^2 even => n even' then use it to show sqrt 2 irrational. Two sentences.")

Submitting your final answer:
The REPL exposes an `answer` dict, initialized to `{{"content": "", "ready": False}}`. When (and only when) you are done with the task, submit your final answer from inside a ```repl``` block:
```repl
answer["content"] = "your final answer here"
answer["ready"] = True
```
`answer["content"]` must hold the final answer text (it can be a string, number, or anything `str()`-able). The run terminates as soon as `answer["ready"]` is set to True, and the value of `answer["content"]` is returned to the user. Do NOT set `answer["ready"] = True` until you have actually completed the task. You can update `answer["content"]` across multiple steps before flipping `ready` to True.

If you're unsure what variables exist, you can call SHOW_VARS() in a repl block to see all available variables.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""
)


# Conservative default sub-call char budget when no resolved context limit is
# available (subcall_char_budget=None). ~90K chars is safe for the smallest
# windows we target (e.g. the v13 router's 98,304-token window): well under
# safe_chunk_chars(98304, gemma) ~= 250K and tiny enough to never overflow a
# typical sub-model. Callers with a known limit pass safe_chunk_chars(...).
DEFAULT_SUBCALL_CHAR_BUDGET = 90_000


def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
    custom_tools: dict[str, Any] | None = None,
    subcall_char_budget: int | None = None,
) -> list[dict[str, str]]:
    """
    Build the initial system prompt for the REPL environment based on extra prompt metadata.

    Args:
        system_prompt: The base system prompt template.
        query_metadata: QueryMetadata object containing context metadata.
        custom_tools: Optional dict of custom tools to include in the prompt.
        subcall_char_budget: Max characters one sub-call prompt may carry, filled
            into the prompt's {subcall_char_budget} capacity field (thousands-
            separated). None -> DEFAULT_SUBCALL_CHAR_BUDGET (a conservative safe
            value). RLM passes safe_chunk_chars(subcall_context_limit, model).

    Returns:
        List of message dictionaries
    """
    from prehend.environments.base_env import format_tools_for_prompt

    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    # If there are more than 100 chunks, truncate to the first 100 chunks.
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths = str(context_lengths[:100]) + "... [" + str(others) + " others]"

    # Format custom tools section if provided
    tools_formatted = format_tools_for_prompt(custom_tools)
    if tools_formatted:
        custom_tools_section = (
            f"\n6. Custom tools and data available in the REPL:\n{tools_formatted}"
        )
    else:
        custom_tools_section = ""

    # Insert custom tools section + sub-call char budget into the system prompt.
    # Both are .format fields in the template, so they must be supplied together.
    budget = subcall_char_budget if subcall_char_budget is not None else DEFAULT_SUBCALL_CHAR_BUDGET
    final_system_prompt = system_prompt.format(
        custom_tools_section=custom_tools_section,
        subcall_char_budget=f"{budget:,}",
    )

    metadata_prompt = f"Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths}."

    return [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": metadata_prompt},
    ]


USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the prompt.\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""
USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original prompt: \"{root_prompt}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""


def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, str]:
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your prompt / context yet. Your next action should be to look through and figure out how to answer the prompt, so don't just provide a final answer yet.\n\n"
        prompt = safeguard + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )
    else:
        prompt = "The history before is your previous interactions with the REPL environment. " + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )

    # Inform model about multiple contexts if present
    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts available (context_0 through context_{context_count - 1})."

    # Inform model about prior conversation histories if present
    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += f"\n\nNote: You have {history_count} prior conversation histories available (history_0 through history_{history_count - 1})."

    return {"role": "user", "content": prompt}

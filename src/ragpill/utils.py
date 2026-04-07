import re
import unicodedata
from collections.abc import Sequence
from functools import reduce
from typing import Any

from httpx import AsyncClient
from openai import AsyncOpenAI
from pydantic_ai import models
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Dataset
from pydantic_settings import BaseSettings

from ragpill.base import BaseEvaluator, CaseMetadataT


def _prefix_settings_key(input: tuple[BaseSettings | dict[str, Any], str]) -> dict[str, Any]:
    setting, prefix = input
    if isinstance(setting, BaseSettings):
        setting_dict = setting.model_dump()
    else:
        setting_dict = setting
    return {f"{prefix}_{k}": v for k, v in setting_dict.items()}


def _clean_quote_text(text: str, depth: int = 0, quote_char: str | None = None) -> tuple[str, str | None]:
    """
    Recursively clean quote text by detecting quote characters and ensuring proper nesting and alternation.
    Returns cleaned text and the detected quote char of the outermost level if any.
    If quote_char is provided, this function ensures that this quote_char is used for this level
    and replaces any other quote chars with the provided one, while alternating quote chars for nested levels.
    """
    QUOTES = ('"', "'")

    def alternate_quote(q: str) -> str:
        return "'" if q == '"' else '"'

    def find_matching_quote(s: str, start: int, q: str) -> int:
        """Find the index of the matching closing quote, or -1 if not found."""
        i = start + 1
        while i < len(s):
            if s[i] == q:
                return i
            i += 1
        return -1

    # Strip whitespace first
    text = text.strip()

    # Check if entire text is wrapped in matching quotes - if so, strip them
    if len(text) >= 2 and text[0] in QUOTES and text[-1] == text[0]:
        # Check if these are truly the outer quotes (not just coincidental)
        outer_quote = text[0]
        match_idx = find_matching_quote(text, 0, outer_quote)
        if match_idx == len(text) - 1:
            # Strip outer quotes and recurse
            inner_text, inner_quote_char = _clean_quote_text(text[1:-1], depth + 1, quote_char)
            return inner_text.strip(), inner_quote_char

    # Check for unmatched leading quote
    if len(text) >= 1 and text[0] in QUOTES:
        leading_quote = text[0]
        match_idx = find_matching_quote(text, 0, leading_quote)
        if match_idx == -1:
            # Unmatched leading quote - strip it
            inner_text, _ = _clean_quote_text(text[1:], depth + 1, quote_char or leading_quote)
            return inner_text.strip(), quote_char or leading_quote

    # Detect the first quote character in the text
    detected_quote = None
    for c in text:
        if c in QUOTES:
            detected_quote = c
            break

    # If quote_char is provided, normalize all quotes in the text
    if quote_char is not None:
        alt_quote = alternate_quote(quote_char)
        result_chars: list[str] = []
        i = 0
        while i < len(text):
            if text[i] in QUOTES:
                current_quote = text[i]
                match_idx = find_matching_quote(text, i, current_quote)
                if match_idx != -1:
                    # Found a quoted section - recurse with alternating quote
                    inner_content = text[i + 1 : match_idx]
                    cleaned_inner, _ = _clean_quote_text(inner_content, depth + 1, alt_quote)
                    result_chars.append(quote_char)
                    result_chars.append(cleaned_inner)
                    result_chars.append(quote_char)
                    i = match_idx + 1
                else:
                    result_chars.append(text[i])
                    i += 1
            else:
                result_chars.append(text[i])
                i += 1
        return "".join(result_chars), quote_char

    # No quote_char provided - detect and normalize to first found quote type
    if detected_quote is not None:
        alt_quote = alternate_quote(detected_quote)
        result: list[str] = []
        i = 0
        while i < len(text):
            if text[i] in QUOTES:
                current_quote = text[i]
                match_idx = find_matching_quote(text, i, current_quote)
                if match_idx != -1:
                    # Found a quoted section - normalize and recurse
                    inner_content = text[i + 1 : match_idx]
                    cleaned_inner, _ = _clean_quote_text(inner_content, depth + 1, alt_quote)
                    result.append(detected_quote)
                    result.append(cleaned_inner)
                    result.append(detected_quote)
                    i = match_idx + 1
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        return "".join(result), detected_quote

    # No quotes in text
    return text, None


def _get_source(line: str) -> str | None:
    """Extract source reference from a line if it exists."""
    file_match = re.search(r"(?i)\(?(?:file|source):\s*\[([^\]]+)\]\)?", line)
    if file_match:
        return file_match.group(1)
    return None


def _extract_quotes(lines: list[str], depth: int = 0) -> list[tuple[list[str], str | None, int, int]]:
    """Extract blocks of quoted text from lines (as lines), their source, the first line number and the last line number of that quote.

    Note: for nested quotes the last_line_num-first_line_num+1 is generally different from len(lines) because
    quotes get collapsed.
    """

    i = 0
    all_quotes: list[tuple[list[str], str | None, int, int]] = []
    quote_chars = ['"', "'"]
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^(\s*)>(.*)$", line)
        if not match:
            i += 1
            continue

        current_indent = len(match.group(1))
        quote_lines: list[str] = []
        # empty_run = 0

        # Collect contiguous quote lines; break when indentation level changes or line isn't a quote
        first_line_num = last_line_num = i
        while i < len(lines):
            inner_match = re.match(r"^(\s*)>(.*)$", lines[i])
            if not inner_match:
                break
            last_line_num = i
            indent = len(inner_match.group(1))
            if indent != current_indent:
                break
            content = inner_match.group(2)
            if content.strip():
                quote_lines.append(
                    content.strip()
                )  # the strip is a bit lenient if subquotes would have had different indents.

            # Empty quote line
            # empty_run += 1
            i += 1
            # if empty_run >= 2 and quote_lines:
            #     break
        if not quote_lines:
            continue
        src = _get_source(lines[i] if i < len(lines) else "")
        if not src:
            src = _get_source(lines[i - 1])
            if src:
                quote_lines.pop()  # remove the source line from the quote block if it's part of it

        # get subquotes and replace those lines with the text in quotation-marks.
        # quotationmark should be quote_char[depth % len(quote_char)] if quote_char is not None else None

        subquotes = _extract_quotes(quote_lines, depth=depth + 1)
        for subquote_lines, subquote_source, subquote_first_line_num, subquote_last_line_num in subquotes:
            # replace the subquote lines in quote_lines with a single line containing the subquote text in quotation marks

            src_str = f" (source: {subquote_source})" if subquote_source else ""
            subquote_text = " ".join(subquote_lines).strip()
            subquote_text, inner_quote_char = _clean_quote_text(subquote_text)
            remaining_quote_chars = [qc for qc in quote_chars if qc != inner_quote_char]
            quote_char = remaining_quote_chars[depth % len(remaining_quote_chars)]
            quote_lines = (
                [line for i, line in enumerate(quote_lines) if i < subquote_first_line_num]
                + [(f"{quote_char}{subquote_text}{quote_char}" + src_str)]
                + [line for i, line in enumerate(quote_lines) if i > subquote_last_line_num]
            )
        for quote_char in quote_chars:
            if (
                quote_lines
                and quote_lines[0].lstrip().startswith(quote_char)
                and quote_lines[-1].rstrip().endswith(quote_char)
            ):
                # Strip the quote character from the first and last line of the quote block
                quote_lines[0] = quote_lines[0].lstrip()[1:]
                quote_lines[-1] = quote_lines[-1].rstrip()[:-1]
                break

        all_quotes.append((quote_lines, src, first_line_num, last_line_num))

    return all_quotes


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (whitespace + Unicode NFKC).

    Useful for aligning visually identical text such as ``UF₆`` vs ``UF6``.
    """
    # Strip single-tilde markdown subscripts (e.g., UF~6~ -> UF6) with a tight, short payload to avoid altering other tilde uses
    text = re.sub(r"~([0-9A-Za-z]{1,10})~", r"\1", text)
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Normalize all quote-like characters to straight single quote:
    # - Straight double quote: " (U+0022)
    # - Curly double quotes: " " (U+201C, U+201D)
    # - Curly single quotes/apostrophes: ' ' (U+2018, U+2019)
    # - Low quotes: „ ‚ (U+201E, U+201A)  # noqa: RUF003
    # - Guillemets: « » ‹ › (U+00AB, U+00BB, U+2039, U+203A)  # noqa: RUF003
    # - Prime symbols: ′ ″ (U+2032, U+2033)  # noqa: RUF003
    # - Grave/acute accents: ` ´ (U+0060, U+00B4)  # noqa: RUF003
    normalized = re.sub(
        r'["\u201C\u201D\u201E\'\u2018\u2019\u201A\u00AB\u00BB\u2039\u203A\u2032\u2033`\u00B4]', "'", normalized
    )
    # Strip trailing punctuation (periods) for comparison
    normalized = normalized.strip(".")
    return f"{normalized}"


def _extract_markdown_quotes(output: str) -> list[tuple[str, str | None]]:  # pyright: ignore[reportUnusedFunction]
    """Extract and normalizes markdown quotes and their file references from output.

    Only lines that start with '>' (after leading whitespace) are considered markdown quotes.
    Regular quoted text is ignored. Quotation marks are stripped from the extracted quotes.
    Quote text is normalized with ``unicodedata.normalize("NFKC")`` and ``casefold``
    so visually similar bytecode/Unicode sequences compare equal (e.g., ``UF₆`` → ``uf6``).

    Args:
        output: The text to extract quotes from

    Returns:
        List of tuples: (quote_text_without_quotation_marks, referenced_filename_or_none)
    """

    quotes: list[tuple[str, str | None]] = []
    lines = output.split("\n")

    found_quotes = _extract_quotes(lines)
    for quote_lines, source, _, _ in found_quotes:
        quote_text = " ".join(quote_lines).strip() if quote_lines else ""
        # clean quote_text:
        # whitespace normalization (collapse multiple spaces/newlines into single space, trim) + lowercase + Unicode NFKC normalization:
        quote_text = _normalize_text(quote_text)
        quote_text = re.sub(r"\.{2,}", ".*", quote_text)  # normalize ellipsis to be flexible for matching
        quotes.append((quote_text, source))
    return quotes


def merge_settings(settings_prefixes: Sequence[tuple[BaseSettings | dict[str, Any], str]]) -> dict[str, Any]:
    """
    Used to merge muliple pydantic settings into a single dict with prefixed keys to log in mlflow.

    :param settings_prefixes: Description
    :type settings_prefixes: Sequence[tuple[BaseSettings|dict, str]]
    :return: Description
    :rtype: dict[str, Any]

    Example:
    merge_settings((settings.mlflow_settings, "mlflow"),(settings.agent_settings, "agent"), (settings.llm_settings, "llm"), (settings.embedding_settings, "embedding"), (settings.retrieval_settings, "retrieval"))
    """
    return reduce(lambda x, y: x | y, map(_prefix_settings_key, settings_prefixes))


def _fix_evaluator_global_flag(dataset: Dataset[Any, Any, CaseMetadataT]) -> None:  # pyright: ignore[reportUnusedFunction]
    """Ensure that global evaluator metadata is marked correctly."""
    # for case in dataset.cases:
    #     for evaluator in case.evaluators:
    #         if isinstance(evaluator, Evaluator) and hasattr(evaluator, "metadata") and evaluator.metadata is not None:
    #             if evaluator.metadata.is_global_evaluator is None:
    #                 evaluator.metadata.is_global_evaluator = False
    for evaluator in dataset.evaluators:
        if isinstance(evaluator, BaseEvaluator):
            evaluator.is_global = True


def get_pydantic_ai_llm_model(base_url: str, api_key: str, model_name: str, temperature: float = 0.0) -> models.Model:
    """Get a pydantic-ai LLM model based on provided settings."""
    http_client = AsyncClient()
    openai_client = AsyncOpenAI(
        max_retries=3,
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
    )
    pyai_llm_model = OpenAIChatModel(
        model_name, provider=OpenAIProvider(openai_client=openai_client), settings={"temperature": temperature}
    )
    return pyai_llm_model

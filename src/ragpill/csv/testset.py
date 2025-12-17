"""Create Testset from CSV file - Refactored modular version."""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from ragpill.base import BaseEvaluator, TestCaseMetadata
from ragpill.eval_types import Case, Dataset
from ragpill.evaluators import (
    HasQuotesEvaluator,
    LiteralQuoteEvaluator,
    LLMJudge,
    RegexInDocumentMetadataEvaluator,
    RegexInOutputEvaluator,
    RegexInSourcesEvaluator,
)


def _read_csv_with_encoding(csv_path: str | Path) -> list[dict[str, str]]:
    """Read CSV file with automatic encoding detection.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of row dictionaries
    """
    encodings = ["utf-8-sig", "latin-1", "cp1252", "utf-8"]

    for encoding in encodings:
        try:
            with open(csv_path, encoding=encoding) as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            if encoding == encodings[-1]:
                raise RuntimeError(f"Could not read CSV file with any supported encoding: {csv_path}") from e
            continue

    raise RuntimeError(f"Could not read CSV file with any supported encoding: {csv_path}")


def _group_rows_by_question(rows: list[dict[str, str]], question_column: str) -> dict[str, list[dict[str, str]]]:
    """Group CSV rows by question.

    Args:
        rows: List of CSV row dictionaries
        question_column: Name of the column containing questions

    Returns:
        Dictionary mapping questions to their rows
    """
    question_to_rows: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        question = row[question_column]
        question_to_rows[question].append(row)
    return dict(question_to_rows)


def _parse_row_data(
    row: dict[str, str],
    standard_columns: set[str],
    expected_column: str,
    tags_column: str,
    check_column: str,
) -> tuple[bool, str, str, set[str], dict[str, Any]]:
    """Parse data from a CSV row.

    Args:
        row: CSV row dictionary
        standard_columns: Set of column names that shouldn't be treated as attributes
        expected_column: Name of expected column
        tags_column: Name of tags column
        check_column: Name of check column

    Returns:
        Tuple of (expected, tags_str, check, row_tags_set, additional_attrs)
    """
    expected = row[expected_column].strip().upper() in ("TRUE", "1", "YES")

    tags = row.get(tags_column, "")
    check = row.get(check_column, "")

    # Collect tags for this row
    row_tags: set[str] = set()
    if tags:
        row_tags = {tag.strip() for tag in tags.split(",") if tag.strip()}

    # Collect additional attributes (all columns not in standard set)
    additional_attrs = {k: v for k, v in row.items() if k not in standard_columns and v}

    return expected, tags, check, row_tags, additional_attrs


def _find_common_tags_and_attributes(
    all_row_tags: list[set[str]], all_row_attributes: list[dict[str, Any]]
) -> tuple[set[str], dict[str, Any]]:
    """Find tags and attributes common to all rows.

    Args:
        all_row_tags: List of tag sets from each row
        all_row_attributes: List of attribute dicts from each row

    Returns:
        Tuple of (common_tags, common_attributes)
    """
    # Find common tags (present in ALL rows)
    case_tags: set[str] = all_row_tags[0].intersection(*all_row_tags[1:]) if all_row_tags else set()

    # Find common attributes (same key-value in ALL rows)
    case_attributes: dict[str, Any] = {}
    if all_row_attributes:
        # Start with first row's attributes
        potential_common = all_row_attributes[0].copy()
        # Check if each key-value pair exists in all other rows
        for attr_dict in all_row_attributes[1:]:
            keys_to_remove: list[str] = []
            for key, value in potential_common.items():
                if key not in attr_dict or attr_dict[key] != value:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del potential_common[key]
        case_attributes = potential_common

    return case_tags, case_attributes


def _remove_common_metadata_from_evaluators(
    evaluators: list[BaseEvaluator], case_tags: set[str], case_attributes: dict[str, Any]
) -> None:
    """Remove case-level tags and attributes from evaluators (in-place).

    Args:
        evaluators: List of evaluators to modify
        case_tags: Tags to remove from evaluators
        case_attributes: Attributes to remove from evaluators
    """
    for evaluator in evaluators:
        # Remove case-level tags from evaluator tags
        evaluator.tags = evaluator.tags - case_tags

        # Remove case-level attributes from evaluator attributes
        for key in case_attributes:
            evaluator.attributes.pop(key, None)


def create_evaluator_from_row(
    row: dict[str, str],
    evaluator_class: type[BaseEvaluator],
    standard_columns: set[str],
    expected_column: str,
    tags_column: str,
    check_column: str,
) -> tuple[BaseEvaluator, set[str], dict[str, Any]]:
    """Create an evaluator from a CSV row.

    Args:
        row: CSV row dictionary
        evaluator_class: Evaluator class with from_csv_line() class method
        standard_columns: Columns that shouldn't be treated as attributes
        expected_column: Name of expected column
        tags_column: Name of tags column
        check_column: Name of check column

    Returns:
        Tuple of (evaluator, row_tags, additional_attrs)
    """
    expected, _, check, row_tags, additional_attrs = _parse_row_data(
        row, standard_columns, expected_column, tags_column, check_column
    )

    # Create evaluator using from_csv_line class method
    # All dependencies (model, settings, etc.) are expected to be injected
    # or provided via the check column as JSON
    evaluator = evaluator_class.from_csv_line(expected=expected, tags=row_tags, check=check, **additional_attrs)

    return evaluator, row_tags, additional_attrs


def _create_case_from_rows(
    question: str,
    rows: list[dict[str, str]],
    evaluator_classes: dict[str, type[BaseEvaluator]],
    standard_columns: set[str],
    test_type_column: str,
    expected_column: str,
    tags_column: str,
    check_column: str,
    skip_unknown_evaluators: bool,
) -> Case | None:
    """Create a Case from multiple CSV rows with the same question.

    Args:
        question: The question/input for the case
        rows: List of CSV rows for this question
        evaluator_classes: Dict mapping test_type to evaluator classes
        standard_columns: Columns that shouldn't be treated as attributes
        test_type_column: Name of test_type column
        expected_column: Name of expected column
        tags_column: Name of tags column
        check_column: Name of check column
        skip_unknown_evaluators: Whether to skip unknown evaluator types

    Returns:
        Case object, or None if all evaluators were skipped
    """
    evaluators: list[BaseEvaluator] = []
    all_row_tags: list[set[str]] = []
    all_row_attributes: list[dict[str, Any]] = []

    for row in rows:
        test_type = row[test_type_column]
        evaluator_class = evaluator_classes.get(test_type)

        if evaluator_class is None:
            if skip_unknown_evaluators:
                continue
            else:
                raise ValueError(
                    f"Unknown evaluator type: {test_type}. Available types: {list(evaluator_classes.keys())}"
                )

        evaluator, row_tags, additional_attrs = create_evaluator_from_row(
            row, evaluator_class, standard_columns, expected_column, tags_column, check_column
        )

        evaluators.append(evaluator)
        all_row_tags.append(row_tags)
        all_row_attributes.append(additional_attrs)

    # If all evaluators were skipped, return None
    if not evaluators:
        return None

    # Find common tags and attributes
    case_tags, case_attributes = _find_common_tags_and_attributes(all_row_tags, all_row_attributes)

    # Remove common metadata from evaluators
    _remove_common_metadata_from_evaluators(evaluators, case_tags, case_attributes)

    # Extract repeat/threshold (must be consistent across rows for same question)
    # Use None when absent — resolve_repeat() will apply global defaults from MLFlowSettings
    repeat_values: set[int | None] = set()
    threshold_values: set[float | None] = set()
    for row in rows:
        raw_repeat = row.get("repeat", "").strip()
        raw_threshold = row.get("threshold", "").strip()
        repeat_values.add(int(raw_repeat) if raw_repeat else None)
        threshold_values.add(float(raw_threshold) if raw_threshold else None)

    if len(repeat_values) > 1:
        raise ValueError(f"Inconsistent 'repeat' values for question '{question}': {repeat_values}")
    if len(threshold_values) > 1:
        raise ValueError(f"Inconsistent 'threshold' values for question '{question}': {threshold_values}")

    repeat = repeat_values.pop() if repeat_values else None
    threshold = threshold_values.pop() if threshold_values else None

    # Create Case with metadata
    return Case(
        inputs=question,
        evaluators=evaluators,
        metadata=TestCaseMetadata(attributes=case_attributes, tags=case_tags, repeat=repeat, threshold=threshold),
    )


# Default evaluator classes can be extended as more evaluators are implemented
default_evaluator_classes: dict[str, type[BaseEvaluator]] = {
    "LLMJudge": LLMJudge,
    "RegexInSourcesEvaluator": RegexInSourcesEvaluator,
    "RegexInDocumentMetadata": RegexInDocumentMetadataEvaluator,
    "LiteralQuoteEvaluator": LiteralQuoteEvaluator,
    "HasQuotesEvaluator": HasQuotesEvaluator,
    "RegexInOutputEvaluator": RegexInOutputEvaluator,
    # Add other default evaluator classes here as they are implemented:
    # 'REGEX': RegexEvaluator,
    # 'CountSentences': CountSentencesEvaluator,
}


def load_testset(
    csv_path: str | Path,
    evaluator_classes: dict[str, type[BaseEvaluator]] = default_evaluator_classes,
    skip_unknown_evaluators: bool = False,
    question_column: str = "Question",
    test_type_column: str = "test_type",
    expected_column: str = "expected",
    tags_column: str = "tags",
    check_column: str = "check",
) -> Dataset[str, str, TestCaseMetadata]:
    """Create a Dataset from a CSV file with evaluator configurations.

    Each evaluator class must implement a from_csv_line() class method that accepts:
    - Standard CSV columns: expected, tags, check
    - Additional CSV columns as **kwargs (passed to evaluator.attributes)

    CSV Format:
        The CSV file should contain the following standard columns:

        - **Question**: The input question/prompt for the test case
        - **test_type**: Name of the evaluator class (must match key in evaluator_classes dict)
        - **expected**, **tags**, **check**: Standard evaluator parameters

        For detailed descriptions of these parameters, see
        [`ragpill.base.BaseEvaluator.from_csv_line`][ragpill.base.BaseEvaluator.from_csv_line].

        Any additional columns (e.g., priority, category, domain) will be:
        1. Passed to each evaluator's attributes dict via **kwargs in from_csv_line()
        2. If all evaluators for a question have the same value for an attribute,
           that attribute becomes part of the Test Case metadata and will be visible in MLflow

    Global Evaluators:
        Rows with empty questions are treated as global evaluators and will be added to ALL test cases:

        ```csv
        Question,test_type,expected,tags,check
        ,LLMJudge,true,global,"response is polite"
        What is X?,RegexEvaluator,true,factual,"X.*definition"
        ```

        The LLMJudge evaluator will be added to all cases, including the "What is X?" case.

    Custom Attributes:
        You can add custom columns to track metadata:

        ```csv
        Question,test_type,expected,tags,check,priority,category
        What is X?,LLMJudge,true,factual,"contains the fact, that x is ...",high,science
        What is Y's email?,RegexEvaluator,true,"auth,contacts","y@example.com",low,validation
        ```

        These custom attributes (priority, category) are automatically:
        - Available in evaluator.attributes
        - Promoted to Case metadata if all evaluators share the same value
        - Visible in MLflow tracking for analysis and filtering

    Args:
        csv_path: Path to the CSV file
        evaluator_classes: Dictionary mapping test_type names to evaluator classes.
                          Extend default_evaluator_classes with custom evaluators:
                          `default_evaluator_classes | {'MyEval': MyEvaluator}`
        skip_unknown_evaluators: If True, skip rows with unknown evaluator types instead of raising an error
        question_column: Name of the column containing questions (default: 'Question')
        test_type_column: Name of the column containing evaluator class names (default: 'test_type')
        expected_column: Name of the column for expected flag (default: 'expected')
        tags_column: Name of the column for comma-separated tags (default: 'tags')
        check_column: Name of the column for evaluator-specific check data (default: 'check')

    Returns:
        Dataset with Cases grouped by question, each Case having multiple evaluators

    Example:
        ```python
        from ragpill.csv.testset import load_testset, default_evaluator_classes
        from ragpill.evaluators import LLMJudge

        # Extend default evaluators with custom ones
        dataset = load_testset(
            csv_path='testset.csv',
            evaluator_classes=default_evaluator_classes | {'CustomEval': CustomEvaluator}
        )
        ```

    See Also:
        [`ragpill.base.BaseEvaluator.from_csv_line`][ragpill.base.BaseEvaluator.from_csv_line]:
            Detailed descriptions of standard parameters
        [`ragpill.csv.testset.default_evaluator_classes`][ragpill.csv.testset.default_evaluator_classes]:
            Dict of built-in evaluators
    """
    # Read CSV
    rows = _read_csv_with_encoding(csv_path)

    # Group by question
    question_to_rows = _group_rows_by_question(rows, question_column)

    # Standard columns (repeat/threshold are handled separately, not passed to evaluator attributes)
    standard_columns = {
        question_column,
        test_type_column,
        expected_column,
        tags_column,
        check_column,
        "repeat",
        "threshold",
    }

    # Extract global evaluators (rows with empty questions)
    global_evaluators: list[BaseEvaluator] = []
    global_rows = question_to_rows.pop("", None)
    if global_rows:
        for row in global_rows:
            test_type = row[test_type_column]
            evaluator_class = evaluator_classes.get(test_type)

            if evaluator_class is None:
                if skip_unknown_evaluators:
                    continue
                else:
                    raise ValueError(
                        f"Unknown evaluator type in global evaluator: {test_type}. Available types: {list(evaluator_classes.keys())}"
                    )

            evaluator, _, _ = create_evaluator_from_row(
                row, evaluator_class, standard_columns, expected_column, tags_column, check_column
            )
            global_evaluators.append(evaluator)

    # Create cases
    cases: list[Case[str, str, TestCaseMetadata]] = []
    for question, question_rows in question_to_rows.items():
        case = _create_case_from_rows(
            question=question,
            rows=question_rows,
            evaluator_classes=evaluator_classes,
            standard_columns=standard_columns,
            test_type_column=test_type_column,
            expected_column=expected_column,
            tags_column=tags_column,
            check_column=check_column,
            skip_unknown_evaluators=skip_unknown_evaluators,
        )
        if case is not None:
            cases.append(case)

    return Dataset[str, str, TestCaseMetadata](cases=cases, evaluators=global_evaluators)

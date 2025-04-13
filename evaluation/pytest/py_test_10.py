import pytest
import sys
from io import StringIO

# Token to code converter
def detokenize(tokens):
    code = ""
    indent_level = 0
    for token in tokens:
        if token == "NEW_LINE":
            code += "\n" + ("    " * indent_level)
        elif token == "INDENT":
            indent_level += 1
            code += "\n" + ("    " * indent_level)
        elif token == "DEDENT":
            indent_level -= 1
            code += "\n" + ("    " * indent_level)
        else:
            if code and code[-1] not in ("\n", " ", "(", "[", "{"):
                code += " "
            code += token
    return code.strip()

# Code runner
def run_python_code(code: str, input_data: str):
    old_stdout, old_stdin = sys.stdout, sys.stdin
    sys.stdout = output = StringIO()
    sys.stdin = StringIO(input_data)
    try:
        exec(code, {})
        return output.getvalue().strip()
    except Exception as e:
        return f"__ERROR__::{str(e)}"
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin


# Parametrized test
@pytest.mark.parametrize("model_tokens, target_tokens, input_data", [
    # Test Case 1
    (
        [  # model_tokens
            'N', '=', 'int', '(', 'input', '(', ')', ')', 'NEW_LINE',
            'z', ',', 'w', '=', '[', ']', ',', '[', ']', 'NEW_LINE',
            'K', '=', '0', 'NEW_LINE',
            'for', 'i', 'in', 'range', '(', 'N', ')', ':', 'NEW_LINE', 'INDENT',
            'x', ',', 'y', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')', 'NEW_LINE',
            'z', '.', 'append', '(', 'x', ')', 'NEW_LINE',
            'w', '.', 'append', '(', 'y', ')', 'NEW_LINE', 'DEDENT',
            'for', 'j', 'in', 'range', '(', 'N', '-', '2', ')', ':', 'NEW_LINE', 'INDENT',
            'if', 'z', '[', 'j', ']', '==', 'w', '[', 'j', ']', 'and',
            'z', '[', 'j', '+', '1', ']', '==', 'w', '[', 'j', '+', '1', ']', 'and',
            'z', '[', 'j', '+', '2', ']', '==', 'w', '[', 'j', '+', '2', ']', ':', 'NEW_LINE', 'INDENT',
            'K', '+=', '1', 'NEW_LINE', 'DEDENT', 'DEDENT',
            'if', 'K', '>=', '1', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" Yes "', ')', 'NEW_LINE', 'DEDENT',
            'else', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" No "', ')', 'NEW_LINE', 'DEDENT'
        ],
        [  # target_tokens (same for this one)
            'N', '=', 'int', '(', 'input', '(', ')', ')', 'NEW_LINE',
            'z', ',', 'w', '=', '[', ']', ',', '[', ']', 'NEW_LINE',
            'K', '=', '0', 'NEW_LINE',
            'for', 'i', 'in', 'range', '(', 'N', ')', ':', 'NEW_LINE', 'INDENT',
            'x', ',', 'y', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')', 'NEW_LINE',
            'z', '.', 'append', '(', 'x', ')', 'NEW_LINE',
            'w', '.', 'append', '(', 'y', ')', 'NEW_LINE', 'DEDENT',
            'for', 'j', 'in', 'range', '(', 'N', '-', '2', ')', ':', 'NEW_LINE', 'INDENT',
            'if', 'z', '[', 'j', ']', '==', 'w', '[', 'j', ']', 'and',
            'z', '[', 'j', '+', '1', ']', '==', 'w', '[', 'j', '+', '1', ']', 'and',
            'z', '[', 'j', '+', '2', ']', '==', 'w', '[', 'j', '+', '2', ']', ':', 'NEW_LINE', 'INDENT',
            'K', '+=', '1', 'NEW_LINE', 'DEDENT', 'DEDENT',
            'if', 'K', '>=', '1', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" Yes "', ')', 'NEW_LINE', 'DEDENT',
            'else', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" No "', ')', 'NEW_LINE', 'DEDENT'
        ],
        "5\n1 1\n2 2\n3 3\n4 4\n5 5\n"
    ),
        (
        [  # model_tokens
            'N', '=', 'int', '(', 'input', '(', ')', ')', 'NEW_LINE',
            'z', ',', 'w', '=', '[', ']', ',', '[', ']', 'NEW_LINE',
            'K', '=', '0', 'NEW_LINE',
            'for', 'i', 'in', 'range', '(', 'N', ')', ':', 'NEW_LINE', 'INDENT',
            'x', ',', 'y', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')', 'NEW_LINE',
            'z', '.', 'append', '(', 'x', ')', 'NEW_LINE',
            'w', '.', 'append', '(', 'y', ')', 'NEW_LINE', 'DEDENT',
            'for', 'j', 'in', 'range', '(', 'N', '-', '2', ')', ':', 'NEW_LINE', 'INDENT',
            'if', 'z', '[', 'j', ']', '==', 'w', '[', 'j', ']', 'and',
            'z', '[', 'j', '+', '1', ']', '==', 'w', '[', 'j', '+', '1', ']', 'and',
            'z', '[', 'j', '+', '2', ']', '==', 'w', '[', 'j', '+', '2', ']', ':', 'NEW_LINE', 'INDENT',
            'K', '+=', '1', 'NEW_LINE', 'DEDENT', 'DEDENT',
            'if', 'K', '>=', '1', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" Yes "', ')', 'NEW_LINE', 'DEDENT',
            'else', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" No "', ')', 'NEW_LINE', 'DEDENT'
        ],
        [  # target_tokens
            'N', '=', 'int', '(', 'input', '(', ')', ')', 'NEW_LINE',
            'z', ',', 'w', '=', '[', ']', ',', '[', ']', 'NEW_LINE',
            'K', '=', '0', 'NEW_LINE',
            'for', 'i', 'in', 'range', '(', 'N', ')', ':', 'NEW_LINE', 'INDENT',
            'x', ',', 'y', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')', 'NEW_LINE',
            'z', '.', 'append', '(', 'x', ')', 'NEW_LINE',
            'w', '.', 'append', '(', 'y', ')', 'NEW_LINE', 'DEDENT',
            'for', 'j', 'in', 'range', '(', 'N', '-', '2', ')', ':', 'NEW_LINE', 'INDENT',
            'if', 'z', '[', 'j', ']', '==', 'w', '[', 'j', ']', 'and',
            'z', '[', 'j', '+', '1', ']', '==', 'w', '[', 'j', '+', '1', ']', 'and',
            'z', '[', 'j', '+', '2', ']', '==', 'w', '[', 'j', '+', '2', ']', ':', 'NEW_LINE', 'INDENT',
            'K', '+=', '1', 'NEW_LINE', 'DEDENT', 'DEDENT',
            'if', 'K', '>=', '1', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" Yes "', ')', 'NEW_LINE', 'DEDENT',
            'else', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" No "', ')', 'NEW_LINE', 'DEDENT'
        ],
        "5\n1 1\n2 2\n3 3\n4 9\n5 8\n"
    ),
    (
        [  # model_tokens
            'N', '=', 'int', '(', 'input', '(', ')', ')', 'NEW_LINE',
            'z', ',', 'w', '=', '[', ']', ',', '[', ']', 'NEW_LINE',
            'K', '=', '0', 'NEW_LINE',
            'for', 'i', 'in', 'range', '(', 'N', ')', ':', 'NEW_LINE', 'INDENT',
            'x', ',', 'y', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')', 'NEW_LINE',
            'z', '.', 'append', '(', 'x', ')', 'NEW_LINE',
            'w', '.', 'append', '(', 'y', ')', 'NEW_LINE', 'DEDENT',
            'for', 'j', 'in', 'range', '(', 'N', '-', '2', ')', ':', 'NEW_LINE', 'INDENT',
            'if', 'z', '[', 'j', ']', '==', 'w', '[', 'j', ']', 'and',
            'z', '[', 'j', '+', '1', ']', '==', 'w', '[', 'j', '+', '1', ']', 'and',
            'z', '[', 'j', '+', '2', ']', '==', 'w', '[', 'j', '+', '2', ']', ':', 'NEW_LINE', 'INDENT',
            'K', '+=', '1', 'NEW_LINE', 'DEDENT', 'DEDENT',
            'if', 'K', '>=', '1', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" Yes "', ')', 'NEW_LINE', 'DEDENT',
            'else', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" No "', ')', 'NEW_LINE', 'DEDENT'
        ],
        [  # target_tokens
            'N', '=', 'int', '(', 'input', '(', ')', ')', 'NEW_LINE',
            'z', ',', 'w', '=', '[', ']', ',', '[', ']', 'NEW_LINE',
            'K', '=', '0', 'NEW_LINE',
            'for', 'i', 'in', 'range', '(', 'N', ')', ':', 'NEW_LINE', 'INDENT',
            'x', ',', 'y', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')', 'NEW_LINE',
            'z', '.', 'append', '(', 'x', ')', 'NEW_LINE',
            'w', '.', 'append', '(', 'y', ')', 'NEW_LINE', 'DEDENT',
            'for', 'j', 'in', 'range', '(', 'N', '-', '2', ')', ':', 'NEW_LINE', 'INDENT',
            'if', 'z', '[', 'j', ']', '==', 'w', '[', 'j', ']', 'and',
            'z', '[', 'j', '+', '1', ']', '==', 'w', '[', 'j', '+', '1', ']', 'and',
            'z', '[', 'j', '+', '2', ']', '==', 'w', '[', 'j', '+', '2', ']', ':', 'NEW_LINE', 'INDENT',
            'K', '+=', '1', 'NEW_LINE', 'DEDENT', 'DEDENT',
            'if', 'K', '>=', '1', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" Yes "', ')', 'NEW_LINE', 'DEDENT',
            'else', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" No "', ')', 'NEW_LINE', 'DEDENT'
        ],
        "6\n1 9\n2 8\n3 7\n4 6\n5 5\n6 4\n"
    ),
        (
        [  # model_tokens
            'N', '=', 'int', '(', 'input', '(', ')', ')', 'NEW_LINE',
            'z', ',', 'w', '=', '[', ']', ',', '[', ']', 'NEW_LINE',
            'K', '=', '0', 'NEW_LINE',
            'for', 'i', 'in', 'range', '(', 'N', ')', ':', 'NEW_LINE', 'INDENT',
            'x', ',', 'y', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')', 'NEW_LINE',
            'z', '.', 'append', '(', 'x', ')', 'NEW_LINE',
            'w', '.', 'append', '(', 'y', ')', 'NEW_LINE', 'DEDENT',
            'for', 'j', 'in', 'range', '(', 'N', '-', '2', ')', ':', 'NEW_LINE', 'INDENT',
            'if', 'z', '[', 'j', ']', '==', 'w', '[', 'j', ']', 'and',
            'z', '[', 'j', '+', '1', ']', '==', 'w', '[', 'j', '+', '1', ']', 'and',
            'z', '[', 'j', '+', '2', ']', '==', 'w', '[', 'j', '+', '2', ']', ':', 'NEW_LINE', 'INDENT',
            'K', '+=', '1', 'NEW_LINE', 'DEDENT', 'DEDENT',
            'if', 'K', '>=', '1', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" Yes "', ')', 'NEW_LINE', 'DEDENT',
            'else', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" No "', ')', 'NEW_LINE', 'DEDENT'
        ],
        [  # target_tokens (same in this case)
            'N', '=', 'int', '(', 'input', '(', ')', ')', 'NEW_LINE',
            'z', ',', 'w', '=', '[', ']', ',', '[', ']', 'NEW_LINE',
            'K', '=', '0', 'NEW_LINE',
            'for', 'i', 'in', 'range', '(', 'N', ')', ':', 'NEW_LINE', 'INDENT',
            'x', ',', 'y', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')', 'NEW_LINE',
            'z', '.', 'append', '(', 'x', ')', 'NEW_LINE',
            'w', '.', 'append', '(', 'y', ')', 'NEW_LINE', 'DEDENT',
            'for', 'j', 'in', 'range', '(', 'N', '-', '2', ')', ':', 'NEW_LINE', 'INDENT',
            'if', 'z', '[', 'j', ']', '==', 'w', '[', 'j', ']', 'and',
            'z', '[', 'j', '+', '1', ']', '==', 'w', '[', 'j', '+', '1', ']', 'and',
            'z', '[', 'j', '+', '2', ']', '==', 'w', '[', 'j', '+', '2', ']', ':', 'NEW_LINE', 'INDENT',
            'K', '+=', '1', 'NEW_LINE', 'DEDENT', 'DEDENT',
            'if', 'K', '>=', '1', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" Yes "', ')', 'NEW_LINE', 'DEDENT',
            'else', ':', 'NEW_LINE', 'INDENT', 'print', '(', '" No "', ')', 'NEW_LINE', 'DEDENT'
        ],
        "7\n1 9\n2 8\n3 7\n4 4\n5 5\n6 6\n7 1\n"
    ),
    (
    [  # model_tokens
        'x', '=', 'int', '(', 'input', '(', ')', ')', 'NEW_LINE',
        'if', 'x', '==', '0', ':', 'NEW_LINE', 'INDENT',
        'print', '(', '1', ')', 'NEW_LINE', 'DEDENT',
        'else', ':', 'NEW_LINE', 'INDENT',
        'print', '(', '0', ')', 'NEW_LINE', 'DEDENT'
    ],
    [  # target_tokens (same here too)
        'x', '=', 'int', '(', 'input', '(', ')', ')', 'NEW_LINE',
        'if', 'x', '==', '0', ':', 'NEW_LINE', 'INDENT',
        'print', '(', '1', ')', 'NEW_LINE', 'DEDENT',
        'else', ':', 'NEW_LINE', 'INDENT',
        'print', '(', '0', ')', 'NEW_LINE', 'DEDENT'
    ],
    "0\n"
),
(
    [
        'a', ',', 'b', ',', 'c', ',', 'd', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')',
        'NEW_LINE', 'print', '(', 'max', '(', 'a', '*', 'c', ',', 'a', '*', 'd', ',', 'b', '*', 'c', ',', 'b', '*', 'd', ')', ')',
        'NEW_LINE'
    ],
    [
        'a', ',', 'b', ',', 'c', ',', 'd', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')',
        'NEW_LINE', 'print', '(', 'max', '(', 'a', '*', 'c', ',', 'a', '*', 'd', ',', 'b', '*', 'c', ',', 'b', '*', 'd', ')', ')',
        'NEW_LINE'
    ],
    "1 2 3 4\n"
),
(
    [
        'a', ',', 'b', ',', 'c', ',', 'd', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')',
        'NEW_LINE', 'print', '(', 'max', '(', 'a', '*', 'c', ',', 'a', '*', 'd', ',', 'b', '*', 'c', ',', 'b', '*', 'd', ')', ')',
        'NEW_LINE'
    ],
    [
        'a', ',', 'b', ',', 'c', ',', 'd', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')',
        'NEW_LINE', 'print', '(', 'max', '(', 'a', '*', 'c', ',', 'a', '*', 'd', ',', 'b', '*', 'c', ',', 'b', '*', 'd', ')', ')',
        'NEW_LINE'
    ],
    "-1 2 3 -4\n"
),
(
    [
        'N', ',', 'X', ',', 'T', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')',
        'NEW_LINE', 'import', 'math', 'NEW_LINE', 'print', '(', 'math', '.', 'ceil', '(', 'N', '/', 'X', ')', '*', 'T', ')',
        'NEW_LINE'
    ],
    [
        'N', ',', 'X', ',', 'T', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')',
        'NEW_LINE', 'import', 'math', 'NEW_LINE', 'print', '(', 'math', '.', 'ceil', '(', 'N', '/', 'X', ')', '*', 'T', ')',
        'NEW_LINE'
    ],
    "5 2 3\n"
),
(
    [
        'A', ',', 'B', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')',
        'NEW_LINE', 'if', '(', 'A', '+', 'B', ')', '/', '2', '%', '1', '==', '0', ':',
        'NEW_LINE', 'INDENT', 'print', '(', 'int', '(', '(', 'A', '+', 'B', ')', '/', '2', ')', ')',
        'NEW_LINE', 'DEDENT', 'else', ':',
        'NEW_LINE', 'INDENT', 'print', '(', '" IMPOSSIBLE "', ')', 'NEW_LINE', 'DEDENT'
    ],
    [
        'A', ',', 'B', '=', 'map', '(', 'int', ',', 'input', '(', ')', '.', 'split', '(', ')', ')',
        'NEW_LINE', 'if', '(', 'A', '+', 'B', ')', '/', '2', '%', '1', '==', '0', ':',
        'NEW_LINE', 'INDENT', 'print', '(', 'int', '(', '(', 'A', '+', 'B', ')', '/', '2', ')', ')',
        'NEW_LINE', 'DEDENT', 'else', ':',
        'NEW_LINE', 'INDENT', 'print', '(', '" IMPOSSIBLE "', ')', 'NEW_LINE', 'DEDENT'
    ],
    "2 4\n"
),
(
    [
        'N', '=', 'int', '(', 'input', '(', ')', ')',
        'NEW_LINE', 'A', '=', '[', ']',
        'NEW_LINE', 'for', 'i', 'in', 'range', '(', 'N', ')', ':',
        'NEW_LINE', 'INDENT', 'a', '=', 'int', '(', 'input', '(', ')', ')',
        'NEW_LINE', 'A', '.', 'append', '(', 'a', ')',
        'NEW_LINE', 'DEDENT', 'AA', '=', 'sorted', '(', 'A', ',', 'reverse', '=', 'True', ')',
        'NEW_LINE', 'for', 'i', 'in', 'range', '(', 'N', ')', ':',
        'NEW_LINE', 'INDENT', 'if', 'A', '[', 'i', ']', '!=', 'AA', '[', '0', ']', ':',
        'NEW_LINE', 'INDENT', 'print', '(', 'AA', '[', '0', ']', ')',
        'NEW_LINE', 'DEDENT', 'else', ':',
        'NEW_LINE', 'INDENT', 'print', '(', 'AA', '[', '1', ']', ')',
        'NEW_LINE', 'DEDENT', 'DEDENT'
    ],
    [
        'N', '=', 'int', '(', 'input', '(', ')', ')',
        'NEW_LINE', 'A', '=', '[', ']',
        'NEW_LINE', 'for', 'i', 'in', 'range', '(', 'N', ')', ':',
        'NEW_LINE', 'INDENT', 'a', '=', 'int', '(', 'input', '(', ')', ')',
        'NEW_LINE', 'A', '.', 'append', '(', 'a', ')',
        'NEW_LINE', 'DEDENT', 'AA', '=', 'sorted', '(', 'A', ',', 'reverse', '=', 'True', ')',
        'NEW_LINE', 'for', 'i', 'in', 'range', '(', 'N', ')', ':',
        'NEW_LINE', 'INDENT', 'if', 'A', '[', 'i', ']', '!=', 'AA', '[', '0', ']', ':',
        'NEW_LINE', 'INDENT', 'print', '(', 'AA', '[', '0', ']', ')',
        'NEW_LINE', 'DEDENT', 'else', ':',
        'NEW_LINE', 'INDENT', 'print', '(', 'AA', '[', '1', ']', ')',
        'NEW_LINE', 'DEDENT', 'DEDENT'
    ],
    "3\n1\n3\n2\n"
)


])
def test_model_code_matches_target(model_tokens, target_tokens, input_data):
    model_code = detokenize(model_tokens)
    target_code = detokenize(target_tokens)

    model_output = run_python_code(model_code, input_data)
    target_output = run_python_code(target_code, input_data)

    assert model_output == target_output, f"Model: {model_output} | Target: {target_output}"

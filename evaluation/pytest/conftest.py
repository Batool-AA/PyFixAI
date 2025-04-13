import pytest

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    total = terminalreporter._numcollected
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    skipped = len(terminalreporter.stats.get("skipped", []))

    summary = (
        f"Total Tests: {total}\n"
        f"Passed: {passed}\n"
        f"Failed: {failed}\n"
        f"Skipped: {skipped}\n"
    )

    # Print report in the console
    terminalreporter.write(summary)

    # Write report to file
    with open("test_report.txt", "w") as f:
        f.write(summary)

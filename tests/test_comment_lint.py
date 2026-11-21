import unittest
from pathlib import Path

from scripts.check_comments import comment_lines, violations


class CommentLintTests(unittest.TestCase):
    def test_python_urls_and_strings(self):
        self.assertEqual(comment_lines("example.py", 'url = "https://example.com/#part"\n# comment\n'), [2])
        self.assertEqual(comment_lines("example.py", "#!/usr/bin/env python3\n# comment"), [2])

    def test_shell_and_config(self):
        self.assertEqual(comment_lines("run.sh", '#!/bin/sh\necho "# text"\n# comment\n'), [3])
        self.assertEqual(comment_lines("run.sh", "echo done;# comment"), [1])
        self.assertEqual(comment_lines("config.toml", 'url = "https://a/#b" # comment'), [1])
        self.assertEqual(comment_lines("workflow.yml", "url: https://a/#section\n# comment"), [2])
        self.assertEqual(comment_lines(".gitignore", "name#literal\n# comment"), [2])
        self.assertEqual(comment_lines("config.toml", 'key = """\n# literal content\n"""\n# comment'), [4])
        self.assertEqual(comment_lines("run.sh", "cat <<EOF\n# emitted data\nEOF\n# comment"), [4])
        self.assertEqual(comment_lines("run.sh", "cat <<<WORD\n# comment"), [2])
        self.assertEqual(comment_lines("run.sh", "x=$((1 << SHIFT))\n# comment"), [2])

    def test_file_allowances_do_not_mask_each_other(self):
        self.assertEqual(violations({"a.py": 1, "b.py": 1}, {"a.py": 2}),
                         ["b.py: 1 comment lines; allowed 0"])

    def test_ci_watches_non_python_comment_sources(self):
        workflow = Path(".github/workflows/ci.yml").read_text()
        self.assertIn('      - ".gitignore"', workflow)
        self.assertIn('      - ".requirements-*.in"', workflow)


if __name__ == "__main__":
    unittest.main()

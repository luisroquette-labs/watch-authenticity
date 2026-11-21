---
name: delete-comment
description: Audit code comments, record decisions and implementation issues, and reduce the per-file comment allowance.
---

# Delete comments

1. Read each selected comment beside its code. Classify it as tautology, stale, excuse, or decision record. Keep required shebangs and tool directives.
2. Remove tautologies and stale comments. Remove excuses and open specific issues for the underlying implementation. Move decisions into `docs/adr/` with context and consequences.
3. Lower the affected file allowances in `scripts/comment-baseline.json`; new files start at zero. Run `python3 scripts/check_comments.py` to enforce the ratchet.
4. Run `python3 -m unittest discover -s tests -p test_comment_lint.py` and focused project tests. Report removed and remaining comment lines, issues, ADRs, and the remaining estimate.

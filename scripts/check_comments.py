import io
import json
import re
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "comment-baseline.json"
GIT_BIN = shutil.which("git") or "/usr/bin/git"


def comment_lines(path: str, source: str) -> list[int]:
    if path.endswith(".py"):
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return sorted({token.start[0] for token in tokens
                       if token.type == tokenize.COMMENT
                       and not (token.start[0] == 1 and source.startswith("#!"))})
    results = []
    multiline_quote = None
    heredocs: list[tuple[str, bool]] = []
    for number, line in enumerate(source.splitlines(), 1):
        if number == 1 and line.startswith("#!"):
            continue
        if heredocs:
            delimiter, strip_tabs = heredocs[0]
            if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                heredocs.pop(0)
            continue
        if path.startswith("Containerfile.") and not line.lstrip().startswith("#"):
            continue
        quote = None
        escaped = False
        arithmetic_depth = 0
        index = 0
        while index < len(line):
            char = line[index]
            if multiline_quote:
                if line.startswith(multiline_quote, index):
                    index += 3
                    multiline_quote = None
                    continue
                if char == "\\" and multiline_quote == '"""':
                    index += 2
                    continue
                index += 1
                continue
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif quote:
                if char == quote:
                    quote = None
            elif path.endswith(".toml") and line[index:index + 3] in {"'''", '"""'}:
                multiline_quote = line[index:index + 3]
                index += 3
                continue
            elif char in {"'", '"'}:
                quote = char
            elif path.endswith(".sh") and (line.startswith("$((", index) or line.startswith("((", index)):
                arithmetic_depth += 1
                index += 3 if line.startswith("$((", index) else 2
                continue
            elif arithmetic_depth and line.startswith("))", index):
                arithmetic_depth -= 1
                index += 2
                continue
            elif (path.endswith(".sh") and arithmetic_depth == 0
                  and line.startswith("<<", index)
                  and (index == 0 or line[index - 1] != "<")
                  and not line.startswith("<<<", index)):
                marker = re.match(r"<<(-?)\s*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))", line[index:])
                if marker:
                    heredocs.append((next(value for value in marker.groups()[1:] if value), bool(marker.group(1))))
                    index += len(marker.group(0))
                    continue
            elif char == "#" and (
                path.endswith(".toml") or index == 0
                or line[index - 1].isspace() or line[index - 1] in ";|&()"
            ) and (path != ".gitignore" or not line[:index].strip()):
                results.append(number)
                break
            index += 1
    return results


def tracked_sources() -> list[str]:
    files = subprocess.check_output(
        [GIT_BIN, "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    return sorted(
        file for file in files
        if file and (file.endswith((".py", ".sh", ".yml", ".yaml", ".toml"))
                     or file.startswith("Containerfile.")
                     or file.endswith(".in") and "requirements" in Path(file).name
                     or file in {".gitignore", "requirements.txt"})
        and (ROOT / file).is_file()
    )


def current_counts() -> dict[str, int]:
    return {
        file: len(comment_lines(file, (ROOT / file).read_text()))
        for file in tracked_sources()
    }


def violations(current: dict[str, int], baseline: dict[str, int]) -> list[str]:
    return [
        f"{file}: {count} comment lines; allowed {baseline.get(file, 0)}"
        for file, count in current.items()
        if count > baseline.get(file, 0)
    ]


if __name__ == "__main__":
    counts = current_counts()
    if "--write-baseline" in sys.argv:
        BASELINE.write_text(json.dumps(counts, indent=2) + "\n")
    else:
        failures = violations(counts, json.loads(BASELINE.read_text()))
        if failures:
            print("Comment ratchet failed:\n" + "\n".join(failures), file=sys.stderr)
            raise SystemExit(1)
        print(f"Comment ratchet passed: {sum(counts.values())} remaining comment lines")

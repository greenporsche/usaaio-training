#!/usr/bin/env python3
"""
Convert problem variation markdown files to Jupyter notebooks.

Goals:
- Keep content readable in notebooks (spacing/"breathing")
- Preserve code blocks as code cells (Python)
- Render math properly with LaTeX

Usage:
  uv run python convert_to_notebooks.py
  uv run python convert_to_notebooks.py problem-01-variations.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# =============================================================================
# Unicode → LaTeX symbol mappings
# =============================================================================

UNICODE_TO_LATEX = {
    # Greek letters
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "λ": r"\lambda",
    "μ": r"\mu",
    "σ": r"\sigma",
    "τ": r"\tau",
    "φ": r"\phi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Σ": r"\Sigma",
    "Π": r"\Pi",
    "Λ": r"\Lambda",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    # Special symbols
    "∈": r"\in",
    "∉": r"\notin",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "∪": r"\cup",
    "∩": r"\cap",
    "∅": r"\emptyset",
    "∞": r"\infty",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "±": r"\pm",
    "×": r"\times",
    "÷": r"\div",
    "·": r"\cdot",
    "⊥": r"\perp",
    "∥": r"\parallel",
    "∂": r"\partial",
    "∇": r"\nabla",
    "→": r"\rightarrow",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
    "⇔": r"\Leftrightarrow",
    "∀": r"\forall",
    "∃": r"\exists",
    "ℝ": r"\mathbb{R}",
    "ℤ": r"\mathbb{Z}",
    "ℕ": r"\mathbb{N}",
    "ℂ": r"\mathbb{C}",
    # Superscripts
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
    "⁺": "^+",
    "⁻": "^-",
    "ⁿ": "^n",
    # Subscripts
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "₃": "_3",
    "₄": "_4",
    "₅": "_5",
    "₆": "_6",
    "₇": "_7",
    "₈": "_8",
    "₉": "_9",
    "ₐ": "_a",
    "ₑ": "_e",
    "ᵢ": "_i",
    "ⱼ": "_j",
    "ₖ": "_k",
    "ₘ": "_m",
    "ₙ": "_n",
    "ₚ": "_p",
    "ₛ": "_s",
    "ₜ": "_t",
    "ₓ": "_x",
    # Special characters
    "ê": r"\hat{e}",
    "x̂": r"\hat{x}",
    "ŷ": r"\hat{y}",
}


def _extract_code_from_details(details_content: str) -> tuple[str, list[str]]:
    """
    Extract Python code blocks from a <details> block.

    Returns:
        tuple of (details_without_code, list_of_code_blocks)
    """
    code_blocks: list[str] = []
    code_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)

    def extract_and_remove(match: re.Match) -> str:
        code_blocks.append(match.group(1).strip())
        return ""  # Remove from details

    details_without_code = code_pattern.sub(extract_and_remove, details_content)
    # Clean up extra blank lines left by removed code blocks
    details_without_code = re.sub(r"\n{3,}", "\n\n", details_without_code)

    # If details is now essentially empty (just tags), add a placeholder
    # Check if there's any meaningful content between <summary>...</summary> and </details>
    content_check = re.search(r"</summary>\s*(.*?)\s*</details>", details_without_code, re.DOTALL)
    if content_check and not content_check.group(1).strip():
        # Insert placeholder before </details>
        details_without_code = details_without_code.replace(
            "</details>",
            "\n*See code cell below.*\n</details>"
        )

    return details_without_code.strip(), code_blocks


def parse_markdown_to_cells(content: str) -> list[dict]:
    """
    Split markdown into a list of cells, turning fenced Python code blocks into code cells.
    Code blocks inside <details> tags are extracted and placed as code cells right after.
    """
    cells: list[dict] = []

    # First, protect <details>...</details> blocks by keeping them intact
    # Split content by <details> blocks
    details_pattern = re.compile(r"(<details>.*?</details>)", re.DOTALL)
    parts = details_pattern.split(content)

    for part in parts:
        if not part.strip():
            continue

        if part.startswith("<details>"):
            # Extract code blocks and place them after the details
            details_text, code_blocks = _extract_code_from_details(part.strip())
            cells.append({"type": "markdown", "content": details_text})
            # Add extracted code as executable cells right after
            for code in code_blocks:
                cells.append({"type": "code", "content": code})
        else:
            # Regular content - extract Python code blocks as code cells
            code_block_pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
            last_end = 0

            for match in code_block_pattern.finditer(part):
                markdown_before = part[last_end : match.start()].strip()
                if markdown_before:
                    cells.append({"type": "markdown", "content": markdown_before})

                language = (match.group(1) or "").strip().lower()
                code_content = (match.group(2) or "").strip()

                if language == "python":
                    cells.append({"type": "code", "content": code_content})
                else:
                    cells.append(
                        {
                            "type": "markdown",
                            "content": f"```{language}\n{code_content}\n```".strip(),
                        }
                    )

                last_end = match.end()

            remaining = part[last_end:].strip()
            if remaining:
                cells.append({"type": "markdown", "content": remaining})

    return cells


def split_by_headers(cells: list[dict]) -> list[dict]:
    """
    Further split markdown cells at major section headers to improve notebook navigation.
    """
    out: list[dict] = []
    header_split = re.compile(r"(?=^##+ )", re.MULTILINE)

    for cell in cells:
        if cell["type"] != "markdown":
            out.append(cell)
            continue

        parts = [p.strip() for p in header_split.split(cell["content"]) if p.strip()]
        out.extend({"type": "markdown", "content": p} for p in parts)

    return out


# =============================================================================
# Math conversion helpers
# =============================================================================


def _is_inside_fenced_block(text: str, pos: int) -> bool:
    """Check if position is inside a fenced code block."""
    before = text[:pos]
    fence_count = before.count("```")
    return fence_count % 2 == 1


def _convert_unicode_symbols(text: str) -> str:
    """Convert Unicode math symbols to LaTeX equivalents (inside math mode)."""
    for unicode_char, latex in UNICODE_TO_LATEX.items():
        text = text.replace(unicode_char, latex)
    return text


def _wrap_in_math(content: str, display: bool = False) -> str:
    """Wrap content in math delimiters."""
    content = content.strip()
    if not content:
        return content
    if display:
        return f"$${content}$$"
    return f"${content}$"


def _convert_vector_notation(text: str) -> str:
    """Convert vector notation like [a, b, c]^T to proper LaTeX."""

    def convert_inner_content(inner: str) -> str:
        """Convert Greek letters and symbols inside vector brackets."""
        # Convert Greek letters to LaTeX (without wrapping in $)
        greek_map = {
            "θ": "\\theta", "α": "\\alpha", "β": "\\beta", "γ": "\\gamma",
            "δ": "\\delta", "λ": "\\lambda", "μ": "\\mu", "σ": "\\sigma",
            "φ": "\\phi", "ω": "\\omega", "π": "\\pi",
        }
        for greek, latex in greek_map.items():
            inner = inner.replace(greek, latex)
        return inner

    # [a, b, c]^T → $[a, b, c]^\top$
    def vec_replace(m):
        inner = convert_inner_content(m.group(1))
        return _wrap_in_math(f"[{inner}]^\\top")

    text = re.sub(r"\[([^\]]+)\]\^T\b", vec_replace, text)

    # Also handle [a,b,c]^\top that's not wrapped
    text = re.sub(r"(?<!\$)\[([^\]]+)\]\^\\top(?!\$)", lambda m: _wrap_in_math(f"[{convert_inner_content(m.group(1))}]^\\top"), text)

    return text


def _convert_norms(text: str) -> str:
    """Convert norm notation ||x|| to proper LaTeX."""

    def convert_inner(inner: str) -> str:
        """Convert special chars inside norm to LaTeX."""
        inner = inner.replace("ê", "\\hat{e}")
        # Handle subscripted versions
        inner = re.sub(r"ê([₀₁₂₃₄₅₆₇₈₉]+)", lambda m: f"\\hat{{e}}_{{{_subscript_to_num(m.group(1))}}}", inner)
        # Escape underscores in variable names (e.g., x_parallel → x\_parallel)
        # But only if followed by letters (not digits, which are real subscripts)
        inner = re.sub(r"_([a-zA-Z]{2,})", r"\\_\1", inner)
        return inner

    # ||x||² or ||x||^2 → $\|x\|^2$
    def norm_sq_replace(m):
        inner = convert_inner(m.group(1))
        return _wrap_in_math(f"\\|{inner}\\|^2")

    text = re.sub(r"\|\|([^|]+)\|\|[²\^2]", norm_sq_replace, text)

    # ||x|| → $\|x\|$
    def norm_replace(m):
        inner = convert_inner(m.group(1))
        return _wrap_in_math(f"\\|{inner}\\|")

    text = re.sub(r"\|\|([^|]+)\|\|", norm_replace, text)
    return text


def _subscript_to_num(s: str) -> str:
    """Convert Unicode subscripts to numbers."""
    mapping = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}
    return "".join(mapping.get(c, c) for c in s)


def _convert_sqrt(text: str) -> str:
    """Convert square root notation."""
    # √(expr) → $\sqrt{expr}$
    text = re.sub(r"√\(([^)]+)\)", lambda m: _wrap_in_math(f"\\sqrt{{{m.group(1)}}}"), text)

    # Complex patterns: i√3, -i√3
    text = re.sub(
        r"([+-]?i)√(\d+)",
        lambda m: _wrap_in_math(f"{m.group(1)}\\sqrt{{{m.group(2)}}}"),
        text,
    )

    # Fraction with sqrt in denominator: 1/√2 → $1/\sqrt{2}$ or $\frac{1}{\sqrt{2}}$
    text = re.sub(
        r"(\d+)/√(\d+)",
        lambda m: _wrap_in_math(f"\\frac{{{m.group(1)}}}{{\\sqrt{{{m.group(2)}}}}}"),
        text,
    )

    # Coefficient + sqrt: 2√3 → $2\sqrt{3}$, 2√n → $2\sqrt{n}$
    text = re.sub(
        r"(\d+)√(\d+)",
        lambda m: _wrap_in_math(f"{m.group(1)}\\sqrt{{{m.group(2)}}}"),
        text,
    )
    text = re.sub(
        r"(\d+)√([a-zA-Z])\b",
        lambda m: _wrap_in_math(f"{m.group(1)}\\sqrt{{{m.group(2)}}}"),
        text,
    )

    # √n → $\sqrt{n}$
    text = re.sub(r"√(\d+)", lambda m: _wrap_in_math(f"\\sqrt{{{m.group(1)}}}"), text)
    # √a where a is a letter
    text = re.sub(r"√([a-zA-Z])\b", lambda m: _wrap_in_math(f"\\sqrt{{{m.group(1)}}}"), text)

    # Any remaining √ - wrap minimally
    text = re.sub(r"√", r"$\\sqrt{}$", text)
    return text


def _convert_fractions(text: str) -> str:
    """Convert fractions to proper LaTeX."""
    # (1/a) or (1/2) or (1/√a) → $\frac{1}{a}$ etc.
    def paren_frac_replace(m):
        num, den = m.group(1), m.group(2)
        # Convert √ in denominator to \sqrt{}
        if "√" in den:
            den = re.sub(r"√(\w+)", r"\\sqrt{\1}", den)
        return _wrap_in_math(f"\\frac{{{num}}}{{{den}}}")

    # Match (num/den) where num and den are simple (letters, numbers, √)
    text = re.sub(r"\((\d+)/([√]?\w+)\)", paren_frac_replace, text)

    # = a/b at end or before space → = $\frac{a}{b}$
    def eq_frac_replace(m):
        num, den = m.group(1), m.group(2)
        # Only convert if simple numbers
        if num.lstrip("-").isdigit() and den.isdigit():
            return f"= {_wrap_in_math(f'\\frac{{{num}}}{{{den}}}')}"
        return m.group(0)

    text = re.sub(r"=\s*(-?\d+)/(\d+)(?=\s|$|\.|,)", eq_frac_replace, text)

    return text


def _convert_subscripts_superscripts(text: str) -> str:
    """Convert common subscript/superscript patterns."""
    # Convert patterns like x_1, P², etc. that aren't already in math mode
    # ê₁ → $\hat{e}_1$
    text = re.sub(r"ê([₀₁₂₃₄₅₆₇₈₉]+)", lambda m: _wrap_in_math(f"\\hat{{e}}{_convert_unicode_symbols(m.group(1))}"), text)

    # λ₁ → $\lambda_1$
    text = re.sub(r"λ([₀₁₂₃₄₅₆₇₈₉]+)", lambda m: _wrap_in_math(f"\\lambda{_convert_unicode_symbols(m.group(1))}"), text)

    return text


def _convert_standalone_greek(text: str) -> str:
    """Convert standalone Greek letters to math mode."""
    # Standalone θ, λ, etc.
    greek_pattern = r"\b([αβγδεζηθλμσφψωΣΛ])\b"

    def greek_replace(m):
        letter = m.group(1)
        latex = UNICODE_TO_LATEX.get(letter, letter)
        return _wrap_in_math(latex)

    text = re.sub(greek_pattern, greek_replace, text)
    return text


def _convert_math_expressions(text: str) -> str:
    """
    Identify and convert mathematical expressions to LaTeX.
    This is the main function that processes markdown text.
    """
    # Don't process inside fenced code blocks
    chunks = text.split("```")
    for i in range(0, len(chunks), 2):  # Only process even indices (outside fences)
        chunk = chunks[i]

        # Apply conversions in order
        chunk = _convert_vector_notation(chunk)
        chunk = _convert_norms(chunk)
        chunk = _convert_sqrt(chunk)
        chunk = _convert_subscripts_superscripts(chunk)
        chunk = _convert_fractions(chunk)
        chunk = _convert_standalone_greek(chunk)

        # Convert remaining special Unicode symbols that appear in math contexts
        # Look for patterns that suggest math: equations, operators, etc.

        # x · y → $x \cdot y$ (dot product) - both plain text
        chunk = re.sub(
            r"(\b[a-zA-Z_][a-zA-Z0-9_]*)\s*·\s*(\b[a-zA-Z_][a-zA-Z0-9_]*)",
            lambda m: _wrap_in_math(f"{m.group(1)} \\cdot {m.group(2)}"),
            chunk,
        )
        # x · $math$ → $x \cdot math$ (first is plain, second is math)
        chunk = re.sub(
            r"(\b[a-zA-Z_][a-zA-Z0-9_]*)\s*·\s*\$([^$]+)\$",
            lambda m: _wrap_in_math(f"{m.group(1)} \\cdot {m.group(2)}"),
            chunk,
        )
        # Digits: 1·1 → $1 \cdot 1$
        chunk = re.sub(
            r"(\d+)\s*·\s*(\d+)",
            lambda m: _wrap_in_math(f"{m.group(1)} \\cdot {m.group(2)}"),
            chunk,
        )
        # Standalone · between any expressions - catch all
        chunk = re.sub(r"·", r"$\\cdot$", chunk)

        # r ⊥ v → $r \perp v$ - both plain text
        chunk = re.sub(
            r"(\b[a-zA-Z_][a-zA-Z0-9_]*)\s*⊥\s*(\b[a-zA-Z_][a-zA-Z0-9_]*)",
            lambda m: _wrap_in_math(f"{m.group(1)} \\perp {m.group(2)}"),
            chunk,
        )
        # r ⊥ $math$ → $r \perp math$ (first is plain, second is math)
        chunk = re.sub(
            r"(\b[a-zA-Z_][a-zA-Z0-9_]*)\s*⊥\s*\$([^$]+)\$",
            lambda m: _wrap_in_math(f"{m.group(1)} \\perp {m.group(2)}"),
            chunk,
        )
        # $math$ ⊥ $math$ → $math \perp math$ (both already math)
        chunk = re.sub(
            r"\$([^$]+)\$\s*⊥\s*\$([^$]+)\$",
            lambda m: _wrap_in_math(f"{m.group(1)} \\perp {m.group(2)}"),
            chunk,
        )
        # r ⊥ (expr) → $r \perp (expr)$ (with parentheses)
        chunk = re.sub(
            r"(\b[a-zA-Z_][a-zA-Z0-9_]*)\s*⊥\s*(\([^)]+\))",
            lambda m: _wrap_in_math(f"{m.group(1)} \\perp {m.group(2)}"),
            chunk,
        )
        # Handle subscripted vars: u₁ ⊥ u₂
        chunk = re.sub(
            r"(\b[a-zA-Z][₀₁₂₃₄₅₆₇₈₉]+)\s*⊥\s*(\b[a-zA-Z][₀₁₂₃₄₅₆₇₈₉]+)",
            lambda m: _wrap_in_math(f"{_convert_unicode_symbols(m.group(1))} \\perp {_convert_unicode_symbols(m.group(2))}"),
            chunk,
        )
        # Any remaining standalone ⊥
        chunk = re.sub(r"⊥", r"$\\perp$", chunk)

        # y ∈ W → $y \in W$ (general membership - single letters)
        chunk = re.sub(
            r"\b([a-zA-Z])\s*∈\s*([a-zA-Z])\b",
            lambda m: _wrap_in_math(f"{m.group(1)} \\in {m.group(2)}"),
            chunk,
        )

        # ∈ ℝⁿ patterns - convert Unicode superscripts to plain numbers for the exponent
        def convert_superscript_to_number(s):
            mapping = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "ⁿ": "n"}
            return "".join(mapping.get(c, c) for c in s)

        chunk = re.sub(r"∈\s*ℝ([⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]+)", lambda m: _wrap_in_math(f"\\in \\mathbb{{R}}^{{{convert_superscript_to_number(m.group(1))}}}"), chunk)
        chunk = re.sub(r"∈\s*ℝ\^(\d+)", lambda m: _wrap_in_math(f"\\in \\mathbb{{R}}^{{{m.group(1)}}}"), chunk)
        chunk = re.sub(r"∈\s*ℝ\^\(([^)]+)\)", lambda m: _wrap_in_math(f"\\in \\mathbb{{R}}^{{{m.group(1)}}}"), chunk)
        # Standalone ∈ not yet converted
        chunk = re.sub(r"∈", r"$\\in$", chunk)

        # Standalone ê → $\hat{e}$
        chunk = re.sub(r"\bê\b", lambda m: _wrap_in_math(r"\hat{e}"), chunk)

        chunks[i] = chunk

    return "```".join(chunks)


def _breathe_solutions(markdown: str) -> str:
    """
    Add line breaks within solutions so equations don't run together.

    In markdown, single newlines become spaces. We need blank lines or <br> for breaks.
    """
    # Within <details> blocks, convert single newlines to paragraph breaks
    # This ensures each equation step has breathing room
    def add_breaks_in_details(match: re.Match) -> str:
        content = match.group(0)
        # After </summary>, add spacing before equation lines
        parts = content.split("</summary>", 1)
        if len(parts) == 2:
            summary, body = parts
            # Add blank line (paragraph break) before lines starting with equation patterns
            # Variable assignments: r = , proj = , x = , etc.
            body = re.sub(r"\n([a-zA-Z_][a-zA-Z0-9_]*\s*=)", r"\n\n\1", body)
            # Norm calculations: ||
            body = re.sub(r"\n(\|?\|[^|])", r"\n\n\1", body)
            # Also break before $\| (LaTeX norm)
            body = re.sub(r"\n(\$\\?\|)", r"\n\n\1", body)
            return summary + "</summary>" + body
        return content

    markdown = re.sub(r"<details>.*?</details>", add_breaks_in_details, markdown, flags=re.DOTALL)
    return markdown


def _breathe_parts(markdown: str) -> str:
    """
    Keep parts inline with content, but add blank line between each part.
    Also add spacing between variations.
    """
    # Add blank line BEFORE each **Part X.X**: line (but keep part inline with its content)
    markdown = re.sub(
        r"(?m)^(\*\*Part \d+\.\d+\*\*:)",
        r"\n\1",
        markdown,
    )

    # Add blank line BEFORE each solution part **X.X**: line
    markdown = re.sub(
        r"(?m)^(\*\*\d+\.\d+\*\*:)",
        r"\n\1",
        markdown,
    )

    # Add horizontal rule and spacing before each ### Variation header
    markdown = re.sub(
        r"(?m)^(### Variation [A-Z]\d+)",
        r"\n---\n\n\1",
        markdown,
    )

    # Clean up any excessive blank lines (more than 2 consecutive)
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)

    # Remove leading --- if it appears at the very start after a category header
    markdown = re.sub(r"(## CATEGORY [A-Z][^\n]*\n)\n*---\n*", r"\1\n", markdown)

    return markdown


def _cleanup_nested_math(text: str) -> str:
    r"""Fix nested/broken math delimiters."""
    # Merge adjacent math blocks: $a$$b$ → $a b$ (with space)
    # This happens when fraction and vector are converted separately
    text = re.sub(r"\$([^$]+)\$\$([^$]+)\$", r"$\1 \2$", text)

    # Fix specifically malformed patterns like $\|$\hat{e}$\|$
    text = re.sub(r"\$\\?\|?\$([^$]+)\$\\?\|?\$", r"$\|\1\|$", text)

    # Fix doubled exponents: $...\|^2$^2 → $...\|^2$
    text = re.sub(r"\$([^$]*\^2)\$\^2", r"$\1$", text)

    # Fix $...$^2 (exponent outside math) → $...^2$
    text = re.sub(r"\$([^$]+)\$\^(\d+)", r"$\1^{\2}$", text)

    return text


def _wrap_stray_latex(text: str) -> str:
    r"""
    Find LaTeX commands outside of math mode and wrap them in $...$.

    This is a final cleanup pass to catch any LaTeX that wasn't wrapped
    by the earlier conversion functions.
    """
    # Don't process inside fenced code blocks
    chunks = text.split("```")

    for i in range(0, len(chunks), 2):  # Only process even indices (outside fences)
        chunk = chunks[i]

        # Build list of (start, end) for existing math regions
        math_regions = []
        for m in re.finditer(r'\$\$[^$]+\$\$|\$[^$]+\$', chunk):
            math_regions.append((m.start(), m.end()))

        def is_in_math(pos: int) -> bool:
            """Check if position is inside a math region."""
            for start, end in math_regions:
                if start <= pos < end:
                    return True
            return False

        # Patterns to find and wrap (if not already in math mode)
        # Pattern 1: \command{...} like \hat{e}, \sqrt{3}, \frac{a}{b}, \mathbb{R}
        # Pattern 2: \command followed by _ or ^ like \lambda_1, \hat{e}_1
        # Pattern 3: \| ... \| (norm notation)
        # Pattern 4: standalone \command like \alpha, \beta, \theta

        replacements = []

        # Find \command{...} patterns (possibly with subscript/superscript)
        for m in re.finditer(r'\\[a-zA-Z]+\{[^}]*\}(?:[_^]\{?[^}\s]*\}?)?', chunk):
            if not is_in_math(m.start()):
                replacements.append((m.start(), m.end(), f'${m.group()}$'))

        # Find \| ... \| norm patterns
        for m in re.finditer(r'\\?\|[^|]+\\?\|(?:\^2)?', chunk):
            content = m.group()
            # Only if it contains backslash (LaTeX) and not already wrapped
            if '\\' in content and not is_in_math(m.start()):
                # Check it's not already $...$
                start = m.start()
                if start > 0 and chunk[start-1] == '$':
                    continue
                replacements.append((m.start(), m.end(), f'${content}$'))

        # Find standalone Greek letters and other LaTeX commands
        for m in re.finditer(r'\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega|Lambda|Sigma|Phi|Psi|Omega|in|notin|subset|cup|cap|infty|leq|geq|neq|approx|pm|times|cdot|perp|parallel|rightarrow|leftarrow|forall|exists|sqrt|frac|hat|vec|bar|dot|tilde|mathbb|mathrm|mathbf|mathcal)(?![a-zA-Z{])', chunk):
            if not is_in_math(m.start()):
                replacements.append((m.start(), m.end(), f'${m.group()}$'))

        # Apply replacements in reverse order to preserve positions
        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, replacement in replacements:
            # Double-check we're not creating $$...$$
            if start > 0 and chunk[start-1] == '$':
                continue
            if end < len(chunk) and chunk[end] == '$':
                continue
            chunk = chunk[:start] + replacement + chunk[end:]

        chunks[i] = chunk

    return "```".join(chunks)


def improve_markdown(markdown: str) -> str:
    """Apply all markdown improvements."""
    # First, add spacing for readability
    markdown = _breathe_parts(markdown)
    # Add line breaks within solutions
    markdown = _breathe_solutions(markdown)
    # Then convert math expressions
    markdown = _convert_math_expressions(markdown)
    # Clean up any malformed math delimiters
    markdown = _cleanup_nested_math(markdown)
    # Final pass: wrap any stray LaTeX commands that weren't caught
    markdown = _wrap_stray_latex(markdown)
    return markdown


def create_notebook(cells: list[dict]) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    for cell in cells:
        if cell["type"] == "markdown":
            nb.cells.append(new_markdown_cell(improve_markdown(cell["content"])))
        else:
            nb.cells.append(new_code_cell(cell["content"]))

    return nb


def convert_file(md_path: Path) -> Path:
    content = md_path.read_text(encoding="utf-8")
    cells = parse_markdown_to_cells(content)
    cells = split_by_headers(cells)
    nb = create_notebook(cells)

    out_path = md_path.with_suffix(".ipynb")
    with out_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return out_path


def main() -> int:
    base_dir = Path(__file__).parent

    if len(sys.argv) > 1:
        md_files = [Path(p) if Path(p).is_absolute() else base_dir / p for p in sys.argv[1:]]
    else:
        md_files = sorted(base_dir.glob("problem-*-variations.md"))

    if not md_files:
        print("No markdown files found.")
        return 1

    ok = 0
    for md in md_files:
        try:
            out = convert_file(md)
            print(f"✓ Converted: {md.name} → {out.name}")
            ok += 1
        except Exception as e:
            print(f"✗ Error converting {md.name}: {e}")

    print(f"Converted {ok}/{len(md_files)} files.")
    return 0 if ok == len(md_files) else 1


if __name__ == "__main__":
    raise SystemExit(main())

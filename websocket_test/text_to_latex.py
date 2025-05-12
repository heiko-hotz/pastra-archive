import re
from sympy import latex, Symbol, Function, SympifyError
from sympy.parsing.sympy_parser import (
    parse_expr,
    lambda_notation,
    auto_symbol,
    auto_number,
    factorial_notation,
    implicit_multiplication_application, # Handles 2x, (x)(y), x y etc.
    convert_xor,
    function_exponentiation,
    rationalize
)
import datetime # For printing current date in example

# Define the curated list of transformations
# We remove split_symbols_a from standard_transformations to keep multi-letter vars intact
CURATED_TRANSFORMATIONS = (
    lambda_notation,
    auto_symbol,        # Treat undefined names as Symbols
    auto_number,        # Convert numeric strings to SymPy numbers
    factorial_notation, # Convert '!' to factorial
    implicit_multiplication_application, # Crucial for '2x', 'x y', '(a)(b)'
    convert_xor,        # Allow '^' for xor if not caught as power
    function_exponentiation, # Allows f**2(x) -> f(x)**2
    rationalize         # Convert floats to rationals if possible
)

def _is_mangled_to_single_symbols(original_text: str, sympy_expr, threshold_factor=0.8, min_alphanum_chars_for_mangling=4) -> bool:
    """
    Heuristic to detect if original_text (potentially a phrase or complex expression)
    was mangled by the parser into a proliferation of single-letter SymPy Symbols.
    """
    # Count alphanumeric characters (and underscores) in the original text
    alphanum_underscore_original = re.sub(r'[^a-zA-Z0-9_]', '', original_text)
    
    if len(alphanum_underscore_original) < min_alphanum_chars_for_mangling:
        return False

    actual_symbols = sympy_expr.atoms(Symbol)
    single_letter_alpha_symbols_count = 0
    for s in actual_symbols:
        if len(s.name) == 1 and s.name.isalpha():
            single_letter_alpha_symbols_count += 1
    
    if single_letter_alpha_symbols_count >= len(alphanum_underscore_original) * threshold_factor:
        # print(f"DEBUG Mangling (helper) detected for '{original_text}': single_letter_symbols={single_letter_alpha_symbols_count}, alphanum_orig_len={len(alphanum_underscore_original)}")
        return True
    return False

def string_to_latex_aggressive(segment_text: str):
    """
    Attempts to parse a text segment as a mathematical formula and convert it to LaTeX.
    Prioritizes recall, so it tries to be lenient.
    """
    try:
        processed_text = segment_text.strip()
        if not processed_text:
            return None

        processed_text = processed_text.replace('^', '**')
        
        local_dict_for_parser = {}

        is_single_word_alphanum = bool(re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", processed_text))

        if '=' in processed_text:
            parts = processed_text.split('=', 1)
            lhs_str = parts[0].strip()
            rhs_str = parts[1].strip()
            lhs_latex_output = ""
            parsed_lhs_successfully = False
            is_lhs_single_word = bool(re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", lhs_str))

            # Try to parse LHS as math if it looks like a simple variable or function call
            if re.fullmatch(r"[a-zA-Z_][\w.]*(\s*\(.*\))?", lhs_str) and not (" " in lhs_str.split('(')[0] and '(' not in lhs_str):
                try:
                    lhs_expr = parse_expr(lhs_str,
                                          local_dict=local_dict_for_parser,
                                          transformations=CURATED_TRANSFORMATIONS,
                                          evaluate=False)
                    
                    # Safeguard 1.1 (LHS - specific single word mangled, e.g., "LHSvar" -> L*H*S*v*a*r)
                    if is_lhs_single_word and lhs_expr.is_Mul and len(lhs_str) > 1 and len(lhs_expr.args) == len(lhs_str):
                        all_single_constituents = all(arg.is_Symbol and len(arg.name) == 1 and arg.name.isalpha() for arg in lhs_expr.args)
                        if all_single_constituents:
                            lhs_latex_output = latex(Symbol(lhs_str))
                            parsed_lhs_successfully = True
                    
                    # Safeguard 1.2 (LHS - general phrase/expression mangling)
                    if not parsed_lhs_successfully and _is_mangled_to_single_symbols(lhs_str, lhs_expr):
                        lhs_latex_output = "\\\\text{" + lhs_str.replace("_", r"\\_") + "}"
                        parsed_lhs_successfully = True

                    if not parsed_lhs_successfully:
                        lhs_latex_output = latex(lhs_expr)
                        parsed_lhs_successfully = True

                except (SympifyError, SyntaxError, TypeError, Exception):
                    # Parsing failed, try fallback to Symbol or text
                    if is_lhs_single_word:
                        lhs_latex_output = latex(Symbol(lhs_str))
                    else:
                        lhs_latex_output = "\\\\text{" + lhs_str.replace("_", r"\\_") + "}"
                    parsed_lhs_successfully = True
            
            if not parsed_lhs_successfully: # If initial regex failed or other paths didn't set it
                lhs_latex_output = "\\\\text{" + lhs_str.replace("_", r"\\_") + "}"

            rhs_expr = parse_expr(rhs_str,
                                  local_dict=local_dict_for_parser,
                                  transformations=CURATED_TRANSFORMATIONS,
                                  evaluate=False)
            return f"{lhs_latex_output} = {latex(rhs_expr)}"
        else:
            sympy_expr = parse_expr(processed_text,
                                    local_dict=local_dict_for_parser,
                                    transformations=CURATED_TRANSFORMATIONS,
                                    evaluate=False)
            
            # Safeguard 3.1 (Standalone - specific single word mangled)
            if is_single_word_alphanum and sympy_expr.is_Mul and len(processed_text) > 1 and len(sympy_expr.args) == len(processed_text):
                all_single_char_constituents = all(arg.is_Symbol and len(arg.name) == 1 and arg.name.isalpha() for arg in sympy_expr.args)
                if all_single_char_constituents:
                    return latex(Symbol(processed_text))

            # Safeguard 3.2 (Standalone - general phrase/expression mangling)
            if _is_mangled_to_single_symbols(processed_text, sympy_expr):
                return "\\\\text{" + processed_text.replace("_", r"\\_") + "}"

            return latex(sympy_expr)

    except (SympifyError, SyntaxError, TypeError, Exception) as e:
        return None


def generate_candidate_segments(text_line: str):
    """
    Generates potential mathematical segments from a line of text.
    """
    candidates = set()
    text_line = text_line.strip()
    if not text_line:
        return []

    # 1. The whole line itself
    candidates.add(text_line)

    # 2. If an equals sign is present, consider LHS and RHS individually
    if '=' in text_line:
        parts = text_line.split('=', 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        if lhs: candidates.add(lhs)
        if rhs: candidates.add(rhs)

    # 3. Regex for common function call patterns: func(...)
    func_pattern = re.compile(
        r"""
        \b(?P<func_name>sqrt|cbrt|sin|cos|tan|asin|acos|atan|log|ln|exp|abs|min|max|sum|prod|integrate|diff|limit|gamma|beta|zeta|erf)\s*
        \(                                      # Opening parenthesis
        (?P<args>                               # Named group for arguments
            (?:
                [^()]* # Match characters other than parentheses
                |
                \(                              # OR - Opening inner parenthesis
                    (?: [^()]* | \( [^()]* \) )* # Match non-parentheses or one level of nested parentheses
                \)                              # Closing inner parenthesis
            )* # Repeat argument part zero or more times
        )
        \)                                      # Closing outer parenthesis
        """, re.VERBOSE | re.IGNORECASE
    )
    for match in func_pattern.finditer(text_line):
        candidates.add(match.group(0).strip())

    # 4. Regex for simple expressions: operand OPERATOR operand
    simple_expr_pattern = re.compile(
        r"""
        ([\w\.\(\)]+)                             # Operand 1 (word chars, dot, parens)
        \s* # Optional space
        (\+|-|\*(?!\*)|/(?!/)|%|\*{2})            # Operator (avoid single * if part of **, single / if part of //)
        \s* # Optional space
        ([\w\.\(\)]+)                             # Operand 2
        """, re.VERBOSE
    )
    for match in simple_expr_pattern.finditer(text_line):
        candidates.add(match.group(0).strip())

    return sorted([c for c in list(candidates) if c], key=len, reverse=True)


def find_formulas_aggressive(text: str):
    """
    Finds and converts mathematical formulas in text, prioritizing recall.
    Returns a list of dictionaries, each containing original segment and LaTeX.
    """
    all_results = []
    lines = text.splitlines()
    
    unique_conversions_map = {} # Key: (original_segment, latex_code), Value: line_number

    for line_idx, original_line_content in enumerate(lines):
        current_line_number = line_idx + 1
        if not original_line_content.strip():
            continue

        candidate_segments = generate_candidate_segments(original_line_content)

        for segment in candidate_segments:
            if not segment:
                continue

            latex_code = string_to_latex_aggressive(segment)
            if latex_code:
                conversion_key = (segment, latex_code)
                if conversion_key not in unique_conversions_map:
                    result_item = {
                        "original_line_number": current_line_number,
                        "original_segment": segment,
                        "latex": latex_code
                    }
                    all_results.append(result_item)
                    unique_conversions_map[conversion_key] = current_line_number
    
    all_results.sort(key=lambda x: (x["original_line_number"], -len(x["original_segment"])))
    return all_results

# --- Example Usage ---
if __name__ == "__main__":
    sample_text = """
The first equation is x = y + z / 2. This is inline.
Another important calculation is area = pi * r^2.
We also have force_total = G * m1 * m2 / d**2 according to Newton.
Consider the function f(t) = A * sin(omega * t + phi).
A simple sum: 5 + 3*2
The value can be found using result = (10.5 + 3.2) * 4 - 1.
This is a sentence without math. But it might have x y z variables.
The derivative df/dx can be approximated.
E = mc**2 is famous.
A test for functions: val = sqrt(b^2 - 4*a*c) / (2*a)
Also, check log(x) and exp(y-1) and ln(z).
The quick brown fox jumps over 1 lazy dog.
Distance = Speed * Time
Rate of Change = (Final Value - Initial Value) / Time
A simple variable x.
An expression: (alpha + beta)*(gamma - delta_val)
Another: item1 + item_2 / (item_3 - 5)
Test: Pressure = (n * R * Temperature) / Volume
Test for ^: x^3 + y^2
The formula is X = 1.
A text LHS = a + b
velocity = dx/dt
"""
    current_date_str = datetime.date.today().strftime("%Y-%m-%d")
    print(f"--- Identifying Mathematical Formulas (High Recall) --- (Run Date: {current_date_str})")
    
    identified_formulas = find_formulas_aggressive(sample_text)

    if identified_formulas:
        print(f"\nFound {len(identified_formulas)} potential formulas:")
        for item in identified_formulas:
            print(f"  Line {item['original_line_number']}:")
            print(f"    Original: \"{item['original_segment']}\"")
            print(f"    LaTeX:    $${item['latex']}$$")
    else:
        print("No potential formulas identified.")
import re
import sympy as sp
import numpy as np
from typing import Optional, Tuple, List


def tokenize(expr: str) -> List[str]:
    return re.findall(r'\d+|[()+\-*/]', expr)

def detokenize(x: List[str]) -> str:
    return ''.join(x)


def modular_arithmetic_with_brackets(
        *,
        batch_size: int = 1,
        vocab_size: int = 15,
        min_sequence_length: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        context_length: int = 20,
        seed: int = 42,
        mult: bool = True,
        **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    max_sequence_length = max_sequence_length or context_length
    min_sequence_length = min_sequence_length or max_sequence_length

    num_special_tokens = 7
    max_number = vocab_size - num_special_tokens
    if max_number <= 0:
        raise ValueError("vocab_size is too small to accommodate special tokens and numbers.")

    token_to_id = {'+': max_number, '-': max_number + 1, '*': max_number + 2,
                   '(': max_number + 3, ')': max_number + 4, '=': max_number + 5,
                   '[PAD]': max_number + 6, **{str(n): n for n in range(max_number)}}
    id_to_token = {v: k for k, v in token_to_id.items()}

    res = np.zeros([batch_size, context_length], dtype=np.int32)
    prediction_mask = np.zeros_like(res)

    for batch_idx in range(batch_size):
        seq_length = max(rng.integers(min_sequence_length, max_sequence_length + 1), 5)
        tokens, result = generate_one_expression_and_result(max_number,
                                                            seq_length - 3, mult, rng)
        tokens.extend(['=', str(result), '[PAD]'])

        token_ids = [token_to_id[token] for token in tokens]
        padded_sequence = np.pad(token_ids, (0, context_length - len(token_ids)),
                                 constant_values=token_to_id['[PAD]'])
        res[batch_idx] = padded_sequence[:context_length]

        eq_idx = token_ids.index(token_to_id['='])
        if eq_idx + 1 < context_length:
            prediction_mask[batch_idx, eq_idx + 1] = 1

    return res, prediction_mask


def generate_one_expression_and_result(
    modulus: int, length: int, mult: bool = True,
    rng: np.random.Generator = None,
) -> Tuple[List[str], int]:
    rng = rng or np.random.default_rng()

    if length < 1:
        raise ValueError(f"Can't generate expressions of length < 1. Got {length}.")

    if length < 5:
        return generate_short_expression(length, modulus, rng)

    left_length = rng.integers(1, length - 3)
    right_length = length - (left_length + 3)
    left_str, left_val = generate_one_expression_and_result(modulus, left_length, mult, rng)
    right_str, right_val = generate_one_expression_and_result(modulus, right_length, mult, rng)

    op = rng.integers(0, 3 if mult else 2)
    if op == 0:
        return tokenize(f'({left_str}+{right_str})'), (left_val + right_val) % modulus
    elif op == 1:
        return tokenize(f'({left_str}-{right_str})'), (left_val - right_val) % modulus
    else:
        return tokenize(f'({left_str}*{right_str})'), (left_val * right_val) % modulus


def generate_short_expression(length: int, modulus: int, rng: np.random.Generator) -> Tuple[List[str], int]:
    terminal = rng.integers(modulus)
    if length == 1:
        return [str(terminal)], terminal
    elif length == 2:
        return ['-', str(terminal)], -terminal % modulus
    elif length == 3:
        return ['(', str(terminal), ')'], terminal
    elif length == 4:
        return ['(', '-', str(terminal), ')'], -terminal % modulus

def test_modular_arithmetic_with_brackets():
    batch_size, vocab_size, context_length = 10, 12, 256
    res, prediction_mask = modular_arithmetic_with_brackets(
        batch_size=batch_size, vocab_size=vocab_size,
        context_length=context_length, min_sequence_length=3,
        max_sequence_length=40
    )

    num_special_tokens = 7
    max_number = vocab_size - num_special_tokens
    token_to_id = {'+': max_number, '-': max_number + 1, '*': max_number + 2,
                   '(': max_number + 3, ')': max_number + 4, '=': max_number + 5,
                   '[PAD]': max_number + 6, **{str(n): n for n in range(max_number)}}
    id_to_token = {v: k for k, v in token_to_id.items()}

    for batch_idx in range(batch_size):
        sequence = res[batch_idx]
        mask = prediction_mask[batch_idx]
        tokens = [id_to_token.get(token_id, '[UNK]') for token_id in sequence]

        try:
            pos_equal = np.where(sequence == token_to_id['='])[0][0]
            expr_str = ''.join(tokens[:pos_equal])
            result = sp.sympify(expr_str) % max_number
            expected_result = sequence[pos_equal + 1]

            print(f"Batch {batch_idx + 1}:")
            print(f"Expression: {expr_str}")
            print(f"Computed result (mod {max_number}): {result}")
            print(f"Expected result: {expected_result}")
            print("Test passed" if result == expected_result else "Test failed")
            print(f"Sequence tokens: {tokens}")
            print(f"Prediction mask: {mask}")
            print()
        except Exception as e:
            print(f"Error in batch {batch_idx + 1}:", e)
            print()


if __name__ == "__main__":
    test_modular_arithmetic_with_brackets()

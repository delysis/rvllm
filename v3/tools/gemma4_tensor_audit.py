#!/usr/bin/env python3
"""Streaming, offline tensor audit against an explicitly pinned external policy.

No inference, timer, queue, compiler, model download, or accelerator dependency.
A pass only means these complete files satisfy the supplied tensor policy. It
is not full-route, hardware, performance, or promotion evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import stat
import sys
from typing import Any

SCHEMA = 'rvllm.gemma4.tensor-audit.v1'
WIDTH = {'f16': 2, 'bf16': 2, 'f32': 4}
CHUNK_BYTES = 65536
MAX_JSON_BYTES = 1024 * 1024


class AuditError(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f'duplicate JSON member: {key}')
        result[key] = value
    return result


def parse_json(data: bytes):
    if len(data) > MAX_JSON_BYTES:
        raise AuditError('manifest/policy exceeds one MiB')
    def nonfinite(text):
        raise AuditError('nonfinite JSON token: ' + text)
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=nonfinite)


def open_regular(path: Path):
    # The compared file descriptor, not a pre-open pathname check, must name a
    # regular file. O_NONBLOCK prevents a FIFO from blocking the offline audit.
    # Parent-directory aliases remain valid (including macOS /tmp aliases).
    if path.is_symlink():
        raise AuditError('symlinked audit input')
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise AuditError('regular audit input required')
        handle = os.fdopen(fd, 'rb')
    except BaseException:
        os.close(fd)
        raise
    return handle


def read_json(path: Path):
    with open_regular(path) as source:
        return parse_json(source.read(MAX_JSON_BYTES + 1))


def digest_pin(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise AuditError('expected an explicit lowercase SHA-256 pin')
    return value


def resolve(root: Path, pin: dict) -> Path:
    if not isinstance(pin, dict):
        raise AuditError("input pin must be an object")
    path = pin.get('path')
    if not isinstance(path, str) or not path or '\0' in path:
        raise AuditError('expected a nonempty input path')
    # Read-only absolute paths are supported to avoid copying large captures.
    return root / path


def limit(value: Any) -> float:
    try:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise AuditError('numeric limits must be explicit, finite, and nonnegative')
        return float(value)
    except OverflowError as error:
        raise AuditError('numeric limit exceeds finite floating-point range') from error


def sample_count(shape: Any) -> int:
    if not isinstance(shape, list) or not shape or len(shape) > 8:
        raise AuditError('expected a nonempty shape with at most eight dimensions')
    count = 1
    for dim in shape:
        if type(dim) is not int or dim <= 0:
            raise AuditError('shape dimensions must be positive integers, not booleans')
        count *= dim
        if count > (2**63 - 1) // 4:
            raise AuditError('tensor byte-size overflow')
    return count


def floats(data: bytes, dtype: str):
    if dtype == 'bf16':
        return (struct.unpack('<f', struct.pack('<I', bits << 16))[0]
                for (bits,) in struct.iter_unpack('<H', data))
    return (value for (value,) in struct.iter_unpack('<e' if dtype == 'f16' else '<f', data))


def compare(pair: dict, policy: dict, root: Path) -> dict:
    if not isinstance(pair, dict) or not isinstance(policy, dict):
        raise AuditError('tensor pair and tolerance policy must be objects')
    if set(policy) != {'max_abs', 'relative_l2', 'bitwise'}:
        raise AuditError('tensor policy must specify exactly max_abs, relative_l2, bitwise')
    if set(pair) != {'name', 'shape', 'reference', 'candidate'}:
        raise AuditError('unexpected tensor pair fields')
    count = sample_count(pair.get('shape'))
    reference, candidate = pair['reference'], pair['candidate']
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        raise AuditError('tensor input pins must be objects')
    if any(set(pin) != {'path', 'dtype', 'sha256'} for pin in (reference, candidate)):
        raise AuditError('unexpected tensor input pin fields')
    rt, ct = reference.get('dtype'), candidate.get('dtype')
    if rt not in WIDTH or ct not in WIDTH:
        raise AuditError('only explicit little-endian f16, bf16, and f32 are supported')
    max_allowed = limit(policy['max_abs'])
    relative_allowed = limit(policy['relative_l2'])
    exact = policy['bitwise']
    if type(exact) is not bool or (exact and rt != ct):
        raise AuditError('bitwise policy requires a boolean and matching scalar types')
    expected_r = digest_pin(reference.get('sha256'))
    expected_c = digest_pin(candidate.get('sha256'))
    rp, cp = resolve(root, reference), resolve(root, candidate)
    rh, ch = hashlib.sha256(), hashlib.sha256()
    max_abs, worst, nonfinite, bit_mismatches = 0.0, None, 0, 0
    worst_reference, worst_candidate = None, None
    reference_squared, error_squared = 0.0, 0.0
    with open_regular(rp) as rf, open_regular(cp) as cf:
        for handle, dtype in ((rf, rt), (cf, ct)):
            if os.fstat(handle.fileno()).st_size != count * WIDTH[dtype]:
                raise AuditError(f'{pair["name"]}: file length differs from declared shape')
        index = 0
        chunk_samples = CHUNK_BYTES // max(WIDTH[rt], WIDTH[ct])
        while index < count:
            n = min(count - index, chunk_samples)
            rb, cb = rf.read(n * WIDTH[rt]), cf.read(n * WIDTH[ct])
            if len(rb) != n * WIDTH[rt] or len(cb) != n * WIDTH[ct]:
                raise AuditError('tensor truncated during streaming read')
            rh.update(rb)
            ch.update(cb)
            squared_r, squared_e = [], []
            for j, (r, c) in enumerate(zip(floats(rb, rt), floats(cb, ct))):
                if rt == ct and rb[j * WIDTH[rt]:(j + 1) * WIDTH[rt]] != cb[j * WIDTH[ct]:(j + 1) * WIDTH[ct]]:
                    bit_mismatches += 1
                if not (math.isfinite(r) and math.isfinite(c)):
                    nonfinite += 1
                    continue
                delta = c - r
                if abs(delta) > max_abs:
                    max_abs, worst = abs(delta), index + j
                    worst_reference, worst_candidate = r, c
                squared_r.append(r * r)
                squared_e.append(delta * delta)
            reference_squared = math.fsum((reference_squared, math.fsum(squared_r)))
            error_squared = math.fsum((error_squared, math.fsum(squared_e)))
            index += n
        if rf.read(1) or cf.read(1):
            raise AuditError('tensor grew during streaming read')
    relative = (math.sqrt(error_squared / reference_squared) if reference_squared
                else (0.0 if error_squared == 0.0 else None))
    coordinates = None
    if worst is not None:
        coordinate = worst
        coordinates = []
        for dim in reversed(pair['shape']):
            coordinates.append(coordinate % dim)
            coordinate //= dim
        coordinates.reverse()
    hash_r, hash_c = rh.hexdigest(), ch.hexdigest()
    passed = (hash_r == expected_r and hash_c == expected_c and nonfinite == 0
              and max_abs <= max_allowed and relative is not None
              and relative <= relative_allowed and (not exact or bit_mismatches == 0))
    return {
        'name': pair['name'], 'shape': pair['shape'], 'elements': count,
        'reference_dtype': rt, 'candidate_dtype': ct,
        'reference_sha256': hash_r, 'candidate_sha256': hash_c,
        'reference_hash_matches': hash_r == expected_r,
        'candidate_hash_matches': hash_c == expected_c,
        'max_abs': max_abs, 'worst_index': worst, 'relative_l2': relative,
        'worst_coordinates': coordinates, 'worst_reference': worst_reference,
        'worst_candidate': worst_candidate,
        'reference_l2_squared': reference_squared, 'error_l2_squared': error_squared,
        'bit_mismatches': bit_mismatches if rt == ct else None,
        'nonfinite_pairs': nonfinite, 'limits': policy, 'passed': passed,
    }


def run(manifest: dict, root: Path) -> dict:
    if not isinstance(manifest, dict) or set(manifest) != {'schema', 'policy', 'pairs'} or manifest.get('schema') != SCHEMA:
        raise AuditError('unrecognized tensor manifest schema')
    pairs = manifest.get('pairs')
    if not isinstance(pairs, list) or not pairs:
        raise AuditError('no tensor pairs: an empty oracle cannot pass')
    if any(not isinstance(pair, dict) for pair in pairs):
        raise AuditError('tensor pairs must be objects')
    names = [pair.get('name') for pair in pairs]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise AuditError('tensor names must be nonempty and unique')
    pin = manifest['policy']
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise AuditError('unexpected policy pin fields')
    with open_regular(resolve(root, pin)) as source:
        policy_bytes = source.read(MAX_JSON_BYTES + 1)
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    if policy_sha != digest_pin(pin.get('sha256')):
        raise AuditError('external tolerance policy does not match its pin')
    policy = parse_json(policy_bytes)
    if not isinstance(policy, dict) or set(policy) != set(names):
        raise AuditError('every policy tensor must be tested, without omitted or extra names')
    results = [compare(pair, policy[pair['name']], root) for pair in pairs]
    return {'schema': SCHEMA, 'policy_sha256': policy_sha, 'pairs': results,
            'supplied_tensor_policy_passed': all(r['passed'] for r in results),
            'hardware_qualification': False, 'performance_acceptance': False,
            'promotion_authorized': False}


def write_new(path: Path, result: dict) -> None:
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('new_output', type=Path)
    args = parser.parse_args()
    try:
        with open_regular(args.manifest) as source:
            manifest_bytes = source.read(MAX_JSON_BYTES + 1)
        result = run(parse_json(manifest_bytes), args.manifest.resolve().parent)
        result['manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        result = {'schema': SCHEMA, 'supplied_tensor_policy_passed': False,
                  'hardware_qualification': False, 'performance_acceptance': False,
                  'promotion_authorized': False, 'error': str(error)}
    try:
        write_new(args.new_output, result)
    except OSError as error:
        print(f'cannot preserve new output: {error}', file=sys.stderr)
        return 2
    return 0 if result['supplied_tensor_policy_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

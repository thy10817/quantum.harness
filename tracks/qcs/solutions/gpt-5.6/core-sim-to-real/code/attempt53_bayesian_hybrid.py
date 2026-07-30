#!/usr/bin/env python3
"""Attempt 53: noisy BO and BO-to-local calibration in the rank-15 space.

The online methods receive only a counted finite-shot scalar fidelity service.
Exact fidelities are evaluated only after the service closes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import ndtr, ndtri
from scipy.stats import norm, qmc

import attempt44_dimension_cost as base
from phase3_common import TRUTH_FAMILIES, build_nominal_model, scan_widths


HERE = Path(__file__).resolve().parent
CORE = HERE.parent
CONFIG_PATH = HERE / "attempt53_bayesian_hybrid_config.json"
PROTOCOL_PATH = CORE / "docs" / "ATTEMPT53_PROTOCOL.md"
OUTPUT_PATH = CORE / "results_summary" / "QL1F-attempt53-bayesian-hybrid.exploratory.json"
SMOKE_PATH = CORE / "results_summary" / "QL1F-attempt53-bayesian-hybrid.smoke.json"
REPORT_PATH = CORE / "docs" / "ATTEMPT53_REPORT.md"
PLOT_PATH = CORE / "plots" / "attempt53-bayesian-hybrid.png"

METHODS = (
    "principal-global-1cycle-k15",
    "principal-global-2cycle-k15",
    "sequential-quadratic-k15",
    "bayes-32-k15",
    "bayes-64-k15",
    "bayes16-then-global-k15",
    "bayes32-then-global-k15",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=CORE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_config(config: dict[str, Any]) -> None:
    if config["methods"] != list(METHODS):
        raise RuntimeError("method order changed")
    budget = config["budget"]
    if (
        int(budget["decision_shots"]) != 32768
        or int(budget["sentinel_queries"]) != 2
        or int(budget["sentinel_shots"]) != 1024
    ):
        raise RuntimeError("shared query settings changed")
    expected_budgets = {
        "principal-global-1cycle-k15": (32, 34, 1_050_624),
        "principal-global-2cycle-k15": (64, 66, 2_099_200),
        "sequential-quadratic-k15": (64, 66, 2_099_200),
        "bayes-32-k15": (32, 34, 1_050_624),
        "bayes-64-k15": (64, 66, 2_099_200),
        "bayes16-then-global-k15": (48, 50, 1_574_912),
        "bayes32-then-global-k15": (64, 66, 2_099_200),
    }
    for method, expected in expected_budgets.items():
        record = budget["by_method"][method]
        actual = (
            int(record["decision_queries"]),
            int(record["total_queries"]),
            int(record["total_shots"]),
        )
        if actual != expected:
            raise RuntimeError((method, actual, expected))
    bo = config["bo"]
    counts = {key: int(value) for key, value in bo["observation_counts"].items()}
    if int(bo["initial_observations"]) != 12:
        raise RuntimeError("BO initial design changed")
    if counts != {
        "bayes-32-k15": 30,
        "bayes-64-k15": 62,
        "bayes16-then-global-k15": 14,
        "bayes32-then-global-k15": 30,
    }:
        raise RuntimeError("BO budget decomposition changed")
    if int(config["sequential"]["points_per_direction"]) * 15 + 4 != 64:
        raise RuntimeError("sequential budget decomposition changed")


def paired_seed(family_index: int, truth_seed: int, replicate: int) -> int:
    sequence = np.random.SeedSequence(
        [113, 53, int(family_index), int(truth_seed), int(replicate)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def make_attempt53_truth(
    model: Any,
    family: str,
    epsilon: float,
    seed: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extend the frozen truth recipe only to Attempt-53 preregistered seeds."""

    if family not in config["benchmark"]["families"]:
        raise ValueError(f"family {family!r} is not preregistered")
    expected_epsilon = float(
        config["benchmark"]["fixed_epsilon_by_family"][family]
    )
    if float(epsilon) != expected_epsilon:
        raise ValueError(f"epsilon {epsilon!r} is not preregistered")
    if int(seed) not in [int(value) for value in config["benchmark"]["truth_seeds"]]:
        raise ValueError(f"seed {seed!r} is not preregistered")

    seed_sequence = np.random.SeedSequence([113, 11, int(seed)])
    map_ss, drift_ss = seed_sequence.spawn(2)
    map_rng = np.random.default_rng(map_ss)
    drift_rng = np.random.default_rng(drift_ss)
    nominal_drift = np.asarray(model.h_drift, dtype=np.complex128)
    nominal_controls = np.asarray(model.h_controls, dtype=np.complex128)
    control_count = nominal_controls.shape[0]
    dimension = nominal_drift.shape[0]

    mismatch = map_rng.normal(size=(control_count, control_count))
    mismatch /= np.linalg.norm(mismatch, ord=2)
    control_map = np.eye(control_count) + float(epsilon) * mismatch

    raw = drift_rng.normal(size=(dimension, dimension)) + 1j * drift_rng.normal(
        size=(dimension, dimension)
    )
    drift_direction = (raw + raw.conj().T) / 2
    drift_direction -= (
        np.trace(drift_direction) * np.eye(dimension) / dimension
    )
    drift_direction /= np.linalg.norm(drift_direction, ord="fro")
    drift_scale = np.linalg.norm(nominal_drift, ord="fro")

    true_drift = np.array(nominal_drift, copy=True)
    true_controls = np.array(nominal_controls, copy=True)
    if family in ("drift", "combined"):
        true_drift += float(epsilon) * drift_scale * drift_direction
    if family in ("control-map", "combined"):
        true_controls = np.einsum(
            "ij,jab->iab", control_map, nominal_controls
        )
    metadata = {
        "family": family,
        "epsilon": float(epsilon),
        "seed": int(seed),
        "paired_seed_sequence": [113, 11, int(seed)],
        "control_map_minus_identity_spectral_norm": float(
            np.linalg.norm(control_map - np.eye(control_count), ord=2)
        ),
        "relative_drift_frobenius_norm": float(
            np.linalg.norm(true_drift - nominal_drift, ord="fro")
            / drift_scale
        ),
        "attempt53_local_extension_of_frozen_truth_recipe": True,
    }
    return true_drift, true_controls, metadata


def algorithm_seed(noise_seed: int) -> int:
    # All BO variants deliberately share one proposal seed, so their query
    # histories are identical until the shorter method branches or stops.
    sequence = np.random.SeedSequence([113, 53, int(noise_seed), 8675309])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def project_ball(points: np.ndarray, radius: float) -> np.ndarray:
    output = np.asarray(points, dtype=np.float64).copy()
    norms = np.linalg.norm(output, axis=1)
    outside = norms > radius
    if np.any(outside):
        output[outside] *= (radius / norms[outside])[:, None]
    return output


def sobol_ball(count: int, dimension: int, radius: float, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, dimension), dtype=np.float64)
    power = int(math.ceil(math.log2(max(1, count))))
    sampler = qmc.Sobol(d=dimension + 1, scramble=True, seed=int(seed))
    unit = sampler.random_base2(power)[:count]
    normals = ndtri(np.clip(unit[:, :dimension], 1e-12, 1.0 - 1e-12))
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    directions = normals / np.maximum(norms, 1e-15)
    # Uniform radius, rather than volume, deliberately allocates points near
    # the local nominal solution in this high-dimensional ball.
    radii = radius * unit[:, dimension]
    return directions * radii[:, None]


def matern52(
    left: np.ndarray, right: np.ndarray, lengthscales: np.ndarray
) -> np.ndarray:
    delta = (
        left[:, None, :] - right[None, :, :]
    ) / lengthscales[None, None, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    scaled = math.sqrt(5.0) * distance
    return (1.0 + scaled + scaled * scaled / 3.0) * np.exp(-scaled)


def gp_posterior(
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    observed_noise: np.ndarray,
    candidates: np.ndarray,
    lengthscales: np.ndarray,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(observed_y))
    centered = observed_y - mean
    amplitude = max(float(np.var(observed_y)), 1e-8)
    kernel = amplitude * matern52(observed_x, observed_x, lengthscales)
    diagonal = np.maximum(observed_noise, 1e-12)
    last_error: Exception | None = None
    for factor in (1.0, 10.0, 100.0, 1000.0):
        try:
            chol = np.linalg.cholesky(
                kernel + np.diag(diagonal + jitter * factor)
            )
            break
        except np.linalg.LinAlgError as error:
            last_error = error
    else:
        raise np.linalg.LinAlgError("GP Cholesky failed") from last_error
    alpha = cho_solve((chol, True), centered)
    cross = amplitude * matern52(observed_x, candidates, lengthscales)
    posterior_mean = mean + cross.T @ alpha
    solved = solve_triangular(chol, cross, lower=True)
    posterior_variance = np.maximum(
        amplitude - np.sum(solved * solved, axis=0), 1e-15
    )
    return posterior_mean, posterior_variance


def expected_improvement(
    mean: np.ndarray, variance: np.ndarray, best: float
) -> np.ndarray:
    sigma = np.sqrt(np.maximum(variance, 1e-15))
    improvement = best - mean
    standardized = improvement / sigma
    return improvement * ndtr(standardized) + sigma * np.exp(
        -0.5 * standardized * standardized
    ) / math.sqrt(2.0 * math.pi)


def candidate_pool(
    observed: np.ndarray,
    incumbent: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    bo = config["bo"]
    dimension = observed.shape[1]
    radius = float(bo["domain_radius"])
    global_points = sobol_ball(
        int(bo["candidate_pool_global"]),
        dimension,
        radius,
        int(rng.integers(0, 2**31 - 1)),
    )
    local = incumbent[None, :] + float(bo["local_candidate_scale"]) * rng.normal(
        size=(int(bo["candidate_pool_local"]), dimension)
    )
    local = project_ball(local, radius)
    pool = np.vstack([global_points, local])
    distances = np.linalg.norm(
        pool[:, None, :] - observed[None, :, :], axis=2
    )
    keep = np.min(distances, axis=1) > 1e-10
    if not np.any(keep):
        raise RuntimeError("BO candidate pool collapsed to duplicates")
    return pool[keep]


def observation_noise(sampled_fidelity: float, shots: int) -> float:
    value = float(np.clip(sampled_fidelity, 0.0, 1.0))
    return max(value * (1.0 - value) / int(shots), 0.25 / int(shots) ** 2)


def bo_observations(
    client: base.LedgerClient,
    start: np.ndarray,
    basis: np.ndarray,
    curvatures: np.ndarray,
    common_ridge: float,
    total: int,
    initial: int,
    config: dict[str, Any],
    rng: np.random.Generator,
    label: str,
) -> dict[str, Any]:
    dimension = basis.shape[0]
    radius = float(config["bo"]["domain_radius"])
    initial_points = np.vstack(
        [
            np.zeros((1, dimension), dtype=np.float64),
            sobol_ball(
                initial - 1,
                dimension,
                radius,
                int(rng.integers(0, 2**31 - 1)),
            ),
        ]
    )
    denominator = np.asarray(curvatures, dtype=np.float64) + float(common_ridge)
    reference = float(np.median(denominator))
    lengthscales = float(config["bo"]["lengthscale_reference"]) * np.sqrt(
        reference / denominator
    )
    low, high = [float(value) for value in config["bo"]["lengthscale_clip"]]
    lengthscales = np.clip(lengthscales, low, high)
    shots = int(config["budget"]["decision_shots"])
    observed_x: list[np.ndarray] = []
    observed_y: list[float] = []
    observed_noise: list[float] = []

    def observe(coordinates: np.ndarray, purpose: str) -> None:
        parameters = start + coordinates @ basis
        sampled = client.query(parameters, shots, purpose)
        observed_x.append(np.asarray(coordinates, dtype=np.float64).copy())
        observed_y.append(1.0 - sampled)
        observed_noise.append(observation_noise(sampled, shots))

    for index, point in enumerate(initial_points):
        observe(point, f"{label}-initial-{index}")

    while len(observed_x) < total:
        x = np.asarray(observed_x)
        y = np.asarray(observed_y)
        noise = np.asarray(observed_noise)
        posterior_at_observed, _ = gp_posterior(
            x,
            y,
            noise,
            x,
            lengthscales,
            float(config["bo"]["gp_jitter"]),
        )
        incumbent = x[int(np.argmin(posterior_at_observed))]
        pool = candidate_pool(x, incumbent, config, rng)
        mean, variance = gp_posterior(
            x,
            y,
            noise,
            pool,
            lengthscales,
            float(config["bo"]["gp_jitter"]),
        )
        acquisition = expected_improvement(
            mean, variance, float(np.min(posterior_at_observed))
        )
        chosen = pool[int(np.argmax(acquisition))]
        observe(chosen, f"{label}-adaptive-{len(observed_x) - initial}")

    x = np.asarray(observed_x)
    y = np.asarray(observed_y)
    noise = np.asarray(observed_noise)
    posterior_mean, posterior_variance = gp_posterior(
        x,
        y,
        noise,
        x,
        lengthscales,
        float(config["bo"]["gp_jitter"]),
    )
    recommendation_index = int(np.argmin(posterior_mean))
    return {
        "coordinates": x,
        "sampled_infidelities": y,
        "observation_variances": noise,
        "posterior_mean_at_observed": posterior_mean,
        "posterior_variance_at_observed": posterior_variance,
        "recommendation_index": recommendation_index,
        "recommendation": x[recommendation_index].copy(),
        "lengthscales": lengthscales,
    }


def validate_pair(
    client: base.LedgerClient,
    current: np.ndarray,
    proposal: np.ndarray,
    shots: int,
    confidence: float,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    sampled_current = client.query(current, shots, f"{label}-current")
    sampled_proposal = client.query(proposal, shots, f"{label}-proposal")
    standard_error = math.sqrt(
        (
            sampled_current * (1.0 - sampled_current)
            + sampled_proposal * (1.0 - sampled_proposal)
        )
        / shots
    )
    improvement = sampled_proposal - sampled_current
    margin = improvement - float(norm.ppf(confidence)) * standard_error
    accepted = bool(margin > 0.0)
    return (proposal.copy() if accepted else current.copy()), {
        "accepted": accepted,
        "sampled_current_fidelity": sampled_current,
        "sampled_proposal_fidelity": sampled_proposal,
        "measured_improvement": improvement,
        "standard_error": standard_error,
        "confidence_margin": margin,
        "step_norm": float(np.linalg.norm(proposal - current)),
    }


def one_local_cycle(
    client: base.LedgerClient,
    current: np.ndarray,
    basis: np.ndarray,
    curvatures: np.ndarray,
    common_ridge: float,
    config: dict[str, Any],
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    shots = int(config["budget"]["decision_shots"])
    delta = 0.05
    derivatives = []
    variances = []
    for index, direction in enumerate(basis):
        plus = client.query(
            current + delta * direction, shots, f"{label}-gradient-{index}-plus"
        )
        minus = client.query(
            current - delta * direction, shots, f"{label}-gradient-{index}-minus"
        )
        derivatives.append(((1.0 - plus) - (1.0 - minus)) / (2.0 * delta))
        variances.append(
            (plus * (1.0 - plus) + minus * (1.0 - minus))
            / (4.0 * delta * delta * shots)
        )
    gradient = np.asarray(derivatives)
    covariance = np.asarray(variances)
    step = base.bounded_step(gradient, curvatures, common_ridge, 0.25)
    predicted = float(
        -gradient @ step - 0.5 * np.sum(curvatures * step * step)
    )
    predicted_se = float(np.sqrt(np.sum(step * step * covariance)))
    model_margin = predicted - float(norm.ppf(0.995)) * predicted_se
    proposal = current + step @ basis if model_margin > 0.0 else current.copy()
    accepted, validation = validate_pair(
        client, current, proposal, shots, 0.995, f"{label}-validation"
    )
    validation.update(
        {
            "model_pass": bool(model_margin > 0.0),
            "predicted_improvement": predicted,
            "predicted_standard_error": predicted_se,
            "model_margin": model_margin,
            "gradient_sha256": base.array_sha256(gradient),
            "step_sha256": base.array_sha256(step),
        }
    )
    return accepted, validation


def run_principal_global(
    client: base.LedgerClient,
    start: np.ndarray,
    basis: np.ndarray,
    curvatures: np.ndarray,
    common_ridge: float,
    attempt44_config: dict[str, Any],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    constants = base.frozen_constants(attempt44_config)
    scan = base.global_calibration(
        client, start, basis, curvatures, common_ridge, constants
    )
    accepted = [
        np.asarray(value, dtype=np.float64)
        for value in scan.pop("accepted_parameter_vectors_posthoc_only")
    ]
    ledger = scan.pop("query_ledger")
    return accepted, {
        "algorithm": "attempt49-principal-global-k15-2cycle",
        "accepted_count": len(accepted),
        "decisions": [
            {
                "cycle": int(row["cycle"]),
                "accepted": bool(row["accepted"]),
                "model_pass": bool(row["model_pass"]),
                "validation_pass": bool(row["validation_pass"]),
                "step_norm": float(row["step_norm"]),
            }
            for row in scan["decisions"]
        ],
        "ledger": ledger,
    }


def run_principal_one_cycle(
    client: base.LedgerClient,
    start: np.ndarray,
    basis: np.ndarray,
    curvatures: np.ndarray,
    common_ridge: float,
    config: dict[str, Any],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    sentinel = int(config["budget"]["sentinel_shots"])
    client.query(start, sentinel, "initial-sentinel")
    final, validation = one_local_cycle(
        client,
        start,
        basis,
        curvatures,
        common_ridge,
        config,
        "one-cycle-local",
    )
    client.query(final, sentinel, "final-sentinel")
    service = client.end()
    return [start.copy(), final.copy()], {
        "algorithm": "attempt49-principal-global-k15-1cycle",
        "accepted_count": 2,
        "decisions": [validation],
        "service": service,
        "ledger": client.ledger,
    }


def run_sequential(
    client: base.LedgerClient,
    start: np.ndarray,
    basis: np.ndarray,
    curvatures: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    shots = int(config["budget"]["decision_shots"])
    sentinel = int(config["budget"]["sentinel_shots"])
    confidence = float(config["semantics"]["online_acceptance_confidence"])
    client.query(start, sentinel, "initial-sentinel")
    start_samples = [
        client.query(start, shots, "sequential-start-0"),
        client.query(start, shots, "sequential-start-1"),
    ]
    current = start.copy()
    widths = scan_widths(curvatures)
    normalized = np.asarray(config["sequential"]["scan_offsets"], dtype=np.float64)
    decisions = []
    for index, direction in enumerate(basis):
        offsets = widths[index] * normalized
        infidelities = np.asarray(
            [
                1.0
                - client.query(
                    current + offset * direction,
                    shots,
                    f"sequential-direction-{index}-point-{point_index}",
                )
                for point_index, offset in enumerate(offsets)
            ]
        )
        design = np.column_stack([offsets**2, offsets, np.ones_like(offsets)])
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, infidelities, rcond=None
        )
        if (
            rank == 3
            and np.all(np.isfinite(coefficients))
            and coefficients[0] > 0.0
        ):
            chosen = float(
                np.clip(
                    -coefficients[1] / (2.0 * coefficients[0]),
                    -widths[index],
                    widths[index],
                )
            )
            rule = "quadratic-fit"
        else:
            chosen = float(offsets[int(np.argmin(infidelities))])
            rule = "best-measured"
        current = current + chosen * direction
        decisions.append(
            {
                "direction": index,
                "chosen_offset": chosen,
                "rule": rule,
            }
        )
    final_samples = [
        client.query(current, shots, "sequential-final-0"),
        client.query(current, shots, "sequential-final-1"),
    ]
    initial_mean = float(np.mean(start_samples))
    final_mean = float(np.mean(final_samples))
    standard_error = math.sqrt(
        (
            initial_mean * (1.0 - initial_mean)
            + final_mean * (1.0 - final_mean)
        )
        / (2 * shots)
    )
    margin = (
        final_mean
        - initial_mean
        - float(norm.ppf(confidence)) * standard_error
    )
    accepted = bool(margin > 0.0)
    final = current.copy() if accepted else start.copy()
    client.query(final, sentinel, "final-sentinel")
    service = client.end()
    return [start.copy(), final], {
        "algorithm": "budget-matched-sequential-quadratic-proxy",
        "accepted_count": 2,
        "final_accepted": accepted,
        "aggregate_validation_margin": margin,
        "decisions": decisions,
        "service": service,
        "ledger": client.ledger,
    }


def run_bayes(
    client: base.LedgerClient,
    start: np.ndarray,
    basis: np.ndarray,
    curvatures: np.ndarray,
    common_ridge: float,
    config: dict[str, Any],
    rng: np.random.Generator,
    method: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    shots = int(config["budget"]["decision_shots"])
    sentinel = int(config["budget"]["sentinel_shots"])
    confidence = float(config["semantics"]["online_acceptance_confidence"])
    client.query(start, sentinel, "initial-sentinel")
    hybrid = "then-global" in method
    prefix = "shared-bo"
    total_observations = int(config["bo"]["observation_counts"][method])
    observations = bo_observations(
        client,
        start,
        basis,
        curvatures,
        common_ridge,
        total_observations,
        int(config["bo"]["initial_observations"]),
        config,
        rng,
        prefix,
    )
    proposal = start + observations["recommendation"] @ basis
    incumbent, bo_validation = validate_pair(
        client, start, proposal, shots, confidence, f"{prefix}-validation"
    )
    accepted = [start.copy(), incumbent.copy()]
    local_validation = None
    if hybrid:
        incumbent, local_validation = one_local_cycle(
            client,
            incumbent,
            basis,
            curvatures,
            common_ridge,
            config,
            "hybrid-local",
        )
        accepted.append(incumbent.copy())
    client.query(incumbent, sentinel, "final-sentinel")
    service = client.end()
    return accepted, {
        "algorithm": (
            f"fixed-GP-BO-{total_observations}-then-one-principal-global-cycle"
            if hybrid
            else f"fixed-GP-BO-{total_observations}"
        ),
        "accepted_count": len(accepted),
        "bo_validation": bo_validation,
        "local_validation": local_validation,
        "bo": {
            "observation_count": int(observations["coordinates"].shape[0]),
            "recommendation_index": int(observations["recommendation_index"]),
            "recommendation_norm": float(
                np.linalg.norm(observations["recommendation"])
            ),
            "lengthscales": observations["lengthscales"].tolist(),
            "coordinates_sha256": base.array_sha256(observations["coordinates"]),
            "sampled_infidelities_sha256": base.array_sha256(
                observations["sampled_infidelities"]
            ),
        },
        "service": service,
        "ledger": client.ledger,
    }


def compact_online(online: dict[str, Any]) -> dict[str, Any]:
    ledger = online.pop("ledger")
    purpose_counts: dict[str, int] = defaultdict(int)
    purpose_shots: dict[str, int] = defaultdict(int)
    for row in ledger:
        purpose_counts[str(row["purpose"])] += 1
        purpose_shots[str(row["purpose"])] += int(row["shots"])
    online["ledger_closure"] = {
        "row_count": len(ledger),
        "total_shots": sum(int(row["shots"]) for row in ledger),
        "canonical_json_sha256": base.canonical_json_sha256(ledger),
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "purpose_shots": dict(sorted(purpose_shots.items())),
        "exact_fidelity_stored": False,
    }
    return online


def run_one(
    model: Any,
    family: str,
    epsilon: float,
    truth_seed: int,
    replicate: int,
    method: str,
    basis: np.ndarray,
    curvatures: np.ndarray,
    common_ridge: float,
    config: dict[str, Any],
    attempt44_config: dict[str, Any],
) -> dict[str, Any]:
    family_index = TRUTH_FAMILIES.index(family)
    drift, controls, truth_metadata = make_attempt53_truth(
        model, family, epsilon, truth_seed, config
    )

    def exact_evaluator(parameters: Any) -> float:
        return float(
            np.asarray(
                model.average_fidelity(
                    jnp.asarray(parameters),
                    jnp.asarray(drift),
                    jnp.asarray(controls),
                )
            )
        )

    start = np.asarray(model.optimized_parameters, dtype=np.float64)
    noise_seed = paired_seed(family_index, truth_seed, replicate)
    client = base.LedgerClient(exact_evaluator)
    client.start(noise_seed)
    if method == "principal-global-1cycle-k15":
        accepted, online = run_principal_one_cycle(
            client,
            start,
            basis,
            curvatures,
            common_ridge,
            config,
        )
    elif method == "principal-global-2cycle-k15":
        accepted, online = run_principal_global(
            client,
            start,
            basis,
            curvatures,
            common_ridge,
            attempt44_config,
        )
    elif method == "sequential-quadratic-k15":
        accepted, online = run_sequential(
            client, start, basis, curvatures, config
        )
    elif method in (
        "bayes-32-k15",
        "bayes-64-k15",
        "bayes16-then-global-k15",
        "bayes32-then-global-k15",
    ):
        rng = np.random.default_rng(algorithm_seed(noise_seed))
        accepted, online = run_bayes(
            client,
            start,
            basis,
            curvatures,
            common_ridge,
            config,
            rng,
            method,
        )
    else:
        raise ValueError(method)
    if client.active:
        raise AssertionError(f"{method} left the client active")
    ledger = online["ledger"]
    query_count = len(ledger)
    total_shots = sum(int(row["shots"]) for row in ledger)
    method_budget = config["budget"]["by_method"][method]
    if query_count != int(method_budget["total_queries"]):
        raise AssertionError((method, query_count, method_budget))
    if total_shots != int(method_budget["total_shots"]):
        raise AssertionError((method, total_shots, method_budget))

    # Hidden truth begins only after the service is closed.
    accepted_infidelities = [
        1.0 - exact_evaluator(vector) for vector in accepted
    ]
    target = float(config["semantics"]["target_infidelity"])
    return {
        "family": family,
        "epsilon": epsilon,
        "truth_seed": truth_seed,
        "replicate": replicate,
        "method": method,
        "noise_seed": noise_seed,
        "truth_metadata": truth_metadata,
        "online": compact_online(online),
        "posthoc": {
            "accepted_infidelities": accepted_infidelities,
            "best_accepted_infidelity": float(min(accepted_infidelities)),
            "final_accepted_infidelity": float(accepted_infidelities[-1]),
            "terminal_success": bool(accepted_infidelities[-1] <= target),
            "best_accepted_success_supplementary": bool(
                min(accepted_infidelities) <= target
            ),
            "exact_values_used_online": False,
        },
    }


def truth_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[
            (str(run["family"]), int(run["truth_seed"]), str(run["method"]))
        ].append(run)
    rows = []
    for (family, truth_seed, method), values in sorted(grouped.items()):
        rows.append(
            {
                "family": family,
                "truth_seed": truth_seed,
                "method": method,
                "success": float(
                    np.mean(
                        [row["posthoc"]["terminal_success"] for row in values]
                    )
                ),
                "log10_best_infidelity": float(
                    np.mean(
                        [
                            math.log10(
                                max(
                                    row["posthoc"]["best_accepted_infidelity"],
                                    1e-16,
                                )
                            )
                            for row in values
                        ]
                    )
                ),
                "log10_terminal_infidelity": float(
                    np.mean(
                        [
                            math.log10(
                                max(
                                    row["posthoc"][
                                        "final_accepted_infidelity"
                                    ],
                                    1e-16,
                                )
                            )
                            for row in values
                        ]
                    )
                ),
                "replicates": len(values),
            }
        )
    return rows


def bootstrap_summary(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    draws = int(config["bootstrap"]["draws"])
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    present = {str(row["family"]) for row in rows}
    families = [
        family
        for family in config["benchmark"]["families"]
        if family in present
    ]
    by_method_family: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {
            family: [
                row
                for row in rows
                if row["method"] == method and row["family"] == family
            ]
            for family in families
        }
        for method in METHODS
    }
    indices = {
        family: rng.integers(
            0,
            len(by_method_family[METHODS[0]][family]),
            size=(draws, len(by_method_family[METHODS[0]][family])),
        )
        for family in families
    }

    def draws_for(method: str, key: str) -> np.ndarray:
        family_draws = []
        for family in families:
            values = np.asarray(
                [row[key] for row in by_method_family[method][family]],
                dtype=np.float64,
            )
            family_draws.append(np.mean(values[indices[family]], axis=1))
        return np.mean(np.column_stack(family_draws), axis=1)

    method_draws = {
        method: {
            "success": draws_for(method, "success"),
            "log10_best_infidelity": draws_for(
                method, "log10_best_infidelity"
            ),
            "log10_terminal_infidelity": draws_for(
                method, "log10_terminal_infidelity"
            ),
        }
        for method in METHODS
    }

    def interval(values: np.ndarray, estimate: float) -> dict[str, float]:
        return {
            "estimate": float(estimate),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }

    methods = {}
    for method in METHODS:
        success_values = [
            row["success"] for row in rows if row["method"] == method
        ]
        log_values = [
            row["log10_terminal_infidelity"]
            for row in rows
            if row["method"] == method
        ]
        methods[method] = {
            "success": interval(
                method_draws[method]["success"], float(np.mean(success_values))
            ),
            "log10_terminal_infidelity": interval(
                method_draws[method]["log10_terminal_infidelity"],
                float(np.mean(log_values)),
            ),
        }
    baseline = "principal-global-2cycle-k15"
    differences = {}
    for method in METHODS:
        if method == baseline:
            continue
        values = (
            method_draws[method]["success"]
            - method_draws[baseline]["success"]
        )
        paired = []
        lookup = {
            (row["family"], row["truth_seed"], row["method"]): row
            for row in rows
        }
        for family in families:
            truth_seeds = sorted(
                {
                    int(row["truth_seed"])
                    for row in rows
                    if row["family"] == family
                    and row["method"] == baseline
                }
            )
            for truth_seed in truth_seeds:
                paired.append(
                    lookup[(family, truth_seed, method)]["success"]
                    - lookup[(family, truth_seed, baseline)]["success"]
                )
        differences[method] = interval(values, float(np.mean(paired)))
    return {
        "methods": methods,
        "paired_success_difference_vs_principal_global": differences,
        "truth_cell_count_per_method": len(rows) // len(METHODS),
        "bootstrap_draws": draws,
        "independent_unit": "truth-cell",
    }


def make_plot(summary: dict[str, Any]) -> None:
    labels = [
        "local\n1 cycle",
        "local\n2 cycles",
        "sequential\nquadratic",
        "BO\n32",
        "BO\n64",
        "BO16 →\nlocal",
        "BO32 →\nlocal",
    ]
    estimates = [
        summary["methods"][method]["success"]["estimate"] for method in METHODS
    ]
    lower = [
        estimate - summary["methods"][method]["success"]["lower_95"]
        for method, estimate in zip(METHODS, estimates)
    ]
    upper = [
        summary["methods"][method]["success"]["upper_95"] - estimate
        for method, estimate in zip(METHODS, estimates)
    ]
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.bar(
        np.arange(len(METHODS)),
        estimates,
        color=[
            "#14b8a6",
            "#0f766e",
            "#64748b",
            "#f59e0b",
            "#d97706",
            "#8b5cf6",
            "#7c3aed",
        ],
    )
    axis.errorbar(
        np.arange(len(METHODS)),
        estimates,
        yerr=np.asarray([lower, upper]),
        fmt="none",
        ecolor="#111827",
        capsize=4,
    )
    axis.set_xticks(np.arange(len(METHODS)), labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Post-hoc target success")
    axis.set_title("Attempt 53 · terminal success under frozen budgets")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def write_report(result: dict[str, Any]) -> None:
    summary = result["summary"]
    rows = []
    for method in METHODS:
        record = summary["methods"][method]
        success = record["success"]
        log_value = record["log10_terminal_infidelity"]
        budget = result["config"]["budget"]["by_method"][method]
        rows.append(
            f"| `{method}` | {100 * success['estimate']:.2f}% "
            f"[{100 * success['lower_95']:.2f}%, "
            f"{100 * success['upper_95']:.2f}%] | "
            f"{log_value['estimate']:.3f} "
            f"[{log_value['lower_95']:.3f}, "
            f"{log_value['upper_95']:.3f}] | "
            f"{budget['total_queries']} | {budget['total_shots']:,} |"
        )
    differences = []
    for method, record in summary[
        "paired_success_difference_vs_principal_global"
    ].items():
        differences.append(
            f"| `{method}` | {100 * record['estimate']:+.2f} pp "
            f"[{100 * record['lower_95']:+.2f}, "
            f"{100 * record['upper_95']:+.2f}] |"
        )
    text = f"""# Attempt 53 report — Bayesian and hybrid rank-15 calibration

Evidence level: **exploratory development**, not holdout confirmation.

All equal-budget comparisons use identical online resources; reduced variants
test whether success can be retained with fewer queries and shots. The
optimizer never received exact truth fidelity, and the primary score uses only
the online-selected terminal output.

## Results

| Method | Terminal target success (95% truth-cell bootstrap) | Mean log10 terminal infidelity | Queries | Shots |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Paired success difference versus full two-cycle principal-global method

| Method | Difference (percentage points, 95% interval) |
|---|---:|
{chr(10).join(differences)}

## Interpretation rule

An equal-budget Bayesian method is a candidate for follow-up when it improves
paired success. A reduced-budget method is useful only if it remains
non-inferior to the full two-cycle local baseline. This experiment is not
sufficient for a new submission claim: any selected method must be frozen and
rerun on disjoint truth seeds.

The sequential method is a budget-matched four-point line-scan proxy, not a
literal reproduction of the paper's experimental scan allocation.

![Equal-budget success](../plots/{PLOT_PATH.name})
"""
    atomic_text(REPORT_PATH, text)


def execute(smoke: bool) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    validate_config(config)
    attempt44_config = read_json(HERE / "attempt44_dimension_cost_config.json")
    model = build_nominal_model()
    geometries, geometry_audit, common_ridge = base.build_search_geometries(
        model, attempt44_config
    )
    basis, curvatures = geometries["model-informed-k15"]
    families = (
        [config["benchmark"]["families"][0]]
        if smoke
        else list(config["benchmark"]["families"])
    )
    seeds = (
        [config["benchmark"]["truth_seeds"][0]]
        if smoke
        else list(config["benchmark"]["truth_seeds"])
    )
    replicates = 1 if smoke else int(
        config["benchmark"]["replicates_per_truth_cell"]
    )
    runs = []
    total = len(families) * len(seeds) * replicates * len(METHODS)
    completed = 0
    for family in families:
        epsilon = float(
            config["benchmark"]["fixed_epsilon_by_family"][family]
        )
        for truth_seed in seeds:
            for replicate in range(replicates):
                for method in METHODS:
                    runs.append(
                        run_one(
                            model,
                            family,
                            epsilon,
                            int(truth_seed),
                            replicate,
                            method,
                            basis,
                            curvatures,
                            common_ridge,
                            config,
                            attempt44_config,
                        )
                    )
                    completed += 1
                    latest = runs[-1]["posthoc"]
                    print(
                        f"[{completed}/{total}] {family}:{truth_seed}:r{replicate} "
                        f"{method} terminal_success="
                        f"{latest['terminal_success']} "
                        f"terminal={latest['final_accepted_infidelity']:.3e}",
                        flush=True,
                    )
    rows = truth_rows(runs)
    summary = bootstrap_summary(rows, config)
    result = {
        "attempt": 53,
        "status": "smoke" if smoke else "complete",
        "evidence_level": "implementation-smoke"
        if smoke
        else "exploratory-development",
        "git_commit": git("rev-parse", "HEAD"),
        "source_hashes": {
            "runner": canonical_sha256(Path(__file__)),
            "config": canonical_sha256(CONFIG_PATH),
            "protocol": canonical_sha256(PROTOCOL_PATH),
            "attempt44_implementation": canonical_sha256(
                HERE / "attempt44_dimension_cost.py"
            ),
        },
        "config": config,
        "geometry": {
            "basis_sha256": base.array_sha256(basis),
            "curvatures_sha256": base.array_sha256(curvatures),
            "common_ridge": common_ridge,
            "audit": geometry_audit,
        },
        "runs": runs,
        "truth_rows": rows,
        "summary": summary,
        "firewall": {
            "exact_truth_available_online": False,
            "all_runs_retained": True,
            "same_budget_all_methods": True,
            "paired_noise_seed_all_methods": True,
        },
    }
    output = SMOKE_PATH if smoke else OUTPUT_PATH
    atomic_json(output, result)
    if not smoke:
        make_plot(summary)
        write_report(result)
    return result


def verify(path: Path) -> None:
    result = read_json(path)
    config = result["config"]
    validate_config(config)
    expected_runs = (
        len(config["benchmark"]["families"])
        * len(config["benchmark"]["truth_seeds"])
        * int(config["benchmark"]["replicates_per_truth_cell"])
        * len(METHODS)
    )
    if result["status"] == "complete" and len(result["runs"]) != expected_runs:
        raise RuntimeError((len(result["runs"]), expected_runs))
    for run in result["runs"]:
        closure = run["online"]["ledger_closure"]
        budget = config["budget"]["by_method"][run["method"]]
        if (
            int(closure["row_count"]) != int(budget["total_queries"])
            or int(closure["total_shots"])
            != int(budget["total_shots"])
            or run["posthoc"]["exact_values_used_online"] is not False
        ):
            raise RuntimeError(f"closure failure in {run['method']}")
    print(
        f"attempt53 verify pass; runs={len(result['runs'])}; "
        f"methods={len(METHODS)}; budgets=34/50/66 queries",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.smoke:
        execute(smoke=True)
    elif arguments.run:
        execute(smoke=False)
    else:
        verify(OUTPUT_PATH)


if __name__ == "__main__":
    main()

"""Run fiesta model fitting on ZTF photometry CSVs.

Uses numpyro SVI with AutoNormal guide and lr=0.01.
Fits phenomenological models (Bazin, Villar, TDE, Afterglow),
physics models (Arnett, Metzger with fitted distance), and MultiBazin.

Usage:
    # Single source:
    python fiesta_fit_ztf.py --csv data/photometry/ZTF18aafzers.csv

    # Batch (all CSVs in a directory):
    python fiesta_fit_ztf.py --csv-dir data/photometry --output-dir fiesta_results

    # Specific models:
    python fiesta_fit_ztf.py --csv data/photometry/ZTF18aafzers.csv --models BazinModel VillarModel
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import pandas as pd
import sncosmo

from fiesta.inference.analytical_models import (
    AfterglowModel,
    BazinModel,
    PhenomenologicalTDEModel,
    VillarModel,
)
from fiesta.inference.analytical_models.supernova_models import ArnettModel
from fiesta.inference.analytical_models.kilonova_models import MetzgerModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── ZTF filter registration ──────────────────────────────────────────────

ZTF_EFF_WAVELENGTHS = {"ztfg": 4722.7, "ztfr": 6339.6, "ztfi": 7886.1}
_FILTERS_REGISTERED = False


def register_ztf_filters():
    global _FILTERS_REGISTERED
    if _FILTERS_REGISTERED:
        return
    for name, eff_wave in ZTF_EFF_WAVELENGTHS.items():
        half_width = 500.0
        wave = np.array([eff_wave - half_width, eff_wave - 1.0,
                         eff_wave + 1.0, eff_wave + half_width])
        trans = np.array([0.0, 1.0, 1.0, 0.0])
        band = sncosmo.Bandpass(wave, trans, name=name)
        sncosmo.register(band, force=True)
    _FILTERS_REGISTERED = True
    log.info("Registered ZTF top-hat bandpasses: %s",
             list(ZTF_EFF_WAVELENGTHS.keys()))


# ── Data loading ─────────────────────────────────────────────────────────

BAND_MAP = {1: "ztfg", 2: "ztfr", 3: "ztfi"}

# ── Rust lightcurve-fitting result loading ────────────────────────────────

# Mapping from Rust model names → (fiesta model name, param index mapping)
# Rust params are in natural log space; fiesta uses log10.
# Each entry: rust_param_index → (fiesta_param_name, transform)
# transform: "ln_to_log10" for log-space params, "identity" for linear params
_LN_TO_LOG10 = 1.0 / np.log(10.0)

_RUST_TO_FIESTA_MAP = {
    "Bazin": {
        "fiesta_model": "BazinModel",
        # Rust: [log_A(0), B(1), t0(2), log_tau_rise(3), log_tau_fall(4), log_sigma(5)]
        "params": {
            2: ("t0", "identity"),
            3: ("log10_tau_rise", "ln_to_log10"),
            4: ("log10_tau_fall", "ln_to_log10"),
        },
    },
    "Villar": {
        "fiesta_model": "VillarModel",
        # Rust: [log_A(0), beta(1), log_gamma(2), t0(3), log_tau_rise(4), log_tau_fall(5), log_sigma(6)]
        "params": {
            3: ("t0", "identity"),
            4: ("log10_tau_rise", "ln_to_log10"),
            5: ("log10_tau_fall", "ln_to_log10"),
            1: ("beta_slope", "identity"),
            2: ("log10_gamma", "ln_to_log10"),
        },
    },
    "Tde": {
        "fiesta_model": "PhenomenologicalTDEModel",
        # Rust: [log_A(0), B(1), t0(2), log_tau_rise(3), log_tau_fall(4), alpha(5), log_sigma(6)]
        "params": {
            2: ("t0", "identity"),
            3: ("log10_tau_rise", "ln_to_log10"),
            4: ("log10_tau_fall", "ln_to_log10"),
            5: ("alpha_decay", "identity"),
        },
    },
    "Afterglow": {
        "fiesta_model": "AfterglowModel",
        # Rust: [log_A(0), t0(1), log_t_b(2), alpha1(3), alpha2(4), log_sigma(5)]
        "params": {
            1: ("t0", "identity"),
            2: ("log10_t_break", "ln_to_log10"),
            3: ("alpha_1", "identity"),
            4: ("alpha_2", "identity"),
        },
    },
}


def load_rust_fit_results(json_path: str) -> dict | None:
    """Load lightcurve-fitting Rust output JSON.

    Returns dict mapping fiesta_model_name → {param_name → value} for
    shape parameters that can seed SVI initialization.
    Only returns the best-fit band result (longest observation baseline).
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    parametric = data.get("parametric", [])
    if not parametric:
        return None

    # Group by Rust model name, take first band (longest baseline per Rust sort)
    best_per_model = {}
    for band_result in parametric:
        rust_model = band_result.get("model")
        if rust_model not in _RUST_TO_FIESTA_MAP:
            continue
        if rust_model in best_per_model:
            continue  # already have the best (first = longest baseline)
        svi_mu = band_result.get("svi_mu")
        if not svi_mu:
            svi_mu = band_result.get("pso_params")
        if not svi_mu:
            continue
        best_per_model[rust_model] = svi_mu

    # Convert to fiesta parameter space
    fiesta_inits = {}
    for rust_model, mu in best_per_model.items():
        mapping = _RUST_TO_FIESTA_MAP[rust_model]
        fiesta_name = mapping["fiesta_model"]
        params = {}
        for rust_idx, (fiesta_param, transform) in mapping["params"].items():
            if rust_idx >= len(mu):
                continue
            val = mu[rust_idx]
            if transform == "ln_to_log10":
                val = val * _LN_TO_LOG10
            params[fiesta_param] = val
        if params:
            fiesta_inits[fiesta_name] = params

    return fiesta_inits if fiesta_inits else None


def load_csv_for_fiesta(csv_path: str) -> dict[str, np.ndarray] | None:
    """Convert a ZTF photometry CSV to fiesta data format.

    Returns:
        dict[filter_name -> array(n_obs, 3)] with columns [time_days, mag, mag_err]
        Times are relative to first observation.
    """
    raw = pd.read_csv(csv_path)
    raw = raw.dropna(subset=["magpsf", "sigmapsf"])
    if raw.empty:
        return None

    mjd = raw["jd"].values - 2400000.5
    mags = raw["magpsf"].values
    mag_errs = raw["sigmapsf"].values
    fids = raw["fid"].values.astype(int)

    t_first = np.min(mjd)
    times_rel = mjd - t_first

    data = {}
    for fid_val, filt_name in BAND_MAP.items():
        mask = fids == fid_val
        if not np.any(mask):
            continue
        valid = np.isfinite(mags[mask]) & np.isfinite(mag_errs[mask]) & (mag_errs[mask] > 0)
        if not np.any(valid):
            continue
        t = times_rel[mask][valid]
        m = mags[mask][valid]
        e = mag_errs[mask][valid]
        data[filt_name] = np.column_stack([t, m, e])

    return data if data else None


# ── Model definitions ────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "BazinModel": {
        "class": BazinModel,
        "type": "phenomenological",
        "shape_priors": {
            "t0": [-50.0, 200.0],
            "log10_tau_rise": [-2.0, 3.0],
            "log10_tau_fall": [-2.0, 4.0],
        },
        "has_baseline": True,
        "time_grid": {"start": 0.01, "stop": 300.0, "n_points": 200},
    },
    "VillarModel": {
        "class": VillarModel,
        "type": "phenomenological",
        "shape_priors": {
            "t0": [-50.0, 200.0],
            "log10_tau_rise": [-1.0, 2.5],
            "log10_tau_fall": [-0.5, 3.0],
            "beta_slope": [0.0, 2.0],
            "log10_gamma": [-1.0, 2.5],
        },
        "has_baseline": False,
        "time_grid": {"start": 0.01, "stop": 300.0, "n_points": 200},
    },
    "PhenomenologicalTDEModel": {
        "class": PhenomenologicalTDEModel,
        "type": "phenomenological",
        "shape_priors": {
            "t0": [-50.0, 500.0],
            "log10_tau_rise": [-2.0, 3.0],
            "log10_tau_fall": [-2.0, 4.0],
            "alpha_decay": [0.5, 5.0],
        },
        "has_baseline": True,
        "time_grid": {"start": 0.01, "stop": 500.0, "n_points": 200},
    },
    "AfterglowModel": {
        "class": AfterglowModel,
        "type": "phenomenological",
        "shape_priors": {
            "t0": [-50.0, 200.0],
            "log10_t_break": [-1.0, 4.0],
            "alpha_1": [-5.0, 5.0],
            "alpha_2": [-5.0, 5.0],
        },
        "has_baseline": False,
        "time_grid": {"start": 0.01, "stop": 300.0, "n_points": 200},
    },
    "ArnettModel": {
        "class": ArnettModel,
        "type": "physics",
        "shape_priors": {
            "tau_m": [1.0, 60.0],
            "log10_mni": [-3.0, 1.0],
            "v_phot": [0.1, 5.0],
            "log10_dl_Mpc": [-1.0, 4.0],  # 0.1 to 10000 Mpc
        },
        "time_grid": {"start": 0.1, "stop": 100.0, "n_points": 200},
    },
    "MetzgerModel": {
        "class": MetzgerModel,
        "type": "physics",
        "shape_priors": {
            "log10_mej": [-4.0, -0.5],
            "log10_vej": [-1.5, -0.3],
            "beta": [1.0, 10.0],
            "log10_kappa_r": [-1.0, 2.0],
            "log10_dl_Mpc": [-1.0, 4.0],
        },
        "time_grid": {"start": 0.1, "stop": 30.0, "n_points": 100},
    },
    "MultiBazin": {
        "class": None,
        "type": "multi_bazin",
        "shape_priors": {},
        "has_baseline": True,
        "time_grid": {"start": 0.01, "stop": 300.0, "n_points": 200},
    },
}


def get_prior_bounds(model, config, data_dict):
    """Return dict of {param_name: (lo, hi)} for each model parameter."""
    bounds = {}
    for param_name in model.parameter_names:
        if param_name in config["shape_priors"]:
            bounds[param_name] = tuple(config["shape_priors"][param_name])
        elif param_name.startswith("amp_mag_") or param_name.startswith("base_mag_"):
            filt = param_name.replace("amp_mag_", "").replace("base_mag_", "")
            if filt in data_dict:
                mags = data_dict[filt][:, 1]
                bounds[param_name] = (float(np.nanmin(mags)) - 3.0,
                                      float(np.nanmax(mags)) + 3.0)
            else:
                bounds[param_name] = (10.0, 25.0)
        else:
            bounds[param_name] = (-10.0, 10.0)
    return bounds


# ── SVI fitting ──────────────────────────────────────────────────────────

ERROR_BUDGET = 0.3  # systematic uncertainty floor (mag)


def make_numpyro_model(fiesta_model, prior_bounds, obs_times, obs_mags,
                       obs_errs, obs_filters):
    """Build a numpyro model for phenomenological fiesta models."""
    param_names = fiesta_model.parameter_names
    filter_names = fiesta_model.filters

    def model_fn():
        params = {}
        for pname in param_names:
            lo, hi = prior_bounds[pname]
            params[pname] = numpyro.sample(pname, dist.Uniform(lo, hi))

        model_times, mag_pred_dict = fiesta_model.predict(params)

        for filt in filter_names:
            if filt not in obs_times:
                continue
            t_obs = obs_times[filt]
            m_obs = obs_mags[filt]
            e_obs = obs_errs[filt]
            m_pred = jnp.interp(t_obs, model_times, mag_pred_dict[filt])
            sigma = jnp.sqrt(e_obs ** 2 + ERROR_BUDGET ** 2)
            numpyro.sample(f"obs_{filt}", dist.Normal(m_pred, sigma), obs=m_obs)

    return model_fn


def make_physics_numpyro_model(fiesta_model, prior_bounds, obs_times, obs_mags,
                               obs_errs, obs_filters):
    """Build a numpyro model for physics models with fitted distance.

    Samples log10_dl_Mpc as a free parameter, converts to luminosity_distance
    in cm, and passes it to fiesta's predict().
    """
    # Model's own params (no distance) + our distance param
    model_param_names = fiesta_model.parameter_names
    filter_names = fiesta_model.filters

    def model_fn():
        params = {}
        for pname in model_param_names:
            lo, hi = prior_bounds[pname]
            params[pname] = numpyro.sample(pname, dist.Uniform(lo, hi))

        # Distance: sample in log10(Mpc), pass as Mpc to fiesta
        lo_d, hi_d = prior_bounds["log10_dl_Mpc"]
        log10_dl = numpyro.sample("log10_dl_Mpc", dist.Uniform(lo_d, hi_d))
        params["luminosity_distance"] = jnp.power(10.0, log10_dl)  # Mpc

        model_times, mag_pred_dict = fiesta_model.predict(params)

        for filt in filter_names:
            if filt not in obs_times:
                continue
            t_obs = obs_times[filt]
            m_obs = obs_mags[filt]
            e_obs = obs_errs[filt]
            m_pred = jnp.interp(t_obs, model_times, mag_pred_dict[filt])
            sigma = jnp.sqrt(e_obs ** 2 + ERROR_BUDGET ** 2)
            numpyro.sample(f"obs_{filt}", dist.Normal(m_pred, sigma), obs=m_obs)

    return model_fn


def _compute_init_values(fiesta_model, prior_bounds, data_dict, filters,
                         is_physics=False, rust_init=None):
    """Compute sensible initial parameter values from data.

    Args:
        rust_init: Optional dict of {param_name: value} from Rust fit results.
                   When provided, these values override the data-derived defaults
                   for shape parameters (t0, timescales, slopes).
    """
    init = {}
    for pname in fiesta_model.parameter_names:
        lo, hi = prior_bounds[pname]

        # Check if Rust fit provides this parameter
        if rust_init and pname in rust_init:
            init[pname] = float(np.clip(rust_init[pname], lo, hi))
            continue

        if pname == "t0":
            all_mags, all_times = [], []
            for filt in filters:
                if filt in data_dict:
                    all_mags.extend(data_dict[filt][:, 1].tolist())
                    all_times.extend(data_dict[filt][:, 0].tolist())
            if all_mags:
                best_idx = int(np.argmin(all_mags))
                init[pname] = np.clip(all_times[best_idx], lo, hi)
            else:
                init[pname] = (lo + hi) / 2
        elif pname.startswith("amp_mag_"):
            filt = pname.replace("amp_mag_", "")
            if filt in data_dict:
                init[pname] = float(np.min(data_dict[filt][:, 1]))
            else:
                init[pname] = (lo + hi) / 2
        elif pname.startswith("base_mag_"):
            filt = pname.replace("base_mag_", "")
            if filt in data_dict:
                init[pname] = float(np.max(data_dict[filt][:, 1]))
            else:
                init[pname] = (lo + hi) / 2
        elif pname == "beta_slope":
            init[pname] = lo + 0.1 * (hi - lo)
        elif pname == "log10_gamma":
            all_t = []
            for filt in filters:
                if filt in data_dict:
                    all_t.extend(data_dict[filt][:, 0].tolist())
            if all_t:
                span = max(all_t) - min(all_t)
                init[pname] = np.clip(np.log10(max(span / 3.0, 0.1)), lo, hi)
            else:
                init[pname] = (lo + hi) / 2
        else:
            init[pname] = (lo + hi) / 2

    # Physics models: init distance from apparent magnitude
    if is_physics and "log10_dl_Mpc" in prior_bounds:
        lo, hi = prior_bounds["log10_dl_Mpc"]
        if rust_init and "log10_dl_Mpc" in rust_init:
            init["log10_dl_Mpc"] = float(np.clip(rust_init["log10_dl_Mpc"], lo, hi))
        else:
            init["log10_dl_Mpc"] = np.clip(2.0, lo, hi)  # ~100 Mpc default

    return init


def _run_svi(svi, rng_key, num_iter):
    """Run SVI using svi.run() for full JIT speed."""
    svi_result = svi.run(rng_key, num_iter, progress_bar=False)
    return svi_result.params, np.array(svi_result.losses), num_iter


def fit_model_svi(
    model_name: str,
    config: dict,
    data_dict: dict[str, np.ndarray],
    filters: list[str],
    num_iter: int = 2_000,
    num_samples: int = 1000,
    learning_rate: float = 0.01,
    random_seed: int = 42,
    rust_init: dict | None = None,
) -> dict | None:
    """Fit a single model using numpyro SVI.

    Args:
        rust_init: Optional dict of {param_name: value} from Rust fit results
                   to seed SVI initialization for shape parameters.
    """

    is_physics = config.get("type") == "physics"

    # Build time grid
    tg = config["time_grid"]
    all_times = np.concatenate([d[:, 0] for d in data_dict.values()])
    t_stop = max(tg["stop"], float(np.max(all_times)) + 10.0)
    if is_physics:
        times = jnp.geomspace(max(tg["start"], 0.1), t_stop, tg["n_points"])
    else:
        times = jnp.linspace(max(tg["start"], 0.001), t_stop, tg["n_points"])

    # Create fiesta model
    fiesta_model = config["class"](filters=filters, times=times)
    prior_bounds = get_prior_bounds(fiesta_model, config, data_dict)

    # Add distance prior for physics models
    if is_physics:
        prior_bounds["log10_dl_Mpc"] = tuple(config["shape_priors"]["log10_dl_Mpc"])

    # Prepare observation arrays per filter
    obs_times = {}
    obs_mags = {}
    obs_errs = {}
    for filt in filters:
        if filt in data_dict:
            obs_times[filt] = jnp.array(data_dict[filt][:, 0])
            obs_mags[filt] = jnp.array(data_dict[filt][:, 1])
            obs_errs[filt] = jnp.array(data_dict[filt][:, 2])

    # Build numpyro model
    if is_physics:
        numpyro_model = make_physics_numpyro_model(
            fiesta_model, prior_bounds, obs_times, obs_mags, obs_errs, filters,
        )
    else:
        numpyro_model = make_numpyro_model(
            fiesta_model, prior_bounds, obs_times, obs_mags, obs_errs, filters,
        )

    # All parameter names for posteriors
    all_param_names = list(fiesta_model.parameter_names)
    if is_physics:
        all_param_names.append("log10_dl_Mpc")

    # Compute data-informed initialization
    init_values = _compute_init_values(
        fiesta_model, prior_bounds, data_dict, filters, is_physics=is_physics,
        rust_init=rust_init,
    )

    guide = AutoNormal(
        numpyro_model,
        init_loc_fn=numpyro.infer.init_to_value(values=init_values),
    )

    optimizer = numpyro.optim.ClippedAdam(learning_rate, clip_norm=10.0)
    svi = SVI(numpyro_model, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(random_seed)
    svi_params, losses, n_iter_actual = _run_svi(svi, rng_key, num_iter)

    final_elbo = float(losses[-1])

    # Draw posterior samples
    predictive = numpyro.infer.Predictive(guide, params=svi_params,
                                          num_samples=num_samples)
    rng_key = jax.random.PRNGKey(random_seed + 1)
    posterior_samples = predictive(rng_key)

    result = {
        "model": model_name,
        "parameters": {},
        "final_elbo": final_elbo,
        "n_iter": n_iter_actual,
    }

    for pname in all_param_names:
        if pname not in posterior_samples:
            continue
        vals = np.array(posterior_samples[pname])
        result["parameters"][pname] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "q16": float(np.percentile(vals, 16)),
            "q84": float(np.percentile(vals, 84)),
        }

    # Per-sample reduced chi-squared scoring
    n_obs_total = sum(len(obs_mags[f]) for f in filters if f in obs_mags)
    n_params = len(all_param_names)
    dof = max(n_obs_total - n_params, 1)
    result["n_obs"] = n_obs_total

    try:
        if is_physics:
            def _chi2_single(param_vals):
                params = {pname: param_vals[j]
                          for j, pname in enumerate(fiesta_model.parameter_names)}
                params["luminosity_distance"] = jnp.power(10.0, param_vals[-1])  # Mpc
                _, mag_pred_dict = fiesta_model.predict(params)
                chi2 = 0.0
                for filt in filters:
                    if filt not in obs_times:
                        continue
                    m_pred = jnp.interp(obs_times[filt], times, mag_pred_dict[filt])
                    sigma = jnp.sqrt(obs_errs[filt] ** 2 + ERROR_BUDGET ** 2)
                    chi2 = chi2 + jnp.sum(((obs_mags[filt] - m_pred) / sigma) ** 2)
                return chi2 / dof
        else:
            def _chi2_single(param_vals):
                params = {pname: param_vals[j]
                          for j, pname in enumerate(fiesta_model.parameter_names)}
                _, mag_pred_dict = fiesta_model.predict(params)
                chi2 = 0.0
                for filt in filters:
                    if filt not in obs_times:
                        continue
                    m_pred = jnp.interp(obs_times[filt], times, mag_pred_dict[filt])
                    sigma = jnp.sqrt(obs_errs[filt] ** 2 + ERROR_BUDGET ** 2)
                    chi2 = chi2 + jnp.sum(((obs_mags[filt] - m_pred) / sigma) ** 2)
                return chi2 / dof

        sample_stack = jnp.column_stack([
            posterior_samples[pname] for pname in all_param_names
            if pname in posterior_samples
        ])
        scores = np.array(jax.vmap(_chi2_single)(sample_stack))

        result["scores"] = scores.tolist()
        result["score_mean"] = float(np.mean(scores))
        result["score_std"] = float(np.std(scores))
        result["score_median"] = float(np.median(scores))

        mean_vals = jnp.array([result["parameters"][pname]["mean"]
                               for pname in all_param_names
                               if pname in result["parameters"]])
        result["chi2_per_dof"] = float(_chi2_single(mean_vals))
    except Exception:
        pass

    return result


# ── MultiBazin model (padded to max K for single JIT) ────────────────────

MULTI_BAZIN_MAX_K = 4
MULTI_BAZIN_BIC_DELTA = 2.0


def _bazin_component(t, log_a, t0, log_tau_rise, log_tau_fall):
    """Single Bazin component: A * exp(-dt/tau_fall) * sigmoid(dt/tau_rise)."""
    tau_rise = jnp.exp(log_tau_rise)
    tau_fall = jnp.exp(log_tau_fall)
    dt = t - t0
    # Clip exponent to prevent inf * 0 = NaN when dt << 0
    fall_exp = jnp.exp(jnp.clip(log_a - dt / tau_fall, -30.0, 30.0))
    return fall_exp * jax.nn.sigmoid(dt / tau_rise)


def _multi_bazin_flux_padded(t, comp_params, mask, baseline):
    """Sum of masked Bazin components + baseline.

    comp_params: (MAX_K, 4) — [log_a, t0, log_tau_rise, log_tau_fall] per component
    mask: (MAX_K,) — 1.0 for active components, 0.0 for inactive
    """
    def _eval_one(params_and_mask):
        p, m = params_and_mask
        return m * _bazin_component(t, p[0], p[1], p[2], p[3])

    contributions = jax.vmap(_eval_one)((comp_params, mask))
    return jnp.sum(contributions) + baseline


def _make_padded_multi_bazin_model(obs_t, obs_flux, obs_flux_err, filt_name,
                                   active_k):
    """Numpyro model with MAX_K components, masking inactive ones.

    Always samples MAX_K components so JAX traces the same structure,
    but zeros out components beyond active_k via a static mask.
    """
    mask = jnp.array([1.0 if c < active_k else 0.0
                      for c in range(MULTI_BAZIN_MAX_K)])

    def model_fn():
        comp_params_list = []
        for c in range(MULTI_BAZIN_MAX_K):
            log_a = numpyro.sample(f"log_a_{c}", dist.Uniform(-3.0, 3.0))
            t0 = numpyro.sample(f"t0_{c}", dist.Uniform(-50.0, 300.0))
            log_tr = numpyro.sample(f"log_tau_rise_{c}", dist.Uniform(-2.0, 5.0))
            log_tf = numpyro.sample(f"log_tau_fall_{c}", dist.Uniform(-2.0, 6.0))
            comp_params_list.append(jnp.array([log_a, t0, log_tr, log_tf]))
        comp_params = jnp.stack(comp_params_list)

        baseline = numpyro.sample("baseline", dist.Uniform(-3.0, 3.0))
        log_sigma = numpyro.sample("log_sigma_extra", dist.Uniform(-5.0, 0.0))
        sigma_extra = jnp.exp(log_sigma)

        pred = jax.vmap(
            lambda ti: _multi_bazin_flux_padded(ti, comp_params, mask, baseline)
        )(obs_t)
        sigma = jnp.sqrt(obs_flux_err ** 2 + sigma_extra ** 2)
        numpyro.sample(f"obs_{filt_name}", dist.Normal(pred, sigma), obs=obs_flux)

    return model_fn

    return model_fn


def _mag_to_flux(mag, zp=23.9):
    return 10 ** ((zp - mag) / 2.5)


def fit_multi_bazin(
    data_dict: dict[str, np.ndarray],
    filters: list[str],
    num_iter: int = 2_000,
    num_samples: int = 1000,
    learning_rate: float = 0.01,
    random_seed: int = 42,
) -> dict | None:
    """Fit MultiBazin (K=1..4) per band with padded model, select best K by BIC."""

    all_results = {"model": "MultiBazin", "parameters": {}, "bands": {}}
    total_best_bic = 0.0
    total_n_obs = 0

    for filt in filters:
        if filt not in data_dict:
            continue

        t_obs = jnp.array(data_dict[filt][:, 0])
        m_obs = data_dict[filt][:, 1]
        e_obs = data_dict[filt][:, 2]

        f_obs = jnp.array(_mag_to_flux(m_obs))
        f_err = jnp.array(np.abs(f_obs) * e_obs * np.log(10) / 2.5)
        n_obs = len(t_obs)
        total_n_obs += n_obs

        best_k = 1
        best_bic = float("inf")
        best_result = None

        for k in range(1, MULTI_BAZIN_MAX_K + 1):
            n_params_k = 4 * k + 2  # active params only for BIC

            model_fn = _make_padded_multi_bazin_model(
                t_obs, f_obs, f_err, filt, active_k=k,
            )
            guide = AutoNormal(model_fn)
            optimizer = numpyro.optim.ClippedAdam(learning_rate, clip_norm=10.0)
            svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

            rng_key = jax.random.PRNGKey(random_seed + hash(filt) % 10000 + k)
            svi_params, losses, n_iter_actual = _run_svi(svi, rng_key, num_iter)

            final_loss = float(losses[-1])
            bic = 2.0 * final_loss + n_params_k * np.log(n_obs)

            log.debug("[%s] K=%d: ELBO=%.1f, BIC=%.1f, n_iter=%d",
                      filt, k, final_loss, bic, n_iter_actual)

            if np.isfinite(bic) and bic < best_bic:
                best_bic = bic
                best_k = k
                best_result = (svi_params, guide, model_fn, n_iter_actual, final_loss)
            elif bic > best_bic + MULTI_BAZIN_BIC_DELTA:
                break

        total_best_bic += best_bic

        if best_result is None:
            log.warning("[%s] All K values produced NaN ELBO, skipping", filt)
            continue

        svi_params, guide, model_fn, n_iter_actual, final_loss = best_result
        predictive = numpyro.infer.Predictive(guide, params=svi_params,
                                              num_samples=num_samples)
        rng_key = jax.random.PRNGKey(random_seed + hash(filt) % 10000 + 100)
        posterior_samples = predictive(rng_key)

        band_result = {"best_k": best_k, "bic": best_bic,
                       "n_iter": n_iter_actual, "elbo": final_loss}

        # Only extract active component parameters
        for pname, vals in posterior_samples.items():
            if pname.startswith("obs_"):
                continue
            # Skip inactive component params
            for c in range(MULTI_BAZIN_MAX_K):
                if c >= best_k and f"_{c}" in pname:
                    break
            else:
                full_name = f"{filt}__{pname}"
                arr = np.array(vals)
                all_results["parameters"][full_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "q16": float(np.percentile(arr, 16)),
                    "q84": float(np.percentile(arr, 84)),
                }
                band_result[pname] = {"mean": float(np.mean(arr)),
                                      "std": float(np.std(arr))}

        all_results["bands"][filt] = band_result

    all_results["total_bic"] = total_best_bic
    all_results["n_obs"] = total_n_obs
    return all_results


# ── Per-source pipeline ──────────────────────────────────────────────────

def _fit_one_model(model_name, config, data_dict, filters, num_iter,
                   num_samples, rust_init=None):
    """Fit a single model."""
    t0 = time.time()
    try:
        if model_name == "MultiBazin":
            result = fit_multi_bazin(
                data_dict, filters, num_iter=num_iter,
                num_samples=num_samples,
            )
        else:
            result = fit_model_svi(
                model_name, config, data_dict, filters,
                num_iter=num_iter, num_samples=num_samples,
                rust_init=rust_init,
            )
        elapsed = time.time() - t0
        if result is not None:
            result["elapsed_seconds"] = round(elapsed, 2)
        return model_name, result, elapsed, None
    except Exception as e:
        elapsed = time.time() - t0
        return model_name, None, elapsed, str(e)


def run_fiesta_from_csv(
    csv_path: str,
    output_dir: str = "fiesta_results",
    models: list[str] | None = None,
    num_iter: int = 2_000,
    num_samples: int = 1000,
    rust_results_path: str | None = None,
) -> dict | None:
    """Run fiesta model fits on a ZTF photometry CSV.

    Args:
        csv_path: Path to the ZTF photometry CSV file.
        output_dir: Directory for outputs.
        models: List of model names to fit (default: all).
        num_iter: SVI iterations (default: 2000).
        num_samples: Posterior samples to draw (default: 1000).
        rust_results_path: Optional path to Rust lightcurve-fitting JSON output.
            If provided, shape parameters from the Rust fit seed SVI initialization,
            giving faster convergence.  If None, tries {output_dir}/../rust_results/{source_id}.json.
    """
    csv_path = Path(csv_path)
    source_id = csv_path.stem

    register_ztf_filters()

    data_dict = load_csv_for_fiesta(str(csv_path))
    if data_dict is None:
        log.warning("[%s] No valid photometry", source_id)
        return None

    filters = sorted(data_dict.keys())
    n_obs = sum(d.shape[0] for d in data_dict.values())
    log.info("[%s] Loaded %d observations in %d bands: %s",
             source_id, n_obs, len(filters), filters)

    if n_obs < 5:
        log.warning("[%s] Too few observations (%d)", source_id, n_obs)
        return None

    # Load Rust fit results for SVI initialization
    rust_inits = None
    if rust_results_path is None:
        # Try default location
        default_rust = os.path.join(output_dir, "..", "rust_results", f"{source_id}.json")
        if os.path.exists(default_rust):
            rust_results_path = default_rust
    if rust_results_path:
        rust_inits = load_rust_fit_results(rust_results_path)
        if rust_inits:
            log.info("[%s] Loaded Rust init for models: %s",
                     source_id, list(rust_inits.keys()))

    if models is None:
        models = list(MODEL_CONFIGS.keys())

    valid_models = [(m, MODEL_CONFIGS[m]) for m in models if m in MODEL_CONFIGS]

    source_out = os.path.join(output_dir, source_id)
    os.makedirs(source_out, exist_ok=True)

    all_results = {"source_id": source_id, "n_obs": n_obs, "filters": filters,
                   "models": {}}

    t_total = time.time()

    for mname, config in valid_models:
        # Get Rust init for this specific model, if available
        model_rust_init = rust_inits.get(mname) if rust_inits else None
        if model_rust_init:
            log.info("[%s] Using Rust init for %s: %s", source_id, mname,
                     {k: f"{v:.3f}" for k, v in model_rust_init.items()})
        log.info("[%s] Fitting %s ...", source_id, mname)
        mname, result, elapsed, error = _fit_one_model(
            mname, config, data_dict, filters, num_iter, num_samples,
            rust_init=model_rust_init,
        )
        if error:
            log.error("[%s] %s failed in %.1fs: %s",
                      source_id, mname, elapsed, error)
            all_results["models"][mname] = {
                "model": mname, "error": error,
                "elapsed_seconds": round(elapsed, 2),
            }
        elif result is not None:
            all_results["models"][mname] = result
            n_iter = result.get("n_iter", "?")
            log.info("[%s] %s done in %.1fs (%s iter), score=%.2f±%.2f, chi2/dof=%.2f",
                     source_id, mname, elapsed, n_iter,
                     result.get("score_mean", float("nan")),
                     result.get("score_std", float("nan")),
                     result.get("chi2_per_dof", float("nan")))

    total_elapsed = time.time() - t_total
    log.info("[%s] All %d models done in %.1fs (wall-clock)",
             source_id, len(valid_models), total_elapsed)

    # Save combined results (strip scores arrays to save space)
    save_results = json.loads(json.dumps(all_results, default=str))
    for mresult in save_results.get("models", {}).values():
        mresult.pop("scores", None)
    results_path = os.path.join(source_out, "results.json")
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    log.info("[%s] Results saved to %s", source_id, results_path)

    # Build flat feature dict
    features = {"source_id": source_id}
    for mname, mresult in all_results["models"].items():
        if "parameters" not in mresult:
            continue
        for pname, stats in mresult["parameters"].items():
            features[f"{mname}__{pname}__mean"] = stats["mean"]
            features[f"{mname}__{pname}__std"] = stats["std"]
        if "chi2_per_dof" in mresult:
            features[f"{mname}__chi2_per_dof"] = mresult["chi2_per_dof"]
        if "score_mean" in mresult:
            features[f"{mname}__score_mean"] = mresult["score_mean"]
            features[f"{mname}__score_std"] = mresult["score_std"]
        if "final_elbo" in mresult:
            features[f"{mname}__final_elbo"] = mresult["final_elbo"]
        if "total_bic" in mresult:
            features[f"{mname}__total_bic"] = mresult["total_bic"]
        if "bands" in mresult:
            for band, binfo in mresult["bands"].items():
                features[f"{mname}__{band}__best_k"] = binfo["best_k"]
                features[f"{mname}__{band}__bic"] = binfo["bic"]

    return features


# ── Batch processing ─────────────────────────────────────────────────────

def run_batch(
    csv_dir: str,
    output_dir: str = "fiesta_results",
    models: list[str] | None = None,
    num_iter: int = 2_000,
    rust_results_dir: str | None = None,
):
    """Process all CSV files in a directory."""
    csv_dir = Path(csv_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    log.info("Found %d CSV files in %s", len(csv_files), csv_dir)

    results_csv = output_dir_path / "fiesta_features.csv"
    header_written = results_csv.exists()

    for i, csv_path in enumerate(csv_files):
        source_id = csv_path.stem
        log.info("[%d/%d] Processing %s", i + 1, len(csv_files), source_id)

        # Look for per-source Rust results
        rust_path = None
        if rust_results_dir:
            candidate = os.path.join(rust_results_dir, f"{source_id}.json")
            if os.path.exists(candidate):
                rust_path = candidate

        try:
            features = run_fiesta_from_csv(
                str(csv_path), output_dir=output_dir,
                models=models, num_iter=num_iter,
                rust_results_path=rust_path,
            )
        except Exception:
            log.exception("[%s] Failed", source_id)
            continue

        if features is None:
            continue

        row = pd.DataFrame([features])
        row.to_csv(results_csv, mode="a", index=False, header=not header_written)
        header_written = True

    log.info("Batch complete. Features saved to %s", results_csv)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit fiesta models to ZTF photometry (SVI)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="Single CSV file to process")
    group.add_argument("--csv-dir", help="Directory of CSV files for batch")
    parser.add_argument("--output-dir", default="fiesta_results",
                        help="Output directory (default: fiesta_results)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Models to fit (default: all)")
    parser.add_argument("--num-iter", type=int, default=2_000,
                        help="SVI iterations (default: 2000)")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Posterior samples to draw (default: 1000)")
    parser.add_argument("--rust-results", default=None,
                        help="Path to Rust lightcurve-fitting JSON output "
                             "for SVI init seeding")
    args = parser.parse_args()

    if args.csv:
        features = run_fiesta_from_csv(
            args.csv, output_dir=args.output_dir,
            models=args.models, num_iter=args.num_iter,
            num_samples=args.num_samples,
            rust_results_path=args.rust_results,
        )
        if features:
            print("\n=== Features ===")
            for k, v in features.items():
                if k != "source_id":
                    print(f"  {k}: {v}")
    else:
        run_batch(
            args.csv_dir, output_dir=args.output_dir,
            models=args.models, num_iter=args.num_iter,
            rust_results_dir=args.rust_results,
        )


if __name__ == "__main__":
    main()

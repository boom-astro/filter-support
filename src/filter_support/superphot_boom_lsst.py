import io
import base64
import json
import logging
from datetime import datetime

from pymongo import MongoClient
import pandas as pd
import numpy as np
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u
import dustmaps.sfd
# dustmaps.sfd.fetch()
from superphot_plus.samplers.numpyro_sampler import SVISampler
from superphot_plus.priors import SuperphotPrior
from superphot_plus.model import SuperphotLightGBM
import matplotlib.pyplot as plt
import jax

from snapi import Photometry, Formatter
import os
from pathlib import Path
from dotenv import load_dotenv


import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("ignore", category=InconsistentVersionWarning)

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Load .env from home directory
# Reading mongodb password
load_dotenv(Path.home() / ".env")


def fetch_mongo(collection_name, url="mongodb://localhost:27017", db_name="boom"):
    """
    Fetch a MongoDB collection.

    Args:
        collection_name (str): Name of the collection to fetch.
        url (str, optional): MongoDB connection URL. Defaults to "mongodb://localhost:27017".
        db_name (str, optional): Name of the database. Defaults to "boom".

    Returns:
        pymongo.collection.Collection or None: The MongoDB collection object if it exists,
            None otherwise.
    """
    
    if url is None:
        user = os.getenv("BOOM_DATABASE__USERNAME")
        password = os.getenv("BOOM_DATABASE__PASSWORD")
        host = "localhost"
        port = "27017"
        
        if user and password:
            url = f"mongodb://{user}:{password}@localhost:27017"
        else:
            url = f"mongodb://{host}:{port}"
    
    db = MongoClient(url)[db_name]
    if collection_name not in db.list_collection_names():
        return None

    return db[collection_name]


def process_photometry(cand_info, source):
    """
    Process photometry data from candidate information into a pandas DataFrame.

    Extracts photometric measurements from both alert candidates and forced photometry,
    converting Julian dates to Modified Julian Dates (MJD) and organizing the data
    into a structured format.

    Args:
        cand_info (dict): Dictionary containing candidate information with keys:
            - 'prv_candidates': List of previous candidate detections
            - 'fp_hists': List of forced photometry measurements
        source (str): Source identifier (e.g., 'ZTF', 'LSST') to label the data.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - mjd: Modified Julian Date
            - mag: Magnitude
            - mag_err: Magnitude error
            - filter: Filter/band name
            - type: Type of photometry ('alert' or 'forced_photometry')
            - source: Source identifier
    """
    candidates = cand_info["prv_candidates"]
    forced_photometry = cand_info["fp_hists"]
    data_dict_list = []
    
    # Process alert candidates
    for obj in candidates:
        temp_dict = {
            "mjd": obj["jd"] - 2400000.5,
            "mag": obj["magpsf"],
            "mag_err": obj["sigmapsf"],
            "filter": obj["band"],
            "type": "alert",
            "source": source
        }
        data_dict_list.append(temp_dict)

    # Process forced photometry
    for obj in forced_photometry:
        if "magpsf" in obj:
            temp_dict = {
                "mjd": obj["jd"] - 2400000.5,
                "mag": obj["magpsf"],
                "mag_err": obj["sigmapsf"],
                "filter": obj["band"],
                "type": "forced_photometry",
                "source": source
            }
            data_dict_list.append(temp_dict)

    return pd.DataFrame.from_dict(data_dict_list)


def evaluate_cal_probs(model, orig_features):
    """
    Evaluate classification probabilities using a calibrated LightGBM model.

    This function normalizes input features, predicts class probabilities, and
    calculates the most likely class and its probability using a frequentist
    approach (counting best classes across fits).

    Args:
        model (SuperphotLightGBM): Pre-trained Superphot LightGBM classification model.
        orig_features (pd.DataFrame): DataFrame containing unnormalized feature values.

    Returns:
        tuple: A tuple containing:
            - str: Best predicted supernova class
            - float: Probability of the best class
            - dict: Probabilities for all classes (keyed by class name)
    """
    allowed_types = ['SLSN-I', 'SN Ia', 'SN Ibc', 'SN II', 'SN IIn']
    input_features = model.best_model.feature_name_
    test_features = model.normalize(orig_features[input_features])
    
    probabilities = pd.DataFrame(
        model.best_model.predict_proba(test_features),
        index=test_features.index
    )
    
    probabilities.columns = np.sort(allowed_types)
    best_classes = probabilities.idxmax(axis=1)
    
    # Calculate probability as fraction of fits where each class is best
    # (frequentist interpretation for better calibration)
    probs = best_classes.value_counts() / best_classes.count()

    # Build dict with all class probabilities (0.0 for classes not voted best)
    all_probs = {cls: probs.get(cls, 0.0) for cls in allowed_types}

    return probs.idxmax(), probs.max(), all_probs


def _get_annotation_id(lsst_id, origin, headers, base_url):
    """Fetch the annotation ID for a source by origin."""
    endpoint = f"{base_url}/sources/{lsst_id}/annotations"
    try:
        response = requests.get(endpoint, headers=headers)
        resp_json = response.json()
        if resp_json.get("status") == "success":
            for ann in resp_json.get("data", []):
                if ann.get("origin") == origin:
                    return ann.get("annotation_id") or ann.get("id")
    except Exception:
        logger.exception("[%s] Failed to fetch existing annotations", lsst_id)
    return None


def annotate_fritz(
    event_dict,
    lsst_id,
    previous_annotation_id=None,
    group_ids=None,
    origin="superphot_plus",
    token=os.getenv("ORCUS_TOKEN"),
    base_url="https://orcusgate.org/api",
):
    """Post or update per-class probability annotations on Fritz/SkyPortal.

    Args:
        event_dict (dict): Classification results containing per-class probabilities.
        lsst_id (str): LSST identifier used as the source ID on Fritz.
        previous_annotation_id (int or None): Annotation ID to update via PUT. If None, POST a new one.
        group_ids (list of int or None): Group IDs that can view the annotation.
        origin (str): Origin label for the annotation.
        token (str): Fritz API token for authentication.
        base_url (str): Base URL of the Fritz API.

    Returns:
        int or None: The annotation_id of the created/updated annotation.
    """
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json",
    }

    allowed_types = ['SLSN-I', 'SN Ia', 'SN Ibc', 'SN II', 'SN IIn']
    data = {cls: event_dict.get(f'superphot_plus_prob_{cls}', 0.0) for cls in allowed_types}

    payload = {
        "origin": origin,
        "data": data,
    }
    if group_ids is not None:
        payload["group_ids"] = group_ids

    if previous_annotation_id is not None:
        endpoint = f"{base_url}/sources/{lsst_id}/annotations/{previous_annotation_id}"
        response = requests.put(endpoint, json=payload, headers=headers)
    else:
        endpoint = f"{base_url}/sources/{lsst_id}/annotations"
        response = requests.post(endpoint, json=payload, headers=headers)

    resp_json = response.json()
    data_resp = resp_json.get("data", {})

    if resp_json.get("status") == "success":
        logger.info("[%s] Annotation saved.", lsst_id)
        return data_resp.get("annotation_id") or data_resp.get("id")

    # POST likely failed due to duplicate — fetch existing annotation and PUT
    logger.warning("[%s] Annotation POST failed: %s", lsst_id, resp_json.get("message"))
    if previous_annotation_id is None:
        existing_id = _get_annotation_id(lsst_id, origin, headers, base_url)
        if existing_id is not None:
            endpoint = f"{base_url}/sources/{lsst_id}/annotations/{existing_id}"
            retry = requests.put(endpoint, json=payload, headers=headers)
            retry_json = retry.json()
            if retry_json.get("status") == "success":
                logger.info("[%s] Annotation updated via fallback PUT.", lsst_id)
                return existing_id
            logger.error("[%s] Fallback PUT failed: %s", lsst_id, retry_json.get("message"))

    return None


def post_to_fritz(
    event_dict,
    image_path,
    lsst_id,
    token=os.getenv("ORCUS_TOKEN"),
    base_url="https://orcusgate.org/api",
):
    """
    Post classification results and diagnostic plot to a Fritz/SkyPortal instance.

    Posts the event_dict as a JSON-formatted comment on the source, with the
    diagnostic plot attached as an image.

    Args:
        event_dict (dict): Classification results and fit parameters.
        image_path (str or None): Path to the diagnostic plot image.
        lsst_id (str): LSST identifier used as the source ID on Fritz.
        token (str): Fritz API token for authentication.
        base_url (str): Base URL of the Fritz API.

    Returns:
        dict: API response JSON on success.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json",
    }

    endpoint = f"{base_url}/sources/{lsst_id}/comments"

    best_class = event_dict.get("superphot_plus_class", event_dict.get("superphot_plus_class_without_redshift"))
    best_prob = event_dict.get("superphot_plus_prob", event_dict.get("superphot_plus_prob_without_redshift"))
    payload = {"text": f"Superphot+ Classification: {best_class} (Probability: {best_prob})"}

    response = requests.post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def post_to_fritz_with_replace(
    event_dict,
    image_path,
    lsst_id,
    previous_comment_id=None,
    token=os.getenv("ORCUS_TOKEN"),
    base_url="https://orcusgate.org/api",
):
    """Post results to Fritz, replacing a previous comment if one exists.

    Deletes the old comment (if provided) then posts a new one with the
    latest classification results and diagnostic plot.

    Args:
        event_dict (dict): Classification results and fit parameters.
        image_path (str or None): Path to the diagnostic plot image.
        lsst_id (str): LSST identifier used as the source ID on Fritz.
        previous_comment_id (int or None): Fritz comment_id to delete before posting.
        token (str): Fritz API token for authentication.
        base_url (str): Base URL of the Fritz API.

    Returns:
        int or None: The comment_id of the newly created comment.
    """
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json",
    }

    # Delete old comment if it exists
    if previous_comment_id is not None:
        delete_url = f"{base_url}/sources/{lsst_id}/comments/{previous_comment_id}"
        try:
            del_resp = requests.delete(delete_url, headers=headers)
            del_resp.raise_for_status()
            logger.info("[%s] Deleted previous Fritz comment %s", lsst_id, previous_comment_id)
        except requests.HTTPError as e:
            logger.warning("[%s] Could not delete comment %s: %s", lsst_id, previous_comment_id, e)

    # Create new comment
    endpoint = f"{base_url}/sources/{lsst_id}/comments"
    best_class = event_dict.get("superphot_plus_class", event_dict.get("superphot_plus_class_without_redshift"))
    best_prob = event_dict.get("superphot_plus_prob", event_dict.get("superphot_plus_prob_without_redshift"))
    payload = {"text": f"Superphot+ Classification: {best_class} (Probability: {best_prob})"}

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        payload["attachment"] = {
            "body": image_data,
            "name": os.path.basename(image_path),
        }

    response = requests.post(endpoint, json=payload, headers=headers)
    if not response.ok:
        logger.error("[%s] Fritz comment failed (%s): %s", lsst_id, response.status_code, response.text)
    response.raise_for_status()
    resp_json = response.json()
    return resp_json.get("data", {}).get("comment_id")


def run_superphot(lsst_id):
    """
    Run the complete Superphot Plus analysis pipeline for a given transient.

    This function performs the following steps:
    1. Fetches photometry data from MongoDB for LSST and optionally ZTF
    2. Processes and combines multi-survey photometry
    3. Applies extinction correction and phase calculation
    4. Fits light curves using Superphot Plus SVI sampler
    5. Classifies the transient using pre-trained LightGBM models
    6. Generates and saves a diagnostic plot

    Args:
        lsst_id (str): LSST identifier for the transient to analyze.

    Returns:
        tuple or None: A tuple of (event_dict, image_path) where event_dict contains
            fit parameters and classification results, and image_path is the path to
            the saved diagnostic plot (or None if prob <= 0.5). Returns None if
            processing fails due to insufficient data or errors.
    """
    # Fetch LSST photometry
    cand_info = fetch_mongo("LSST_alerts_aux").find_one({"_id": str(lsst_id)})
    if cand_info is None:
        logger.warning("No data found for %s", lsst_id)
        return
    df_lsst = process_photometry(cand_info, "LSST")

    # Attempt to fetch and combine ZTF photometry if available
    if len(cand_info["aliases"]["ZTF"]) != 0:
        ztf_id = cand_info["aliases"]["ZTF"][0]
        ztf_cand_info = fetch_mongo("ZTF_alerts_aux").find_one({"_id": str(ztf_id)})
        df_ztf = process_photometry(ztf_cand_info, "ZTF")

        # Only include ZTF data if it has sufficient coverage in both filters
        if (len(df_ztf.loc[df_ztf["filter"] == "r"]) >= 2 and
            len(df_ztf.loc[df_ztf["filter"] == "g"]) >= 2):
            df_final = pd.concat([df_lsst, df_ztf])
        else:
            df_final = df_lsst
    else:
        df_final = df_lsst

    # Filter to only r and g bands
    df_final = df_final.loc[df_final['filter'].isin(["r", "g"])].copy()
    df_final.reset_index(drop=True, inplace=True)

    # Check for minimum data requirements
    if (len(df_final.loc[df_final["filter"] == "r"]) <= 2 or
        len(df_final.loc[df_final["filter"] == "g"]) <= 2):
        logger.warning("Not enough points for %s", lsst_id)
        return

    # Add filter metadata for SNAPI
    df_final['filt_center'] = np.where(df_final['filter'] == 'r', 6366.38, 4746.48)
    df_final['filt_width'] = np.where(df_final['filter'] == 'r', 1553.43, 1317.15)
    df_final['filter'] = np.where(df_final['filter'] == 'r', 'ZTF_r', 'ZTF_g')
    df_final['zeropoint'] = 8.9  # AB mag
    df_final['upper_limit'] = False

    df_final = df_final.loc[df_final["type"] == "alert"]

    # Create SNAPI photometry object
    phot = Photometry(df_final)

    # Merge close-time observations
    new_lcs = []
    for lc in phot.light_curves:
        lc.merge_close_times(inplace=True)
        new_lcs.append(lc)

    phot = Photometry.from_light_curves(new_lcs)
    phot.upper_limit = False
    
    # Phase and truncate light curves
    phot.phase(inplace=True)
    phot.truncate(min_t=-50., max_t=100.)

    # Check if we have valid detections after truncation
    if phot.detections.empty or phot.detections['flux'].dropna().empty:
        logger.warning("No valid detections after truncation for %s", lsst_id)
        return

    # Apply Milky Way extinction correction
    phot.correct_extinction(
        coordinates=SkyCoord(ra=92.44 * u.deg, dec=35.7 * u.deg),
        inplace=True
    )
    
    redshift = np.nan
    
    # Calculate peak absolute magnitude
    phot_abs = phot.absolute(redshift)
    peak_abs_mag = phot_abs.detections.mag.dropna().min()

    # Normalize photometry
    phot.normalize(inplace=True)

    # Check that both filters still have data after normalization
    for filt in phot._unique_filters:
        filt_data = phot.detections[phot.detections['filter'] == filt]
        if filt_data.empty:
            logger.warning("Filter %s has no data after normalization for %s", filt, lsst_id)
            return

    # Pad light curves to nearest power of 2 for model input
    padded_lcs = []
    orig_size = len(phot.detections)
    num_pad = int(2**np.ceil(np.log2(orig_size)))
    fill = {
        'phase': 1000.,
        'flux': 0.1,
        'flux_error': 1000.,
        'zeropoint': 23.90,
        'upper_limit': False
    }

    for lc in phot.light_curves:
        padded_lc = lc.pad(fill, num_pad - len(lc.detections))
        padded_lcs.append(padded_lc)
    padded_phot = Photometry.from_light_curves(padded_lcs)

    # Load priors and fit using SVI sampler
    priors = SuperphotPrior.load('../../data/models/global_priors_hier_svi')
    random_seed = 42

    try:
        svi_sampler = SVISampler(
            priors=priors,
            num_iter=3000,
            random_state=random_seed)
        svi_sampler.fit_photometry(padded_phot, orig_num_times=orig_size)
    except Exception:
        logger.exception("SVI sampler failed for %s", lsst_id)
        return None, None

    res = svi_sampler.result

    # Store fit parameters (convert numpy types to native Python for JSON serialization)
    event_dict = {}
    for param in res.fit_parameters.columns:
        event_dict[f'superphot_plus_{param}'] = float(res.fit_parameters[param].median())

    # Filter fits by quality score
    score_cutoff = 1.2
    if orig_size >= 6:
        valid_fits = res.fit_parameters[res.score <= score_cutoff]
    else:
        valid_fits = res.fit_parameters
    
    if valid_fits.empty:
        logger.warning("Empty fits for %s", lsst_id)
        return None, None
    
    try:
        # Identify early-phase fits (all observations before piecewise transition)
        early_fit_mask = (valid_fits['gamma_ZTF_r'] + valid_fits['t_0_ZTF_r'] > 
                          np.max(phot.times))
    except UnboundLocalError:
        logger.warning("No valid returns for %s", lsst_id)
        return None, None

    # Convert fit parameters to uncorrelated Gaussian draws
    uncorr_fits = priors.reverse_transform(valid_fits)
    event_dict['name'] = str(lsst_id)
    uncorr_fits.index = [event_dict['name']] * len(uncorr_fits)

    

    # Load classification models
    full_model_fn = "../../data/models/model_superphot_full.pt"
    early_model_fn = "../../data/models/model_superphot_early.pt"
    full_model_fn_z = "../../data/models/model_superphot_redshift.pt"
    early_model_fn_z = "../../data/models/model_superphot_early_redshift.pt"

    full_model = SuperphotLightGBM.load(full_model_fn)
    early_model = SuperphotLightGBM.load(early_model_fn)
    full_model_z = SuperphotLightGBM.load(full_model_fn_z)
    early_model_z = SuperphotLightGBM.load(early_model_fn_z)

    # Classify using appropriate model (early vs. full phase)
    if len(valid_fits[early_fit_mask]) > len(valid_fits[~early_fit_mask]):
        # Use early-phase classifier
        class_noz, prob_noz, all_probs_noz = evaluate_cal_probs(early_model, uncorr_fits)
        event_dict['superphot_plus_classifier'] = 'early_lightgbm_02_2025'
    else:
        # Use full-phase classifier
        class_noz, prob_noz, all_probs_noz = evaluate_cal_probs(full_model, uncorr_fits)
        event_dict['superphot_plus_classifier'] = 'full_lightgbm_02_2025'

    # Store classification results
    if ~np.isnan(redshift):
        event_dict['superphot_plus_class_without_redshift'] = class_noz
        event_dict['superphot_plus_prob_without_redshift'] = float(np.round(prob_noz, 3))
    else:
        event_dict['superphot_plus_class'] = class_noz
        event_dict['superphot_plus_prob'] = float(np.round(prob_noz, 3))

    # Store per-class probabilities
    for cls, cls_prob in all_probs_noz.items():
        event_dict[f'superphot_plus_prob_{cls}'] = float(np.round(cls_prob, 3))

    event_dict['superphot_plus_classified'] = True

    image_path = None
    if event_dict['superphot_plus_prob'] > 0.5:

        # Generate diagnostic plot
        fig, ax = plt.subplots(figsize=(8, 6))
        formatter = Formatter()
        ax = svi_sampler.plot_fit(ax, formatter, phot)
        phot.plot(ax, formatter, mags=False)
        formatter.add_legend(ax)
        formatter.make_plot_pretty(ax)
        ax.set_xlabel('Phase', fontsize=15)
        ax.set_ylabel('Flux', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend()
        plt.title(
            f"{lsst_id}, Class: {event_dict['superphot_plus_class']}, "
            f"Probability: {event_dict['superphot_plus_prob']}",
            fontsize=18
        )
        image_path = f"superphot_results/{lsst_id}_superphot.png"
        plt.savefig(image_path)
        plt.close(fig)

    import gc
    gc.collect()
    jax.clear_caches()

    return event_dict, image_path


def run_superphot_from_csv(csv_path):
    """
    Run the Superphot Plus pipeline on photometry loaded from a local CSV file.

    The CSV is expected to have ZTF-format columns including at least:
    jd, fid (1=g, 2=r), magpsf, sigmapsf, ra, dec, obj_id.

    Args:
        csv_path (str or Path): Path to the photometry CSV file.

    Returns:
        tuple or None: (event_dict, image_path) on success, None on failure.
    """
    csv_path = Path(csv_path)
    source_id = csv_path.stem

    raw = pd.read_csv(csv_path)

    # Keep only rows with valid photometry
    raw = raw.dropna(subset=["magpsf", "sigmapsf"])
    if raw.empty:
        logger.warning("[%s] No valid photometry rows in CSV", source_id)
        return None

    # Map fid to filter name
    fid_map = {1: "g", 2: "r"}
    raw["filter"] = raw["fid"].map(fid_map)
    raw = raw.dropna(subset=["filter"])

    df_final = pd.DataFrame({
        "mjd": raw["jd"] - 2400000.5,
        "mag": raw["magpsf"],
        "mag_err": raw["sigmapsf"],
        "filter": raw["filter"],
    })

    # Filter to only r and g bands
    df_final = df_final.loc[df_final["filter"].isin(["r", "g"])].copy()
    df_final.reset_index(drop=True, inplace=True)

    if (len(df_final.loc[df_final["filter"] == "r"]) <= 2 or
        len(df_final.loc[df_final["filter"] == "g"]) <= 2):
        logger.warning("[%s] Not enough points", source_id)
        return None

    # Add filter metadata for SNAPI
    df_final["filt_center"] = np.where(df_final["filter"] == "r", 6366.38, 4746.48)
    df_final["filt_width"] = np.where(df_final["filter"] == "r", 1553.43, 1317.15)
    df_final["filter"] = np.where(df_final["filter"] == "r", "ZTF_r", "ZTF_g")
    df_final["zeropoint"] = 23.90
    df_final["upper_limit"] = False

    # Create SNAPI photometry object
    phot = Photometry(df_final)

    # Merge close-time observations
    new_lcs = []
    for lc in phot.light_curves:
        lc.merge_close_times(inplace=True)
        new_lcs.append(lc)

    phot = Photometry.from_light_curves(new_lcs)
    phot.upper_limit = False

    # Phase and truncate
    phot.phase(inplace=True)
    phot.truncate(min_t=-50.0, max_t=100.0)

    if phot.detections.empty or phot.detections["flux"].dropna().empty:
        logger.warning("[%s] No valid detections after truncation", source_id)
        return None

    # Extinction correction using median ra/dec from the CSV
    median_ra = raw["ra"].dropna().median()
    median_dec = raw["dec"].dropna().median()
    phot.correct_extinction(
        coordinates=SkyCoord(ra=median_ra * u.deg, dec=median_dec * u.deg),
        inplace=True,
    )

    redshift = np.nan

    phot_abs = phot.absolute(redshift)
    peak_abs_mag = phot_abs.detections.mag.dropna().min()

    phot.normalize(inplace=True)

    for filt in phot._unique_filters:
        filt_data = phot.detections[phot.detections["filter"] == filt]
        if filt_data.empty:
            logger.warning("[%s] Filter %s empty after normalization", source_id, filt)
            return None

    # Pad light curves
    padded_lcs = []
    orig_size = len(phot.detections)
    num_pad = int(2 ** np.ceil(np.log2(orig_size)))
    fill = {
        "phase": 1000.0,
        "flux": 0.1,
        "flux_error": 1000.0,
        "zeropoint": 23.90,
        "upper_limit": False,
    }

    for lc in phot.light_curves:
        padded_lc = lc.pad(fill, num_pad - len(lc.detections))
        padded_lcs.append(padded_lc)
    padded_phot = Photometry.from_light_curves(padded_lcs)

    # Fit using SVI sampler
    priors = SuperphotPrior.load("../../data/models/global_priors_hier_svi")
    random_seed = 42

    try:
        svi_sampler = SVISampler(
            priors=priors,
            num_iter=3000,
            random_state=random_seed,
        )
        svi_sampler.fit_photometry(padded_phot, orig_num_times=orig_size)
    except Exception:
        logger.exception("[%s] SVI sampler failed", source_id)
        return None

    res = svi_sampler.result

    event_dict = {}
    for param in res.fit_parameters.columns:
        event_dict[f"superphot_plus_{param}"] = float(res.fit_parameters[param].median())

    score_cutoff = 1.2
    if orig_size >= 6:
        valid_fits = res.fit_parameters[res.score <= score_cutoff]
    else:
        valid_fits = res.fit_parameters

    if valid_fits.empty:
        logger.warning("[%s] Empty fits", source_id)
        return None

    try:
        early_fit_mask = (
            valid_fits["gamma_ZTF_r"] + valid_fits["t_0_ZTF_r"] > np.max(phot.times)
        )
    except UnboundLocalError:
        logger.warning("[%s] No valid returns", source_id)
        return None

    uncorr_fits = priors.reverse_transform(valid_fits)
    event_dict["name"] = str(source_id)
    uncorr_fits.index = [event_dict["name"]] * len(uncorr_fits)

    # Load classification models
    full_model_fn = "../../data/models/model_superphot_full.pt"
    early_model_fn = "../../data/models/model_superphot_early.pt"

    full_model = SuperphotLightGBM.load(full_model_fn)
    early_model = SuperphotLightGBM.load(early_model_fn)

    if len(valid_fits[early_fit_mask]) > len(valid_fits[~early_fit_mask]):
        class_noz, prob_noz, all_probs_noz = evaluate_cal_probs(early_model, uncorr_fits)
        event_dict["superphot_plus_classifier"] = "early_lightgbm_02_2025"
    else:
        class_noz, prob_noz, all_probs_noz = evaluate_cal_probs(full_model, uncorr_fits)
        event_dict["superphot_plus_classifier"] = "full_lightgbm_02_2025"

    event_dict["superphot_plus_class"] = class_noz
    event_dict["superphot_plus_prob"] = float(np.round(prob_noz, 3))

    for cls, cls_prob in all_probs_noz.items():
        event_dict[f"superphot_plus_prob_{cls}"] = float(np.round(cls_prob, 3))

    event_dict["superphot_plus_classified"] = True

    image_path = None
    if event_dict["superphot_plus_prob"] > 0.5:
        os.makedirs("superphot_results", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        formatter = Formatter()
        ax = svi_sampler.plot_fit(ax, formatter, phot)
        phot.plot(ax, formatter, mags=False)
        formatter.add_legend(ax)
        formatter.make_plot_pretty(ax)
        ax.set_xlabel("Phase", fontsize=15)
        ax.set_ylabel("Flux", fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.legend()
        plt.title(
            f"{source_id}, Class: {event_dict['superphot_plus_class']}, "
            f"Probability: {event_dict['superphot_plus_prob']}",
            fontsize=18,
        )
        image_path = f"superphot_results/{source_id}_superphot.png"
        plt.savefig(image_path)
        plt.close(fig)

    import gc
    gc.collect()
    jax.clear_caches()

    return event_dict, image_path


def run_batch_from_csv(csv_dir="data/photometry", output_dir="superphot_results"):
    """
    Process all CSV files in a directory and save results locally.

    For each CSV, runs the Superphot Plus pipeline and writes:
    - A combined results CSV at <output_dir>/superphot_results_lsst_batch.csv
    - Individual JSON results at <output_dir>/<source_id>.json
    - Diagnostic plots at <output_dir>/<source_id>_superphot.png

    Args:
        csv_dir (str): Directory containing photometry CSV files.
        output_dir (str): Directory to save outputs.
    """
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    logger.info("Found %d CSV files in %s", len(csv_files), csv_dir)

    results_csv = output_dir / "superphot_results_lsst_batch.csv"
    header_written = results_csv.exists()

    for i, csv_path in enumerate(csv_files):
        source_id = csv_path.stem
        logger.info("[%d/%d] Processing %s", i + 1, len(csv_files), source_id)

        try:
            result = run_superphot_from_csv(csv_path)
        except Exception:
            logger.exception("[%s] Failed", source_id)
            continue

        if result is None:
            logger.warning("[%s] No result", source_id)
            continue

        event_dict, image_path = result
        if event_dict is None:
            logger.warning("[%s] Empty classification", source_id)
            continue

        # Save individual JSON
        json_path = output_dir / f"{source_id}.json"
        with open(json_path, "w") as f:
            json.dump(event_dict, f, indent=2, cls=NumpyEncoder)

        # Append to combined CSV
        row = pd.DataFrame([event_dict])
        row.to_csv(results_csv, mode="a", index=False, header=not header_written)
        header_written = True

        logger.info("[%s] Saved results to %s and %s", source_id, json_path, results_csv)

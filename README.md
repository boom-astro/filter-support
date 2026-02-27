# filter-support

Real-time transient classification pipeline using [Superphot+](https://github.com/VTDA-Group/superphot-plus). Consumes astronomical alerts from Kafka (LSST and ZTF surveys), fits light curves with SVI sampling, classifies supernovae using LightGBM, and posts results to a Fritz/SkyPortal instance.

## Architecture

```
Kafka Topics                 Consumers                    Pipeline                    Output
─────────────          ─────────────────────       ──────────────────          ─────────────────
LSST_alerts_results → alerts_consumer_lsst.py → superphot_boom_lsst.py → Fritz annotations
                                                                           → CSV results
                                                                           → diagnostic plots

ZTF_alerts_results  → alerts_consumer_ztf.py  → superphot_boom_ztf.py  → Fritz annotations
                                                                           → CSV results
                                                                           → diagnostic plots
```

## Project Structure

```
filter-support/
├── src/filter_support/
│   ├── superphot_boom_lsst.py      # LSST classification pipeline
│   ├── superphot_boom_ztf.py       # ZTF classification pipeline
│   ├── alerts_consumer_lsst.py     # LSST Kafka consumer
│   ├── alerts_consumer_ztf.py      # ZTF Kafka consumer
│   └── superphot_results/          # Output diagnostic plots
├── data/models/
│   ├── global_priors_hier_svi/     # SVI sampler priors
│   ├── model_superphot_full.pt     # Full-phase LightGBM classifier
│   ├── model_superphot_early.pt    # Early-phase LightGBM classifier
│   ├── model_superphot_redshift.pt
│   └── model_superphot_early_redshift.pt
├── pyproject.toml
└── README.md
```

## Pipeline

Each `run_superphot()` call performs the following:

1. Fetch photometry from MongoDB (`LSST_alerts_aux` or `ZTF_alerts_aux`)
2. Optionally combine cross-survey photometry via aliases
3. Filter to r and g bands, validate minimum data (>2 points per filter)
4. Create SNAPI photometry, merge close-time observations
5. Phase and truncate light curves (-50 to +100 days)
6. Apply Milky Way extinction correction
7. Normalize and pad to power-of-2 length
8. Fit using numpyro SVI sampler (3000 iterations)
9. Filter fits by quality score (cutoff 1.2)
10. Classify using early-phase or full-phase LightGBM model
11. Generate diagnostic plot (if probability > 0.5)

Returns `(event_dict, image_path)` with per-class probabilities for: SLSN-I, SN Ia, SN Ibc, SN II, SN IIn.

## Consumers

The two Kafka consumers run independently and can operate simultaneously:

| | LSST | ZTF |
|---|---|---|
| Topic | `LSST_alerts_results` | `ZTF_alerts_results` |
| CSV output | `superphot_results_lsst.csv` | `superphot_results_ztf.csv` |
| State file | `consumer_state_lsst.json` | `consumer_state_ztf.json` |
| Log file | `superphot_lsst.log` | `superphot_ztf.log` |

Each consumer:
- Runs superphot on every alert that passes the filter
- Posts/updates annotations on Fritz with per-class probabilities
- Posts/replaces comments with best class and probability
- Persists state to JSON for crash recovery

## Fritz/SkyPortal Integration

Results are posted to the Orcus instance (`orcusgate.org/api`):

- **Annotations**: Per-class probability dict (POST on first run, PUT on updates)
- **Comments**: `Superphot+ Classification: SN Ia (Probability: 0.996)`

The `annotate_fritz()` function handles both creation and update: if a POST fails because an annotation already exists, it automatically fetches the existing annotation ID and performs a PUT.

## Setup

**Note:** This pipeline can only be run from a machine with access to the [BOOM](https://github.com/boom-astro) database (MongoDB) and the Kafka alert streams (`LSST_alerts_results`, `ZTF_alerts_results`).

### Requirements

- Python >= 3.11
- MongoDB (local or remote)
- Kafka broker
- Conda environment recommended (`superphot`)

### Installation

```bash
poetry install
```

### Environment Variables

Create `~/.env` with:

```
BOOM_DATABASE__USERNAME=<mongodb_user>
BOOM_DATABASE__PASSWORD=<mongodb_password>
ORCUS_TOKEN=<fritz_api_token>
SLACK_BOT_TOKEN=<slack_token>  # optional
```

### Running

Start from `src/filter_support/`:

```bash
# Run LSST consumer
python alerts_consumer_lsst.py

# Run ZTF consumer (separate terminal)
python alerts_consumer_ztf.py
```


## Dependencies

- [superphot-plus](https://github.com/VTDA-Group/superphot-plus) - SVI light curve fitting and LightGBM classification
- [snapi](https://github.com/kdesoto-astro/snapi) - Photometry processing (phasing, truncation, normalization)
- jax / numpyro - Probabilistic SVI sampler backend
- pymongo - MongoDB access
- confluent-kafka / fastavro - Kafka consumer with Avro deserialization
- astropy / dustmaps - Coordinates, extinction correction

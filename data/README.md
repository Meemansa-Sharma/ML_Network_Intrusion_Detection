# Dataset — UNSW-NB15

## Overview

This project uses the **UNSW-NB15** network intrusion dataset, created by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra.

- **Source:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Citation:** Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). *2015 Military Communications and Information Systems Conference (MilCIS)*, 1–6. https://doi.org/10.1109/MilCIS.2015.7348942

---

## Files required

Place the following two CSV files inside this `data/` directory before running any code:

| Filename | Split | Rows (approx.) |
|---|---|---|
| `UNSW_NB15_training-set.csv` | Training | ~175,341 |
| `UNSW_NB15_testing-set.csv`  | Testing  | ~82,332  |

These are the **official predefined train/test split** files published by UNSW. Do **not** substitute other versions or combine and re-split — the official split is preserved throughout this project.

---

## How to download

1. Visit https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Locate the section titled **"The UNSW-NB15 data set CSV files"**.
3. Download:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`
4. Place both files in this `data/` directory so the paths become:
   ```
   data/UNSW_NB15_training-set.csv
   data/UNSW_NB15_testing-set.csv
   ```

---

## Expected schema

Both files share the same 49-column structure. Column names and types are verified at load time by `src/data_loader.py`. The table below documents the schema as published; actual column names are confirmed on first load and logged.

| Column | Type | Notes |
|---|---|---|
| `id` | int | Row identifier — reviewed in Phase 3 for exclusion |
| `dur` | float | Record total duration |
| `proto` | str | Transaction protocol |
| `service` | str | http, ftp, smtp, ssh, dns, ftp-data, irc, or `-` |
| `state` | str | Connection state |
| `spkts` | int | Source-to-dest packet count |
| `dpkts` | int | Dest-to-source packet count |
| `sbytes` | int | Source-to-dest bytes |
| `dbytes` | int | Dest-to-source bytes |
| `rate` | float | Transfer rate |
| `sttl` | int | Source-to-dest TTL |
| `dttl` | int | Dest-to-source TTL |
| `sload` | float | Source bits per second |
| `dload` | float | Dest bits per second |
| `sloss` | int | Source packets retransmitted/dropped |
| `dloss` | int | Dest packets retransmitted/dropped |
| `sinpkt` | float | Source inter-packet arrival time (ms) |
| `dinpkt` | float | Dest inter-packet arrival time (ms) |
| `sjit` | float | Source jitter (ms) |
| `djit` | float | Dest jitter (ms) |
| `swin` | int | Source TCP window size |
| `stcpb` | int | Source TCP base sequence number |
| `dtcpb` | int | Dest TCP base sequence number |
| `dwin` | int | Dest TCP window size |
| `tcprtt` | float | TCP connection setup round-trip time |
| `synack` | float | TCP SYN → SYN-ACK time |
| `ackdat` | float | TCP SYN-ACK → ACK time |
| `smean` | int | Mean packet size transmitted by source |
| `dmean` | int | Mean packet size transmitted by dest |
| `trans_depth` | int | Depth into re-assembled pipeline |
| `response_body_len` | int | Actual uncompressed content size |
| `ct_srv_src` | int | Connections with same service and src in last 100 |
| `ct_state_ttl` | int | Connections with same state/TTL in last 100 |
| `ct_dst_ltm` | int | Connections to same dest in last 100 by time |
| `ct_src_dport_ltm` | int | Connections with same src and dst port in last 100 |
| `ct_dst_sport_ltm` | int | Connections with same dst and src port in last 100 |
| `ct_dst_src_ltm` | int | Connections with same src and dst in last 100 |
| `is_ftp_login` | int | 1 if FTP session uses admin/user/root/anonymous |
| `ct_ftp_cmd` | int | FTP session command count |
| `ct_flw_http_mthd` | int | HTTP method count in last 100 flows |
| `ct_src_ltm` | int | Connections from same src in last 100 |
| `ct_srv_dst` | int | Connections with same service and dst in last 100 |
| `is_sm_ips_ports` | int | 1 if src and dst IPs equal and ports < 1024 |
| `attack_cat` | str/NaN | Attack category (NaN for normal traffic — this is expected) |
| `label` | int | **Binary target: 0 = normal, 1 = attack** |

> **Note on `attack_cat`:** Normal records (label = 0) have NaN in `attack_cat`. This is by design in the dataset and is not treated as a data-quality error.

---

## Data integrity notes

- Column names are stripped of leading/trailing whitespace on load; original and normalised names are both logged.
- If whitespace-normalisation produces a duplicate column name, loading is aborted with a clear error.
- The test set is **never used for preprocessing, imputation, scaling, or model selection** — only for final evaluation.
- `label` and `attack_cat` are excluded from all binary model inputs. Rationale: `label` is the target; `attack_cat` leaks attack-category information derived from the outcome.
- Identifier columns (`id`) and raw network address columns (`srcip`, `dstip`, `sport`, `dsport` if present) are reviewed in Phase 3 before being included or excluded as features.

---

## Do not commit dataset files

The CSV files are large (combined ~60 MB) and must not be committed to the repository. They are listed in `.gitignore`.

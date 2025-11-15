#!/usr/bin/env python3
import json, math, random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import trange

RNG = np.random.default_rng(42)
random.seed(42)
fake = Faker()

# --------------------------- knobs ---------------------------
N_WORK_ORDERS = 4000
AVG_LINES_PER_WO = 10  # expect 30k–50k lines total at 3–12 per WO
STOPWORKS_COVERAGE = 0.35  # fraction of WOs that get a note
MISATTRIBUTION_RATE = 0.25  # fraction of notes that misattribute
BASE_FAILURE_RATE = 0.12    # baseline failure prior, modified by fingerprints
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
# ------------------------------------------------------------

# Categorical pools
SUPPLIERS = [f"SPLR_{i:02d}" for i in range(1, 18)]
DEVICE_TYPES = ["NUC", "U25", "RACK1U", "RACK2U", "EMBEDDED_A", "EMBEDDED_B", "EDGE_AI", "STORAGE_NODE"]
TECHS = [fake.first_name() for _ in range(90)]
CATALOGS = [f"CAT-{1000+i}" for i in range(120)]
SHIFTS = ["A", "B", "C"]

# Fingerprint library
FINGERPRINTS = {
    "BIOS_REDFISH_TIMEOUT": {
        "weight": 0.35,  # increases failure odds
        "priors": {"device_type": {"RACK1U": 2.0, "RACK2U": 1.6}, "supplier": {"SPLR_03": 1.5}},
        "templates": [
            "redfish client timeout contacting BMC at {ip}; session={hexid}",
            "BMC REST call exceeded 30s; endpoint=/redfish/v1/Systems; token={hexid}",
            "IPMI fallback triggered; redfish unresponsive from {ip}",
        ],
        "stopworks": [
            "BMC mgmt timed out over Redfish; suspect vendor firmware lag.",
            "Mgmt interface unresponsive; redfish timeout.",
            "Network mgmt failure; redfish API stalled.",
        ],
    },
    "DRIVE_NOT_SEATED": {
        "weight": 0.45,
        "priors": {"device_type": {"STORAGE_NODE": 2.5, "RACK2U": 1.3}},
        "templates": [
            "slot {slot}: drive presence=0; expected SAS/SATA; serial={hexid}",
            "storage verify failed: tray {slot} empty or bad seating",
            "hotplug event missed: bay {slot} did not enumerate",
        ],
        "stopworks": [
            "Drive not fully seated in bay.",
            "Storage bay loose, reseated successfully.",
            "Missing enumeration on storage tray."
        ],
    },
    "FIRMWARE_FLASH_MISMATCH": {
        "weight": 0.28,
        "priors": {"catalog_id": {"CAT-1010": 1.8, "CAT-1042": 1.6}},
        "templates": [
            "firmware expected={verA} actual={verB}; aborting flash",
            "catalog policy violation: image {verA} vs device {verB}",
            "flasher refused downgrade from {verB} to {verA}",
        ],
        "stopworks": [
            "Firmware version mismatch against catalog policy.",
            "Flash blocked due to version skew.",
        ],
    },
    "NIC_PXE_DHCP_EXHAUSTED": {
        "weight": 0.22,
        "priors": {"shift": {"B": 1.6, "C": 1.3}},
        "templates": [
            "PXE boot failed: DHCP pool exhausted on VLAN {vlan}",
            "iPXE timeout on {iface}; no lease acquired",
            "PXE-E51: No DHCP or proxyDHCP offers on {iface}",
        ],
        "stopworks": [
            "DHCP scope exhausted; PXE failed.",
            "Network lease starvation during imaging.",
        ],
    },
    "SECURE_BOOT_POLICY_BLOCK": {
        "weight": 0.18,
        "priors": {"supplier": {"SPLR_11": 1.7}},
        "templates": [
            "Secure Boot policy blocked unsigned bootloader {hexid}",
            "UEFI verification failed: signature invalid",
        ],
        "stopworks": [
            "Secure Boot blocked unsigned image.",
            "UEFI signature verification failure.",
        ],
    },
    "TPM_ATTESTATION_FAIL": {
        "weight": 0.25,
        "priors": {"device_type": {"EDGE_AI": 1.9}},
        "templates": [
            "TPM quote invalid; PCR mismatch on indices [0,7]",
            "Attestation failed: EK cert chain not trusted",
        ],
        "stopworks": [
            "TPM attestation failed; PCR mismatch.",
            "EK chain trust issue during attestation.",
        ],
    },
    "USB_ENUMERATION_STALL": {
        "weight": 0.12,
        "priors": {"device_type": {"NUC": 1.8}},
        "templates": [
            "USB bus stall: device {hexid} on port {port} failed to enumerate",
            "Hub reset required; descriptor read timeout on port {port}",
        ],
        "stopworks": [
            "USB device failed to enumerate; hub reset.",
            "Enumeration timeout on front panel port.",
        ],
    },
    "MEMTEST_INTERMITTENT": {
        "weight": 0.30,
        "priors": {"shift": {"C": 1.5}, "supplier": {"SPLR_07": 1.4}},
        "templates": [
            "memtest intermittent failure: pattern=0xAA55; addr=0x{hexid}",
            "DIMM error corrected>threshold; slot {slot}",
        ],
        "stopworks": [
            "Intermittent memory failure under test.",
            "DIMM marginal; errors above threshold.",
        ],
    },
    "GPU_PCIE_TRAINING_ERROR": {
        "weight": 0.27,
        "priors": {"device_type": {"EDGE_AI": 2.3}},
        "templates": [
            "PCIe training failed for GPU0: link width x{w} speed {s}",
            "AER reported fatal error on GPU function; retraining required",
        ],
        "stopworks": [
            "PCIe training error on GPU; retrain.",
            "AER fatal during GPU bring-up.",
        ],
    },
    "SAS_BACKPLANE_LINK_DROP": {
        "weight": 0.26,
        "priors": {"device_type": {"STORAGE_NODE": 2.2}},
        "templates": [
            "SAS link drop: phy{p} reset; negotiated 0 Gbps",
            "HBA reported link loss on expander port {p}",
        ],
        "stopworks": [
            "SAS backplane link instability.",
            "HBA/expander link drop observed.",
        ],
    },
}

def rand_hex(n=8):
    return "".join(RNG.choice(list("0123456789abcdef"), size=n))

def pick_weighted(d):
    # d is {key: weight}
    keys, w = zip(*d.items())
    probs = np.array(w, dtype=float) / np.sum(w)
    return RNG.choice(keys, p=probs)

def contextual_multiplier(fp, wo_row):
    m = 1.0
    priors = fp.get("priors", {})
    for k, table in priors.items():
        val = wo_row[k]
        if val in table:
            m *= table[val]
    return m

def instantiate(template):
    return template.format(
        ip=fake.ipv4_private(),
        hexid=rand_hex(12),
        vlan=RNG.integers(10, 4094),
        iface=RNG.choice(["eno1", "eno2", "p1p1", "p1p2"]),
        slot=RNG.integers(0, 11),
        port=RNG.integers(1, 8),
        verA=f"v{RNG.integers(1,5)}.{RNG.integers(0,10)}.{RNG.integers(0,20)}",
        verB=f"v{RNG.integers(1,5)}.{RNG.integers(0,10)}.{RNG.integers(0,20)}",
        w=RNG.choice([1,4,8,16]),
        s=RNG.choice(["2.5GT/s", "5GT/s", "8GT/s", "16GT/s"]),
        p=RNG.integers(0, 7),
    )

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw").mkdir(exist_ok=True)
    (OUT_DIR / "processed").mkdir(exist_ok=True)

    # --- Generate work orders
    start_date = datetime(2025, 1, 1)
    wo_rows = []
    for i in range(N_WORK_ORDERS):
        wo_id = f"WO-{i:07d}"
        row = {
            "work_order_id": wo_id,
            "catalog_id": RNG.choice(CATALOGS),
            "supplier": RNG.choice(SUPPLIERS, p=np.array([1+0.02*i for i in range(len(SUPPLIERS))])/sum(1+0.02*i for i in range(len(SUPPLIERS)))),
            "device_type": RNG.choice(DEVICE_TYPES),
            "technician": RNG.choice(TECHS),
            "shift": RNG.choice(SHIFTS, p=[0.45, 0.35, 0.20]),
            "build_date": (start_date + timedelta(days=int(i/80))).date().isoformat(),
        }
        wo_rows.append(row)
    wo_df = pd.DataFrame(wo_rows)

    # --- Inject fingerprints & logs
    log_rows = []
    sw_rows = []
    failure_flags = []

    for idx in trange(len(wo_df), desc="Synthesizing WOs"):
        wo = wo_df.iloc[idx].to_dict()

        # Decide how many lines for this WO
        n_lines = max(3, int(RNG.normal(AVG_LINES_PER_WO, 3)))
        # Decide which fingerprints appear (0–3) with context bias
        available = list(FINGERPRINTS.keys())
        n_fp = RNG.choice([0,1,2,3], p=[0.35,0.35,0.22,0.08])
        chosen = []
        context_scores = []
        for fp_key in available:
            fp = FINGERPRINTS[fp_key]
            m = contextual_multiplier(fp, wo)
            context_scores.append(m)
        probs = np.array(context_scores) / np.sum(context_scores)
        if n_fp > 0:
            chosen = list(RNG.choice(available, size=n_fp, replace=False, p=probs))
        # Build baseline log lines (noise/info)
        base_ts = datetime.fromisoformat(wo["build_date"]) + timedelta(hours=int(RNG.integers(6, 18)))
        for j in range(n_lines):
            t = base_ts + timedelta(seconds=int(RNG.integers(0, 3600)))
            level = RNG.choice(["INFO","WARN","ERROR"], p=[0.65,0.20,0.15])
            msg = RNG.choice([
                f"step completed: {RNG.choice(['provision','flash','audit','qc'])} duration={RNG.integers(2,120)}s",
                f"checksum {rand_hex(8)} verified",
                f"invoking task {RNG.choice(['stage_image','install_driver','run_memtest','collect_telemetry'])}",
                f"device id {rand_hex(6)} detected",
            ])
            log_rows.append({"work_order_id": wo["work_order_id"], "ts": t.isoformat(), "level": level, "message": msg})

        # Add fingerprint lines (each adds 1–3 error/warn lines)
        fp_failure_contrib = 0.0
        for fp_key in chosen:
            fp = FINGERPRINTS[fp_key]
            k = RNG.integers(1,4)
            for _ in range(k):
                template = RNG.choice(fp["templates"])
                t = base_ts + timedelta(seconds=int(RNG.integers(0, 3600)))
                log_rows.append({"work_order_id": wo["work_order_id"], "ts": t.isoformat(), "level": RNG.choice(["ERROR","WARN"], p=[0.7,0.3]), "message": instantiate(template)})
            fp_failure_contrib += fp["weight"] * contextual_multiplier(fp, wo)

        # Stopworks note?
        if RNG.random() < STOPWORKS_COVERAGE:
            if chosen and RNG.random() > MISATTRIBUTION_RATE:
                # accurate-ish
                fp_key = RNG.choice(chosen)
            else:
                # misattribute or no fp present: random pick
                fp_key = RNG.choice(list(FINGERPRINTS.keys()))
            sw_text = RNG.choice(FINGERPRINTS[fp_key]["stopworks"])
            # 50% chance to add a normalized mapping
            if RNG.random() < 0.5:
                sw_rows.append({
                    "work_order_id": wo["work_order_id"],
                    "note_text": sw_text,
                    "norm_subsystem": fp_key.split("_")[0],  # crude normalization
                    "norm_root_cause": fp_key,
                })
            else:
                sw_rows.append({
                    "work_order_id": wo["work_order_id"],
                    "note_text": sw_text,
                    "norm_subsystem": "",
                    "norm_root_cause": "",
                })

        # Failure label
        p_fail = 1 - math.exp(-(BASE_FAILURE_RATE + 0.22*fp_failure_contrib))
        failure_flags.append(int(RNG.random() < p_fail))

    wo_df["failure_label"] = failure_flags

    # --- Save
    raw_dir = OUT_DIR / "raw"
    wo_df.to_csv(raw_dir / "work_orders.csv", index=False)
    pd.DataFrame(log_rows).sort_values(["work_order_id","ts"]).to_csv(raw_dir / "logs.csv", index=False)
    pd.DataFrame(sw_rows).to_csv(raw_dir / "stopworks.csv", index=False)

    print("Wrote:")
    print(" -", raw_dir / "work_orders.csv")
    print(" -", raw_dir / "logs.csv")
    print(" -", raw_dir / "stopworks.csv")

if __name__ == "__main__":
    main()

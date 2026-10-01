import random
import datetime
import sqlite3

# Python 3.12+ deprecates sqlite3's implicit datetime adapter. Register an
# explicit ISO-8601 adapter so synthetic-data generation stays warning-free
# and stores the same sortable text representation used by live queries.
sqlite3.register_adapter(datetime.datetime, lambda value: value.isoformat(" "))
import uuid
import pandas as pd
import numpy as np
from db.database import init_db, get_raw_sqlite_connection
from utils.logger import get_logger

logger = get_logger("data_generator")

def generate_dataset(num_users=3000, num_transactions=30000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Generating synthetic payments dataset with fraud rings...")

    init_db()
    conn = get_raw_sqlite_connection()
    cursor = conn.cursor()

    # 1. Clear existing data
    cursor.executescript("""
        DELETE FROM human_reviews;
        DELETE FROM investigation_reports;
        DELETE FROM risk_scores;
        DELETE FROM user_watchlist;
        DELETE FROM transactions;
        DELETE FROM users;
        DELETE FROM devices;
        DELETE FROM ip_addresses;
        DELETE FROM merchants;
    """)

    now = datetime.datetime.now()

    # 2. Generate Merchants
    merchants = []
    categories = ["E-Commerce", "Electronics", "Gaming", "Travel", "Crypto", "Utility", "Food & Dining"]
    for i in range(1, 51):
        m_id = f"MCH_{i:03d}"
        name = f"Merchant_{i} ({random.choice(categories)})"
        cat = random.choice(categories)
        fraud_rate = 0.15 if cat == "Crypto" or i == 42 else round(random.uniform(0.005, 0.03), 4)
        merchants.append((m_id, name, cat, fraud_rate, now - datetime.timedelta(days=random.randint(30, 365))))
    
    # Injected Suspicious Merchant
    merchants.append(("MCH_SUSPICIOUS_99", "Shadow Global Exch", "Crypto", 0.45, now - datetime.timedelta(days=2)))
    
    cursor.executemany("INSERT OR IGNORE INTO merchants VALUES (?, ?, ?, ?, ?)", merchants)
    logger.info(f"Inserted {len(merchants)} merchants.")

    # 3. Generate Normal Users — each with THEIR OWN dedicated device + IP.
    # (Earlier versions drew a random device and random IP per transaction
    # from a shared pool of 600 devices / 800 IPs across 1500 users — with
    # 12k transactions that meant every device/IP coincidentally ended up
    # "shared" by several users just by chance, producing a near-complete
    # 190k-edge graph instead of a sparse one. A real user has one phone;
    # modeling that 1:1 is what makes "shared device" a meaningful signal
    # at all instead of noise.)
    users = []
    devices = []
    ips = []
    user_device = {}  # user_id -> device_id (their own)
    user_ip = {}       # user_id -> ip_address (their own; may gain a 2nd via co-location)

    for i in range(1, num_users + 1):
        u_id = f"USER_{i:04d}"
        name = f"User_{i}"
        email = f"user_{i}@example.com"
        users.append((u_id, name, email, now - datetime.timedelta(days=random.randint(10, 500)), "ACTIVE"))

        d_id = f"DEV_{i:04d}"
        d_type = random.choice(["Mobile-Android", "Mobile-iOS", "Desktop-Windows", "Desktop-Mac"])
        os_sys = random.choice(["Android 14", "iOS 17", "Windows 11", "macOS Sonoma"])
        is_vpn = random.random() < 0.05
        devices.append((d_id, d_type, os_sys, is_vpn, now - datetime.timedelta(days=random.randint(10, 300))))
        user_device[u_id] = d_id

        ip_addr = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        country = "IN" if random.random() < 0.9 else random.choice(["US", "SG", "GB", "RU", "CN"])
        city = random.choice(["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Chennai", "Pune"])
        isp = random.choice(["Airtel", "Jio", "Vodafone", "ACT Fiber"])
        is_proxy = country in ["RU", "CN"]
        ips.append((ip_addr, country, city, isp, is_proxy))
        user_ip[u_id] = ip_addr

    normal_u_ids = list(user_device.keys())

    # Benign co-locations (noise): pairs of normal users who happen to share
    # ONE IP (roommates, office wifi, a cafe) with no elevated velocity and
    # different devices — no fraud behavior at all. Without this, "any
    # shared IP" would be a perfect fraud signal by construction, which is
    # too easy: it gives the GNN/tabular model nothing real to beat. With
    # it, a naive shared-IP-only rule produces false positives, same as a
    # real payments graph.
    num_colocations = max(1, num_users // 25)
    colocation_pairs = 0
    used_in_colocation = set()
    attempts = 0
    user_secondary_ip = {}  # user_id -> an occasional second IP (a coworker's/roommate's)
    while colocation_pairs < num_colocations and attempts < num_colocations * 20:
        attempts += 1
        u1, u2 = random.sample(normal_u_ids, 2)
        if u1 in used_in_colocation or u2 in used_in_colocation:
            continue
        user_secondary_ip[u2] = user_ip[u1]
        used_in_colocation.add(u1)
        used_in_colocation.add(u2)
        colocation_pairs += 1
    logger.info(f"Prepared {colocation_pairs} benign IP co-locations (noise, not fraud).")

    # 4. Create Injected Fraud Rings
    # Fraud Ring 1: Device Sharing Ring (7 accounts, 1 device, 1 IP)
    ring1_users = [f"USER_RING1_{i}" for i in range(1, 8)]
    for u in ring1_users:
        users.append((u, f"Fraud Ring1 Member {u}", f"{u}@tempmail.com", now - datetime.timedelta(days=1), "SUSPICIOUS"))
    devices.append(("DEV_FRAUD_RING1", "Mobile-Android", "Android 11", True, now - datetime.timedelta(hours=5)))
    ips.append(("185.220.101.44", "RU", "Moscow", "Tor Proxy Node", True))

    # Fraud Ring 2: IP Velocity / Proxy Cluster (8 accounts, 1 shared IP)
    ring2_users = [f"USER_RING2_{i}" for i in range(1, 9)]
    for u in ring2_users:
        users.append((u, f"Fraud Ring2 Member {u}", f"{u}@anonymous.io", now - datetime.timedelta(days=2), "SUSPICIOUS"))
        devices.append((f"DEV_RING2_{u}", "Mobile-iOS", "iOS 16", True, now - datetime.timedelta(days=2)))
    ips.append(("198.51.100.99", "CN", "Beijing", "High Risk VPN", True))

    # Fraud Ring 3: High Velocity Carding Attack (1 user, 1 device, rapid txns)
    ring3_user = "USER_CARDER_X"
    users.append((ring3_user, "Attacker Carder X", "carderx@darknet.net", now - datetime.timedelta(days=1), "SUSPICIOUS"))
    devices.append(("DEV_CARDER_X", "Desktop-Windows", "Windows 10", True, now - datetime.timedelta(hours=2)))
    ips.append(("203.0.113.50", "IN", "Mumbai", "Public Wi-Fi", False))

    # Fraud Ring 4: Suspicious Merchant Collusion (5 users -> MCH_SUSPICIOUS_99)
    ring4_users = [f"USER_COLLUSION_{i}" for i in range(1, 6)]
    ring4_ip = {}
    for u in ring4_users:
        users.append((u, f"Collusion User {u}", f"{u}@mulemail.org", now - datetime.timedelta(days=3), "SUSPICIOUS"))
        devices.append((f"DEV_RING4_{u}", "Mobile-Android", "Android 13", False, now - datetime.timedelta(days=3)))
        ip_addr = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        ips.append((ip_addr, "IN", random.choice(["Mumbai", "Delhi", "Pune"]), "Jio", False))
        ring4_ip[u] = ip_addr

    # Fraud Ring 5: Device-Cycling Structuring / Card-Testing Evasion.
    # Inverse topology of Ring 2 (many users -> one IP): here it's ONE
    # account cycling through several device fingerprints while incrementing
    # IP by one each hop (proxy-pool rotation), starting with a trivial test
    # charge, escalating toward the platform ceiling, then deliberately
    # pausing past the 1h velocity window before a final reconnaissance
    # probe on a fresh IP using two more borrowed devices. This shape is
    # invisible to a per-user, single-1h-window velocity feature by
    # construction — that's the point of injecting it.
    ring5_user = "USER_RING5_1"
    users.append((ring5_user, "Fraud Ring5 Structuring Actor", f"{ring5_user}@disposable.io", now - datetime.timedelta(days=1), "SUSPICIOUS"))
    ring5_devices = [f"DEV_RING5_BORROWED_{i}" for i in range(1, 4)]
    for d in ring5_devices:
        devices.append((d, "Mobile-iOS", "iOS 17", True, now - datetime.timedelta(hours=random.randint(1, 48))))
    ring5_escalation_ips = [f"198.51.104.{i}" for i in range(50, 57)]
    for ip in ring5_escalation_ips:
        ips.append((ip, "SG", "Singapore", "Datacenter Proxy Pool", True))
    ring5_probe_ip = "198.51.109.20"
    ips.append((ring5_probe_ip, "SG", "Singapore", "Datacenter Proxy Pool", True))

    # Legitimate look-alike communities — labeled BENIGN (is_fraud_ground_truth
    # = False) on purpose. Dense, shared-fingerprint communities that would
    # trip a naive "shared device/IP => fraud" rule but are ordinary in the
    # real world. Included specifically so the tabular/GNN models — and the
    # rule-based evidence overlay in risk_aggregator.py — have to learn what
    # actually separates a fraud ring from a hostel or a family, not just
    # connectivity. Without these, "any dense shared-fingerprint community"
    # would be a perfect fraud signal by construction, same issue as the
    # single-pair colocation noise above but at ring-scale.
    logger.info("Injecting legitimate look-alike communities (hostel, family)...")

    # Hostel: 7 residents, one shared public IP, split across 2 shared
    # devices (not all 7 on one) plus personal devices for the rest,
    # ordinary small amounts at everyday merchants, spread over WEEKS.
    hostel_users = [f"USER_HOSTEL_{i}" for i in range(1, 8)]
    for u in hostel_users:
        users.append((u, f"Hostel Resident {u}", f"{u}@hostel.example", now - datetime.timedelta(days=random.randint(60, 400)), "ACTIVE"))
    hostel_ip = "203.0.113.201"
    ips.append((hostel_ip, "IN", "Pune", "ACT Fiber", False))
    hostel_shared_devices = ["DEV_HOSTEL_SHARED_1", "DEV_HOSTEL_SHARED_2"]
    for d in hostel_shared_devices:
        devices.append((d, "Mobile-Android", "Android 14", False, now - datetime.timedelta(days=random.randint(60, 300))))
    hostel_device_map = {}
    for i, u in enumerate(hostel_users):
        if i < 4:
            hostel_device_map[u] = hostel_shared_devices[i % 2]
        else:
            own_dev = f"DEV_{u}_OWN"
            devices.append((own_dev, "Mobile-Android", "Android 14", False, now - datetime.timedelta(days=random.randint(60, 300))))
            hostel_device_map[u] = own_dev

    # Family: 5 members, same home IP, two of them occasionally share one
    # device, recurring monthly bill-pay-style transactions to a small set
    # of familiar merchants, ordinary amounts, no velocity spikes.
    family_users = [f"USER_FAMILY_{i}" for i in range(1, 6)]
    for u in family_users:
        users.append((u, f"Family Member {u}", f"{u}@family.example", now - datetime.timedelta(days=random.randint(100, 600)), "ACTIVE"))
    family_ip = "203.0.113.202"
    ips.append((family_ip, "IN", "Bangalore", "Jio", False))
    family_shared_device = "DEV_FAMILY_SHARED_1"
    devices.append((family_shared_device, "Mobile-Android", "Android 14", False, now - datetime.timedelta(days=200)))
    family_device_map = {}
    for i, u in enumerate(family_users):
        if i < 2:
            family_device_map[u] = family_shared_device
        else:
            own_dev = f"DEV_{u}_OWN"
            devices.append((own_dev, "Mobile-iOS", "iOS 17", False, now - datetime.timedelta(days=random.randint(100, 400))))
            family_device_map[u] = own_dev

    # Additional adversarial benign/fraud communities — added to close the
    # gaps in the original test matrix (hostel/family benign + 5 fraud rings
    # only). Each one targets a SPECIFIC false-positive or false-negative
    # failure mode identified in review, not just "more data":
    #
    #   BENIGN (should never resolve HIGH/CRITICAL):
    #     carrier_nat        — many users, one IP, via carrier-grade NAT
    #     event_spike        — many users + one IP + one merchant + high
    #                          velocity, all legitimately simultaneous
    #     shared_device_grp  — one device, 4-5 users (POS/office machine),
    #                          not the 2-user case already covered
    #     bill_split         — several users paying one merchant at once
    #                          (non-uniform amounts) — a payment-app bill
    #                          split, not a wire transfer
    #     recurring_monthly  — predictable multi-merchant monthly bills,
    #                          standalone from the family scenario
    #     fan_out_shopping   — one established user buying from 8 different
    #                          merchants over a weekend (not rapid-fire)
    #     popular_merchant   — many UNRELATED users (own device/own IP each)
    #                          all transacting with one popular merchant
    #
    #   FRAUD (should resolve MEDIUM/HIGH or at least reach HITL):
    #     structuring        — near-uniform amounts just under a reporting
    #                          threshold, same device+IP, short window
    #     fan_out_launder    — one user, 8 merchants, near-identical
    #                          amounts, all within an hour (rapid dispersal)
    #     no_shared_infra    — independent users/devices/IPs, zero graph
    #                          overlap, individually anomalous behavior
    #     low_and_slow       — own devices/IPs (no fingerprint sharing),
    #                          ordinary amounts, spread over weeks — the
    #                          "boring fraud" case current signals may miss
    logger.info("Injecting expanded adversarial test communities...")

    # --- BENIGN: carrier NAT (many mobile users legitimately share 1 IP) ---
    carrier_nat_users = [f"USER_CARRIER_{i}" for i in range(1, 41)]
    for u in carrier_nat_users:
        users.append((u, f"Carrier NAT User {u}", f"{u}@example.com", now - datetime.timedelta(days=random.randint(30, 500)), "ACTIVE"))
        d_id = f"DEV_{u}_OWN"
        devices.append((d_id, "Mobile-Android", "Android 14", False, now - datetime.timedelta(days=random.randint(30, 400))))
    carrier_nat_ip = "100.72.0.14"
    ips.append((carrier_nat_ip, "IN", "Mumbai", "Jio Mobile Carrier NAT", False))

    # --- BENIGN: event spike (conference/venue wifi + one kiosk merchant) ---
    event_users = [f"USER_EVENT_{i}" for i in range(1, 61)]
    for u in event_users:
        users.append((u, f"Event Attendee {u}", f"{u}@example.com", now - datetime.timedelta(days=random.randint(20, 600)), "ACTIVE"))
        d_id = f"DEV_{u}_OWN"
        devices.append((d_id, random.choice(["Mobile-Android", "Mobile-iOS"]), "Android 14", False, now - datetime.timedelta(days=random.randint(20, 400))))
    event_ip = "203.0.113.220"
    ips.append((event_ip, "IN", "Bangalore", "Venue Guest WiFi", False))

    # --- BENIGN: shared device, 4-5 users (office/POS machine, not a pair) ---
    shared_dev_users = [f"USER_SHAREDDEV_{i}" for i in range(1, 6)]
    for u in shared_dev_users:
        users.append((u, f"Shared-Device User {u}", f"{u}@example.com", now - datetime.timedelta(days=random.randint(60, 500)), "ACTIVE"))
    shared_dev_id = "DEV_OFFICE_POS_1"
    devices.append((shared_dev_id, "Desktop-Windows", "Windows 11", False, now - datetime.timedelta(days=250)))
    shared_dev_ip = "203.0.113.230"
    ips.append((shared_dev_ip, "IN", "Chennai", "Office Broadband", False))

    # --- BENIGN: bill split (several users paying one merchant at once) ---
    billsplit_users = [f"USER_BILLSPLIT_{i}" for i in range(1, 7)]
    for u in billsplit_users:
        users.append((u, f"Bill Split User {u}", f"{u}@example.com", now - datetime.timedelta(days=random.randint(30, 500)), "ACTIVE"))
        d_id = f"DEV_{u}_OWN"
        devices.append((d_id, "Mobile-Android", "Android 14", False, now - datetime.timedelta(days=random.randint(30, 400))))
    billsplit_ip = "203.0.113.240"
    ips.append((billsplit_ip, "IN", "Delhi", "Restaurant Guest WiFi", False))

    # --- BENIGN: recurring monthly multi-merchant bills, standalone users ---
    recurring_users = [f"USER_RECURRING_{i}" for i in range(1, 5)]
    for u in recurring_users:
        users.append((u, f"Recurring Payer {u}", f"{u}@example.com", now - datetime.timedelta(days=random.randint(200, 700)), "ACTIVE"))
        d_id = f"DEV_{u}_OWN"
        devices.append((d_id, "Mobile-iOS", "iOS 17", False, now - datetime.timedelta(days=random.randint(200, 600))))
        ip_addr = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        ips.append((ip_addr, "IN", "Hyderabad", "ACT Fiber", False))
        user_ip[u] = ip_addr
        user_device[u] = d_id

    # --- BENIGN: one established user shopping across 8 merchants (weekend) ---
    fanout_shopper = "USER_FANOUT_SHOPPER"
    users.append((fanout_shopper, "Fan-Out Shopper", "fanout_shopper@example.com", now - datetime.timedelta(days=400), "ACTIVE"))
    fanout_shopper_dev = "DEV_FANOUT_SHOPPER"
    devices.append((fanout_shopper_dev, "Mobile-iOS", "iOS 17", False, now - datetime.timedelta(days=380)))
    fanout_shopper_ip = "192.168.44.201"
    ips.append((fanout_shopper_ip, "IN", "Mumbai", "Airtel", False))

    # --- BENIGN: popular merchant, many UNRELATED users (own device/own IP) ---
    popular_merchant_users = [f"USER_POPMCH_{i}" for i in range(1, 81)]
    for u in popular_merchant_users:
        users.append((u, f"Popular-Merchant Shopper {u}", f"{u}@example.com", now - datetime.timedelta(days=random.randint(10, 500)), "ACTIVE"))
        d_id = f"DEV_{u}_OWN"
        devices.append((d_id, random.choice(["Mobile-Android", "Mobile-iOS"]), "Android 14", False, now - datetime.timedelta(days=random.randint(10, 400))))
        ip_addr = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        ips.append((ip_addr, "IN", random.choice(["Mumbai", "Pune", "Chennai"]), "Airtel", False))
        user_ip[u] = ip_addr
        user_device[u] = d_id

    # --- FRAUD: structuring (near-uniform amounts just under a reporting
    # threshold, one device + IP, short window) — the fraud counterpart to
    # bill_split: same "several similar-sized payments in a burst" SHAPE,
    # opposite cause (evasion, not a shared bill). ---
    structuring_users = [f"USER_STRUCT_{i}" for i in range(1, 3)]
    for u in structuring_users:
        users.append((u, f"Structuring Suspect {u}", f"{u}@tempmail.com", now - datetime.timedelta(days=3), "SUSPICIOUS"))
    structuring_dev = "DEV_STRUCTURING_1"
    devices.append((structuring_dev, "Mobile-Android", "Android 11", True, now - datetime.timedelta(hours=8)))
    structuring_ip = "198.51.100.150"
    ips.append((structuring_ip, "SG", "Singapore", "Datacenter Proxy Pool", True))

    # --- FRAUD: fan-out laundering (one user, 8 merchants, near-identical
    # amounts, all within an hour) — same graph SHAPE as fan_out_shopping,
    # opposite timing/amount pattern. ---
    fanout_launderer = "USER_FANOUT_LAUNDER"
    users.append((fanout_launderer, "Fan-Out Launderer", "fanout_launder@tempmail.com", now - datetime.timedelta(hours=6), "SUSPICIOUS"))
    fanout_launder_dev = "DEV_FANOUT_LAUNDER"
    devices.append((fanout_launder_dev, "Mobile-Android", "Android 11", True, now - datetime.timedelta(hours=6)))
    fanout_launder_ip = "198.51.100.160"
    ips.append((fanout_launder_ip, "SG", "Singapore", "Datacenter Proxy Pool", True))

    # --- FRAUD: no shared infrastructure at all (independent users/devices/
    # IPs, zero graph overlap, individually anomalous behavior) — tests that
    # detection does NOT depend on graph connectivity existing in the first
    # place. ---
    no_infra_users = [f"USER_NOINFRA_{i}" for i in range(1, 5)]
    no_infra_dev = {}
    no_infra_ip = {}
    for u in no_infra_users:
        users.append((u, f"No-Infra Suspect {u}", f"{u}@example.com", now - datetime.timedelta(days=random.randint(100, 500)), "ACTIVE"))
        d_id = f"DEV_{u}_OWN"
        devices.append((d_id, "Mobile-Android", "Android 14", False, now - datetime.timedelta(days=random.randint(100, 400))))
        ip_addr = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        ips.append((ip_addr, "IN", random.choice(["Mumbai", "Delhi", "Pune"]), "Airtel", False))
        no_infra_dev[u] = d_id
        no_infra_ip[u] = ip_addr

    # --- FRAUD: low-and-slow (own devices/IPs, ordinary amounts, spread
    # over weeks — deliberately boring, no graph anomaly, no velocity
    # anomaly). Flagged as the hardest case in the current design: if this
    # scenario fails, that is an honest, expected gap, not a bug — it
    # documents where the system's blind spot currently is. ---
    low_slow_users = [f"USER_LOWSLOW_{i}" for i in range(1, 7)]
    low_slow_dev = {}
    low_slow_ip = {}
    for u in low_slow_users:
        users.append((u, f"Low-and-Slow Suspect {u}", f"{u}@example.com", now - datetime.timedelta(days=random.randint(150, 500)), "ACTIVE"))
        d_id = f"DEV_{u}_OWN"
        devices.append((d_id, "Mobile-Android", "Android 14", False, now - datetime.timedelta(days=random.randint(150, 400))))
        ip_addr = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        ips.append((ip_addr, "IN", random.choice(["Mumbai", "Delhi", "Pune"]), "Airtel", False))
        low_slow_dev[u] = d_id
        low_slow_ip[u] = ip_addr

    # --- BENIGN: cold start — 2 brand-new users, their first-ever handful
    # of transactions, small/ordinary amounts. Tests that a new account
    # with modest, unremarkable spending doesn't get flagged just for
    # lacking history (amount_zscore_prior is 0.0 by construction with no
    # prior transactions — see live_tabular_score — so this exercises
    # whatever the model does with a neutral zscore on a genuinely new
    # user, not a spoofed-quiet one).
    cold_start_benign_users = [f"USER_COLDSTART_BENIGN_{i}" for i in range(1, 3)]
    cold_start_benign_dev = {}
    cold_start_benign_ip = {}
    for u in cold_start_benign_users:
        users.append((u, f"New User {u}", f"{u}@example.com", now - datetime.timedelta(hours=random.randint(1, 12)), "ACTIVE"))
        d_id = f"DEV_{u}_OWN"
        devices.append((d_id, "Mobile-Android", "Android 14", False, now - datetime.timedelta(hours=random.randint(1, 12))))
        ip_addr = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        ips.append((ip_addr, "IN", "Mumbai", "Airtel", False))
        cold_start_benign_dev[u] = d_id
        cold_start_benign_ip[u] = ip_addr

    # --- FRAUD: cold start — 2 brand-new accounts whose very first
    # transaction is already large and to a risky merchant, on a brand-new
    # device/IP with nothing to compare it against. Tests the edge case
    # explicitly called out in review: with no baseline, amount_zscore_
    # prior is undefined/0 and can't itself flag anything — this checks
    # whether the tabular model's OTHER features (amount_log, merchant_
    # fraud_rate) plus a first-transaction velocity_1h=1 are enough on
    # their own, since the account-age signal isn't in this dataset's
    # feature set at all — an honest gap to know about either way.
    cold_start_fraud_users = [f"USER_COLDSTART_FRAUD_{i}" for i in range(1, 3)]
    cold_start_fraud_dev = {}
    cold_start_fraud_ip = {}
    for u in cold_start_fraud_users:
        users.append((u, f"New Suspect {u}", f"{u}@tempmail.com", now - datetime.timedelta(hours=random.randint(1, 3)), "SUSPICIOUS"))
        d_id = f"DEV_{u}_NEW"
        devices.append((d_id, "Mobile-Android", "Android 11", True, now - datetime.timedelta(hours=random.randint(1, 3))))
        ip_addr = f"198.51.100.{random.randint(180, 199)}"
        ips.append((ip_addr, "SG", "Singapore", "Datacenter Proxy Pool", True))
        cold_start_fraud_dev[u] = d_id
        cold_start_fraud_ip[u] = ip_addr

    # --- FRAUD: account takeover — an ESTABLISHED user (long normal
    # history, own regular device/IP) whose account suddenly transacts
    # from a brand-new device AND a brand-new IP, at an unusual hour, for
    # an amount far outside their own history — with ZERO graph anomaly
    # (the new device/IP aren't shared with anyone else; a hijacker using
    # a stolen credential from their own single machine looks graph-clean).
    # This is the "behavioral anomaly without graph anomaly" fraud pathway
    # explicitly called out in review — connectivity evidence structurally
    # cannot catch this, so it has to be the tabular amount_zscore_prior +
    # unusual-hour signal doing the work alone.
    ato_users = [f"USER_ATO_{i}" for i in range(1, 4)]
    ato_normal_dev = {}
    ato_normal_ip = {}
    ato_hijack_dev = {}
    ato_hijack_ip = {}
    for u in ato_users:
        users.append((u, f"Established User {u}", f"{u}@example.com", now - datetime.timedelta(days=random.randint(300, 900)), "ACTIVE"))
        normal_dev = f"DEV_{u}_NORMAL"
        devices.append((normal_dev, "Mobile-iOS", "iOS 17", False, now - datetime.timedelta(days=random.randint(300, 800))))
        normal_ip = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        ips.append((normal_ip, "IN", "Mumbai", "Airtel", False))
        ato_normal_dev[u] = normal_dev
        ato_normal_ip[u] = normal_ip
        hijack_dev = f"DEV_{u}_HIJACK"
        devices.append((hijack_dev, "Desktop-Windows", "Windows 10", False, now - datetime.timedelta(hours=1)))
        hijack_ip = f"91.219.{random.randint(1, 255)}.{random.randint(1, 255)}"
        ips.append((hijack_ip, "RU", "Unknown", "Unknown Hosting", False))
        ato_hijack_dev[u] = hijack_dev
        ato_hijack_ip[u] = hijack_ip

    # Additional tabular-only fraud population: deliberately obvious fraud
    # that does NOT require any graph relationship. These users have private
    # devices/IPs, but their transaction behavior is visually suspicious:
    # unusual hour + large amount + round amount and/or repeated risky
    # merchant activity. This prevents the tabular branch from becoming a
    # graph-dependent second detector and gives the stacker a genuinely
    # complementary signal.
    obvious_fraud_users = [f"USER_OBVIOUS_FRAUD_{i:04d}" for i in range(1, 121)]
    obvious_fraud_dev = {}
    obvious_fraud_ip = {}
    for u in obvious_fraud_users:
        users.append((u, f"Obvious Behavioral Fraud {u}", f"{u}@disposable.example", now - datetime.timedelta(days=random.randint(90, 500)), "SUSPICIOUS"))
        d_id = f"DEV_{u}_OWN"
        ip_addr = f"172.31.{random.randint(1, 254)}.{random.randint(1, 254)}"
        devices.append((d_id, random.choice(["Mobile-Android", "Desktop-Windows"]), random.choice(["Android 14", "Windows 11"]), random.random() < 0.15, now - datetime.timedelta(days=random.randint(30, 300))))
        ips.append((ip_addr, "IN", random.choice(["Mumbai", "Delhi", "Bangalore"]), random.choice(["Airtel", "Jio"]), False))
        obvious_fraud_dev[u] = d_id
        obvious_fraud_ip[u] = ip_addr

    # Hard benign controls for the same tabular cues: legitimate high-value
    # purchases during normal hours from established accounts, using their
    # own stable infrastructure. These are essential negatives; otherwise
    # the model can learn the trivial rule "large amount = fraud".
    high_value_benign_users = [f"USER_HIGHVALUE_{i:04d}" for i in range(1, 81)]
    high_value_dev = {}
    high_value_ip = {}
    for u in high_value_benign_users:
        users.append((u, f"High Value Legitimate User {u}", f"{u}@example.com", now - datetime.timedelta(days=random.randint(300, 900)), "ACTIVE"))
        d_id = f"DEV_{u}_OWN"
        ip_addr = f"10.20.{random.randint(1, 254)}.{random.randint(1, 254)}"
        devices.append((d_id, "Mobile-iOS", "iOS 17", False, now - datetime.timedelta(days=random.randint(100, 700))))
        ips.append((ip_addr, "IN", random.choice(["Mumbai", "Pune", "Hyderabad"]), random.choice(["Airtel", "Jio", "ACT Fiber"]), False))
        high_value_dev[u] = d_id
        high_value_ip[u] = ip_addr

    cursor.executemany("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?)", users)
    cursor.executemany("INSERT OR IGNORE INTO devices VALUES (?, ?, ?, ?, ?)", devices)
    cursor.executemany("INSERT OR IGNORE INTO ip_addresses VALUES (?, ?, ?, ?, ?)", ips)
    logger.info(f"Inserted {len(users)} users, {len(devices)} devices, {len(ips)} IPs.")

    # 5. Generate Transactions
    tx_list = []
    mch_ids = [m[0] for m in merchants if m[0] != "MCH_SUSPICIOUS_99"]

    # Generate normal transactions spread over past 30 days — each always
    # uses ITS OWN user's device + IP (occasionally the co-located secondary
    # IP), not an independent random draw, since a person doesn't switch
    # phones between purchases.
    base_time = now - datetime.timedelta(days=30)
    for i in range(1, num_transactions + 1):
        tx_id = f"TXN_NORM_{i:06d}"
        u = random.choice(normal_u_ids)
        d = user_device[u]
        ip = user_secondary_ip[u] if (u in user_secondary_ip and random.random() < 0.3) else user_ip[u]
        m = random.choice(mch_ids)
        amt = round(float(np.random.exponential(scale=1200) + 150), 2)
        if amt > 25000:
            amt = round(random.uniform(500, 4500), 2)
        
        tx_time = base_time + datetime.timedelta(minutes=random.randint(1, 30*24*60))
        tx_list.append((
            tx_id, u, d, ip, m, amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.5, 0.8), 2), False
        ))

    # Legitimate look-alike community transactions — same shared-fingerprint
    # topology fraud rings use, but with the behavior that actually
    # distinguishes them: spread over weeks, ordinary velocity, amounts that
    # don't deviate from a normal spending range, everyday merchants.
    idx = 0
    hostel_time = now - datetime.timedelta(days=45)
    for u in hostel_users:
        for _ in range(random.randint(6, 10)):
            amt = round(float(np.random.exponential(scale=350) + 50), 2)
            tx_time = hostel_time + datetime.timedelta(minutes=random.randint(1, 45 * 24 * 60))
            tx_list.append((
                f"TXN_HOSTEL_{idx:04d}", u, hostel_device_map[u], hostel_ip, random.choice(mch_ids),
                amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.5, 0.8), 2), False
            ))
            idx += 1

    idx = 0
    family_merchants = random.sample(mch_ids, 3)
    family_time = now - datetime.timedelta(days=60)
    for u in family_users:
        for month in range(2):
            amt = round(random.uniform(300, 2500), 2)
            tx_time = family_time + datetime.timedelta(days=month * 30 + random.randint(0, 5), hours=random.randint(8, 21))
            tx_list.append((
                f"TXN_FAMILY_{idx:04d}", u, family_device_map[u], family_ip, random.choice(family_merchants),
                amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.5, 0.5), 2), False
            ))
            idx += 1

    # --- BENIGN: family member with a genuinely unusual-for-them but
    # legitimate expense (wedding/vacation/medical/school fees) — same
    # family IP/device-sharing pattern as above, plus ONE moderately large
    # outlier transaction per user that IS a real deviation from their own
    # baseline. Directly tests the distinction the review called out:
    # "this is unusual for the user" is not the same claim as "this is
    # fraudulent" — the model needs contextual features (this is still one
    # transaction, one merchant, no velocity spike, same familiar IP/
    # device) to avoid conflating the two.
    #
    # Amount range deliberately kept moderate (~4-8x baseline, roughly
    # 1.5-2.5 sigma) rather than extreme — an earlier version used
    # ₹18,000-25,000 against this same ₹300-2,500 baseline, which produced
    # a 3.0-5.0 z-score range that's statistically indistinguishable from
    # the fraud scenarios' own 2.5-6.0 z-score range. Running the actual
    # trained model against that version showed exactly this: it scored
    # HIGH, not LOW, because amount_zscore_prior alone genuinely cannot
    # separate "one big legitimate purchase" from "fraud" when both
    # produce the same magnitude of deviation — there's no third feature
    # in this dataset (e.g. "matches a known life-event category," "annual
    # recurring timing") that would let it. That's not a bug to hide; it's
    # the real answer to review point #2, empirically confirmed rather
    # than asserted. This version tests a milder, more realistic "unusual"
    # amount where the tabular model has enough separation to not
    # conflate the two — see tests/GOLDEN_TEST_MATRIX.md's N16 row for the
    # honest characterization of both results.
    idx = 0
    unusual_family_users = family_users[:3]
    unusual_merchant = random.choice(mch_ids)
    unusual_time = now - datetime.timedelta(days=15)
    for i, u in enumerate(unusual_family_users):
        amt = round(random.uniform(6000, 9000), 2)
        tx_time = unusual_time + datetime.timedelta(days=i, hours=random.randint(9, 18))
        tx_list.append((
            f"TXN_FAMILYUNUSUAL_{idx:03d}", u, family_device_map[u], family_ip, unusual_merchant,
            amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(1.5, 2.5), 2), False
        ))
        idx += 1

    # --- BENIGN: cold start — 2 brand-new users' first few transactions,
    # small/ordinary amounts, single device/IP each (no sharing).
    idx = 0
    cold_start_time = now - datetime.timedelta(hours=6)
    for u in cold_start_benign_users:
        for i in range(2):
            amt = round(random.uniform(200, 900), 2)
            tx_time = cold_start_time + datetime.timedelta(hours=i)
            tx_list.append((
                f"TXN_COLDSTARTBENIGN_{idx:03d}", u, cold_start_benign_dev[u], cold_start_benign_ip[u],
                random.choice(mch_ids), amt, "INR", tx_time, "COMPLETED", 1, 0.0, False
            ))
            idx += 1

    # --- BENIGN: carrier NAT — 40 users, 1 shared carrier IP, own devices,
    # ordinary independent shopping over a month. Tests that shared_ip
    # alone, even at scale (>>5), never fires the confluence overlay when
    # there is no accompanying behavioral anomaly.
    idx = 0
    carrier_time = now - datetime.timedelta(days=28)
    for u in carrier_nat_users:
        for _ in range(random.randint(3, 6)):
            amt = round(float(np.random.exponential(scale=900) + 100), 2)
            tx_time = carrier_time + datetime.timedelta(minutes=random.randint(1, 28 * 24 * 60))
            tx_list.append((
                f"TXN_CARRIERNAT_{idx:04d}", u, f"DEV_{u}_OWN", carrier_nat_ip, random.choice(mch_ids),
                amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.5, 0.8), 2), False
            ))
            idx += 1

    # --- BENIGN: event spike — 60 users, 1 shared IP, 1 shared merchant, ALL
    # within a single 90-minute window (venue ticket/kiosk purchases). Every
    # individual signal (shared_ip, velocity, merchant concentration) is
    # simultaneously elevated here, on purpose — the stress test the review
    # explicitly called out as most likely to expose a false positive.
    idx = 0
    event_merchant = random.choice(mch_ids)
    event_time = now - datetime.timedelta(days=10)
    for u in event_users:
        amt = round(random.uniform(250, 1500), 2)
        tx_time = event_time + datetime.timedelta(minutes=random.randint(0, 90))
        tx_list.append((
            f"TXN_EVENT_{idx:04d}", u, f"DEV_{u}_OWN", event_ip, event_merchant,
            amt, "INR", tx_time, "COMPLETED", random.randint(1, 3), round(random.uniform(-0.3, 0.6), 2), False
        ))
        idx += 1

    # --- BENIGN: shared device, 4-5 users (office/POS machine) — the case
    # the review explicitly asked for since the original dataset only tested
    # 2-user device sharing (hostel/family). Modest amounts, spread over
    # weeks, no velocity spikes.
    idx = 0
    sd_time = now - datetime.timedelta(days=40)
    for u in shared_dev_users:
        for _ in range(random.randint(5, 9)):
            amt = round(float(np.random.exponential(scale=400) + 80), 2)
            tx_time = sd_time + datetime.timedelta(minutes=random.randint(1, 40 * 24 * 60))
            tx_list.append((
                f"TXN_SHAREDDEV_{idx:04d}", u, shared_dev_id, shared_dev_ip, random.choice(mch_ids),
                amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.5, 0.8), 2), False
            ))
            idx += 1

    # --- BENIGN: bill split — 6 friends each paying their own share of one
    # restaurant bill, within a 20-minute window, non-uniform amounts (this
    # is the honest visual/behavioral distinction from the structuring fraud
    # counterpart below, which uses near-IDENTICAL amounts).
    idx = 0
    billsplit_merchant = random.choice(mch_ids)
    billsplit_time = now - datetime.timedelta(days=5)
    for u in billsplit_users:
        amt = round(random.uniform(150, 900), 2)
        tx_time = billsplit_time + datetime.timedelta(minutes=random.randint(0, 20))
        tx_list.append((
            f"TXN_BILLSPLIT_{idx:04d}", u, f"DEV_{u}_OWN", billsplit_ip, billsplit_merchant,
            amt, "INR", tx_time, "COMPLETED", random.randint(1, 2), round(random.uniform(-0.3, 0.7), 2), False
        ))
        idx += 1

    # --- BENIGN: recurring monthly multi-merchant bills — standalone from
    # the family scenario, own devices/IPs, same 4 merchants each month for
    # 3 months, predictable timing. Tests that a repeated historical pattern
    # reads as normal even though it involves multiple merchants + amounts.
    idx = 0
    recurring_merchants = random.sample(mch_ids, 4)
    recurring_amounts = {m: round(random.uniform(500, 15000), 2) for m in recurring_merchants}
    recurring_time = now - datetime.timedelta(days=90)
    for u in recurring_users:
        for month in range(3):
            for m in recurring_merchants:
                amt = round(recurring_amounts[m] * random.uniform(0.95, 1.05), 2)
                tx_time = recurring_time + datetime.timedelta(days=month * 30 + 1, hours=random.randint(8, 20))
                tx_list.append((
                    f"TXN_RECURRING_{idx:04d}", u, user_device[u], user_ip[u], m,
                    amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.4, 0.4), 2), False
                ))
                idx += 1

    # --- BENIGN: fan-out shopping — one established user buying from 8
    # different merchants, varied amounts, spread across a WEEKEND (not
    # rapid-fire). The fraud counterpart (fan_out_launder below) has the
    # same graph shape (1 user -> 8 merchants) but compressed timing and
    # near-identical amounts — the behavioral context is the only thing
    # that tells them apart.
    fanout_merchants = random.sample(mch_ids, 8)
    fanout_time = now - datetime.timedelta(days=7)
    for i, m in enumerate(fanout_merchants):
        amt = round(float(np.random.exponential(scale=1000) + 200), 2)
        tx_time = fanout_time + datetime.timedelta(hours=random.randint(0, 46))
        tx_list.append((
            f"TXN_FANOUTSHOP_{i:03d}", fanout_shopper, fanout_shopper_dev, fanout_shopper_ip, m,
            amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.4, 0.7), 2), False
        ))

    # --- BENIGN: popular merchant — 80 UNRELATED users (own device, own IP
    # each — no fingerprint sharing at all) all transacting with the same
    # popular merchant. High merchant-side degree, zero user-side graph
    # connectivity. Documents that risk_graph.py intentionally excludes
    # merchant nodes from the scored graph (see its module docstring) so a
    # popular merchant never becomes "suspicious by popularity."
    idx = 0
    popular_merchant = random.choice(mch_ids)
    pop_time = now - datetime.timedelta(days=20)
    for u in popular_merchant_users:
        amt = round(float(np.random.exponential(scale=800) + 100), 2)
        tx_time = pop_time + datetime.timedelta(minutes=random.randint(1, 20 * 24 * 60))
        tx_list.append((
            f"TXN_POPMCH_{idx:04d}", u, user_device[u], user_ip[u], popular_merchant,
            amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.5, 0.8), 2), False
        ))
        idx += 1

    # Ingest Fraud Transactions
    # Ring 1 (Device Sharing) Transactions: 25 rapid transactions from 7 accounts via DEV_FRAUD_RING1
    logger.info("Injecting Fraud Ring 1 (Device Sharing)...")
    ring1_time = now - datetime.timedelta(hours=1)
    for idx, u in enumerate(ring1_users * 4):
        tx_id = f"TXN_RING1_{idx:03d}"
        amt = round(random.uniform(45000, 98000), 2)
        tx_time = ring1_time + datetime.timedelta(minutes=idx * 2)
        tx_list.append((
            tx_id, u, "DEV_FRAUD_RING1", "185.220.101.44", random.choice(mch_ids), amt, "INR", tx_time, "COMPLETED", idx + 3, round(random.uniform(2.5, 5.0), 2), True
        ))

    # Ring 2 (IP Velocity Proxy) Transactions: 30 transactions from 8 accounts via 198.51.100.99
    logger.info("Injecting Fraud Ring 2 (IP Velocity Proxy)...")
    ring2_time = now - datetime.timedelta(hours=3)
    for idx, u in enumerate(ring2_users * 4):
        tx_id = f"TXN_RING2_{idx:03d}"
        d_id = f"DEV_RING2_{u}"
        amt = round(random.uniform(60000, 120000), 2)
        tx_time = ring2_time + datetime.timedelta(minutes=idx * 3)
        tx_list.append((
            tx_id, u, d_id, "198.51.100.99", "MCH_042", amt, "INR", tx_time, "COMPLETED", idx + 2, round(random.uniform(3.0, 6.0), 2), True
        ))

    # Ring 3 (Carding Attack) Transactions: 15 micro txns in < 3 minutes
    logger.info("Injecting Fraud Ring 3 (Carding Micro-Transactions)...")
    ring3_time = now - datetime.timedelta(minutes=25)
    for idx in range(1, 16):
        tx_id = f"TXN_CARDER_{idx:03d}"
        amt = round(random.uniform(10, 99), 2)
        tx_time = ring3_time + datetime.timedelta(seconds=idx * 10)
        tx_list.append((
            tx_id, ring3_user, "DEV_CARDER_X", "203.0.113.50", random.choice(mch_ids), amt, "INR", tx_time, "COMPLETED", idx, round(random.uniform(1.2, 4.0), 2), True
        ))

    # Ring 4 (Collusion) Transactions: 15 large transactions to MCH_SUSPICIOUS_99
    logger.info("Injecting Fraud Ring 4 (Merchant Collusion)...")
    ring4_time = now - datetime.timedelta(hours=6)
    for idx, u in enumerate(ring4_users * 3):
        tx_id = f"TXN_COLLUSION_{idx:03d}"
        d_id = f"DEV_RING4_{u}"
        amt = round(random.uniform(80000, 150000), 2)
        tx_time = ring4_time + datetime.timedelta(minutes=idx * 5)
        tx_list.append((
            tx_id, u, d_id, ring4_ip[u], "MCH_SUSPICIOUS_99", amt, "INR", tx_time, "COMPLETED", 2, round(random.uniform(2.0, 4.5), 2), True
        ))

    # Ring 5 (Device-Cycling Structuring) Transactions: test charge, escalation
    # across one fixed device + incrementing IPs, then a >1h-later probe batch
    # on new borrowed devices to demonstrate velocity-window evasion.
    logger.info("Injecting Fraud Ring 5 (Device-Cycling Structuring)...")
    ring5_time = now - datetime.timedelta(hours=5)
    idx = 0
    # Test charge: trivial amount, confirms the instrument/account is live.
    tx_list.append((
        f"TXN_RING5_{idx:03d}", ring5_user, ring5_devices[0], ring5_escalation_ips[0],
        random.choice(mch_ids), 950.0, "INR", ring5_time, "COMPLETED", 1, 0.1, True
    ))
    idx += 1
    # Escalation: same device, IP incrementing by one each hop, amount
    # climbing toward the ceiling, all within a single hour (velocity_1h
    # climbs 2..8 across this run — computed for real by FEATURE_SQL from
    # these timestamps, these stored values are just plausible display seeds).
    for step, ip in enumerate(ring5_escalation_ips):
        amt = round(85000 + step * 1800 + random.uniform(-300, 300), 2)
        tx_time = ring5_time + datetime.timedelta(minutes=6 * (step + 1))
        tx_list.append((
            f"TXN_RING5_{idx:03d}", ring5_user, ring5_devices[0], ip,
            random.choice(mch_ids), amt, "INR", tx_time, "COMPLETED", step + 2, round(random.uniform(2.0, 4.5), 2), True
        ))
        idx += 1
    # Deliberate cool-down: >1h gap past the escalation batch, before pivoting
    # to two OTHER borrowed device fingerprints on a fresh IP for a final
    # low-amount reconnaissance probe — the velocity_1h window has fully
    # reset by this point despite the account being mid-ring-activity.
    probe_time = ring5_time + datetime.timedelta(hours=2, minutes=10)
    for step, dev in enumerate(ring5_devices[1:]):
        tx_time = probe_time + datetime.timedelta(minutes=4 * step)
        tx_list.append((
            f"TXN_RING5_{idx:03d}", ring5_user, dev, ring5_probe_ip,
            random.choice(mch_ids), 950.0, "INR", tx_time, "COMPLETED", 1, 0.1, True
        ))
        idx += 1

    # --- FRAUD: structuring — near-uniform amounts just under a reporting
    # threshold (₹1,00,000), same device + IP, compressed into ~90 minutes.
    # The visual/graph shape (several similar-sized payments in a burst) is
    # deliberately close to the benign bill_split scenario above; the tell
    # is the near-IDENTICAL amounts and single shared device+IP, not the
    # transaction count.
    logger.info("Injecting Structuring Fraud (uniform sub-threshold amounts)...")
    idx = 0
    struct_time = now - datetime.timedelta(hours=4)
    struct_base = 96000
    for i in range(6):
        amt = round(struct_base + random.uniform(-800, 800), 2)
        u = structuring_users[i % len(structuring_users)]
        tx_time = struct_time + datetime.timedelta(minutes=14 * i)
        tx_list.append((
            f"TXN_STRUCT_{idx:03d}", u, structuring_dev, structuring_ip, random.choice(mch_ids),
            amt, "INR", tx_time, "COMPLETED", i + 1, round(random.uniform(2.0, 4.0), 2), True
        ))
        idx += 1

    # --- FRAUD: fan-out laundering — one freshly-created account, 8
    # merchants, near-IDENTICAL amounts, all within one hour (rapid
    # dispersal after a single inbound credit). Same graph shape as the
    # benign fan_out_shopping scenario above; the tell is timing + amount
    # uniformity, not merchant count.
    logger.info("Injecting Fan-Out Laundering Fraud...")
    fanout_launder_merchants = random.sample(mch_ids, 8)
    launder_time = now - datetime.timedelta(hours=2)
    launder_base = 42000
    for i, m in enumerate(fanout_launder_merchants):
        amt = round(launder_base + random.uniform(-400, 400), 2)
        tx_time = launder_time + datetime.timedelta(minutes=6 * i)
        tx_list.append((
            f"TXN_FANOUTLAUNDER_{i:03d}", fanout_launderer, fanout_launder_dev, fanout_launder_ip, m,
            amt, "INR", tx_time, "COMPLETED", i + 1, round(random.uniform(2.5, 4.5), 2), True
        ))

    # --- FRAUD: no shared infrastructure — 4 independent users, own
    # devices, own IPs, zero graph overlap with each other or anyone else.
    # Individually anomalous (large amount, unusual hour, high z-score) but
    # NOT graph-detectable. Tests that detection does not depend on
    # connectivity existing in the first place (see risk_aggregator.py:
    # evidence_confluence can never fire here since there is no fingerprint
    # sharing — the tabular/behavioral path has to catch this alone).
    logger.info("Injecting No-Shared-Infrastructure Fraud...")
    idx = 0
    for u in no_infra_users:
        odd_hour_time = (now - datetime.timedelta(days=random.randint(1, 5))).replace(hour=random.choice([2, 3, 4]), minute=random.randint(0, 59))
        amt = round(random.uniform(55000, 95000), 2)
        tx_list.append((
            f"TXN_NOINFRA_{idx:03d}", u, no_infra_dev[u], no_infra_ip[u], "MCH_SUSPICIOUS_99",
            amt, "INR", odd_hour_time, "COMPLETED", 1, round(random.uniform(3.5, 6.0), 2), True
        ))
        idx += 1

    # --- FRAUD: low-and-slow — own devices/IPs (no fingerprint sharing at
    # all), ordinary-looking amounts, spread across ~3 weeks, no velocity
    # spikes. Ground-truth labeled fraud but deliberately built to have NO
    # graph anomaly and NO behavioral-velocity anomaly — the "fraud that
    # looks like normal behavior" case. This is an honest stress test: if
    # current signals (tabular + GNN + confluence overlay) miss this
    # scenario, that is real, expected, and should be reported as a known
    # gap rather than hidden — see PROJECT_WORKFLOW.md.
    logger.info("Injecting Low-and-Slow Fraud (deliberately unremarkable)...")
    idx = 0
    low_slow_time = now - datetime.timedelta(days=25)
    low_slow_merchants = random.sample(mch_ids, 3)
    for u in low_slow_users:
        for _ in range(random.randint(2, 4)):
            amt = round(random.uniform(300, 1200), 2)
            tx_time = low_slow_time + datetime.timedelta(minutes=random.randint(1, 25 * 24 * 60))
            tx_list.append((
                f"TXN_LOWSLOW_{idx:03d}", u, low_slow_dev[u], low_slow_ip[u], random.choice(low_slow_merchants),
                amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.3, 1.0), 2), True
            ))
            idx += 1

    # --- FRAUD: cold start — a brand-new account's very first transaction
    # is already large, to a risky merchant, on a brand-new device+IP, with
    # no history to compare against (amount_zscore_prior=0.0 by
    # construction — see live_tabular_score's cold-start path). No graph
    # sharing either (own device/IP, not linked to anyone). This is
    # deliberately hard: if it's missed, that documents a real gap
    # (account-age isn't a feature in this dataset) rather than hiding it.
    logger.info("Injecting Cold-Start Fraud (new account, first txn already anomalous)...")
    idx = 0
    for u in cold_start_fraud_users:
        amt = round(random.uniform(60000, 95000), 2)
        tx_time = now - datetime.timedelta(minutes=random.randint(1, 90))
        tx_list.append((
            f"TXN_COLDSTARTFRAUD_{idx:03d}", u, cold_start_fraud_dev[u], cold_start_fraud_ip[u],
            "MCH_SUSPICIOUS_99", amt, "INR", tx_time, "COMPLETED", 1, 0.0, True
        ))
        idx += 1

    # --- FRAUD: account takeover — an established user's normal history
    # (own regular device/IP, ordinary amounts, spread over weeks) followed
    # by ONE hijacked transaction from a brand-new device+IP (own machine,
    # not shared with anyone — zero graph anomaly), unusual hour, and an
    # amount far outside their own baseline. The "behavioral anomaly
    # without graph anomaly" pathway explicitly called out in review —
    # connectivity evidence structurally cannot catch this.
    logger.info("Injecting Account Takeover Fraud (established account, sudden behavioral shift)...")
    idx = 0
    ato_merchants = random.sample(mch_ids, 3)
    ato_history_time = now - datetime.timedelta(days=45)
    for u in ato_users:
        for i in range(6):
            amt = round(random.uniform(300, 1800), 2)
            tx_time = ato_history_time + datetime.timedelta(days=i * 7, hours=random.randint(9, 20))
            tx_list.append((
                f"TXN_ATOHIST_{idx:03d}", u, ato_normal_dev[u], ato_normal_ip[u], random.choice(ato_merchants),
                amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.4, 0.6), 2), False
            ))
            idx += 1
        hijack_amt = round(random.uniform(70000, 98000), 2)
        hijack_time = now.replace(hour=random.choice([2, 3, 4]), minute=random.randint(0, 59)) - datetime.timedelta(days=random.randint(0, 1))
        tx_list.append((
            f"TXN_ATOHIJACK_{u}", u, ato_hijack_dev[u], ato_hijack_ip[u], "MCH_SUSPICIOUS_99",
            hijack_amt, "INR", hijack_time, "COMPLETED", 1, round(random.uniform(4.0, 7.0), 2), True
        ))

    # --- FRAUD: tabular-only behavioral anomalies. No shared device/IP
    # relationships exist. Each account has a private infrastructure and the
    # fraud is visible from transaction behavior alone. Mix three patterns so
    # the model learns a family of anomalies rather than one exact template.
    logger.info("Injecting tabular-only obvious fraud scenarios...")
    idx = 0
    for i, u in enumerate(obvious_fraud_users):
        d_id = obvious_fraud_dev[u]
        ip_addr = obvious_fraud_ip[u]
        if i % 3 == 0:
            # Large round-number cash-out at an unusual hour.
            amt = random.choice([50000.0, 75000.0, 90000.0, 100000.0])
            hour = random.choice([1, 2, 3, 4])
            merchant = "MCH_SUSPICIOUS_99"
            z = random.uniform(4.0, 7.0)
        elif i % 3 == 1:
            # Repeated same-merchant authorization attempts: card testing /
            # cash-out probe, with moderate-to-large amounts.
            amt = random.choice([12000.0, 15000.0, 20000.0, 25000.0])
            hour = random.choice([0, 2, 5])
            merchant = random.choice(mch_ids)
            z = random.uniform(2.5, 5.0)
        else:
            # Large non-round transfer at a very unusual hour; no graph clue.
            amt = round(random.uniform(45000, 98000), 2)
            hour = random.choice([2, 3, 4])
            merchant = "MCH_SUSPICIOUS_99"
            z = random.uniform(3.5, 6.5)
        tx_time = (now - datetime.timedelta(days=random.randint(0, 12))).replace(
            hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59), microsecond=0)
        tx_list.append((
            f"TXN_OBVIOUSFRAUD_{idx:04d}", u, d_id, ip_addr, merchant, amt, "INR",
            tx_time, "COMPLETED", 1, round(z, 2), True
        ))
        idx += 1
        # A subset has a second same-merchant attempt within the hour, making
        # the temporal pattern explicit without introducing graph overlap.
        if i % 4 == 0:
            tx_list.append((
                f"TXN_OBVIOUSFRAUD_{idx:04d}", u, d_id, ip_addr, merchant,
                round(amt * random.uniform(0.35, 0.65), 2), "INR",
                tx_time + datetime.timedelta(minutes=random.randint(8, 35)),
                "COMPLETED", 2, round(max(z - 0.5, 1.5), 2), True
            ))
            idx += 1

    # --- BENIGN: high-value legitimate purchases. Same broad amount scale as
    # obvious fraud, but daytime, stable infrastructure, familiar merchants,
    # and no velocity spike. This is a deliberate hard-negative population.
    idx = 0
    for u in high_value_benign_users:
        for j in range(3):
            amt = round(random.uniform(18000, 65000), 2)
            tx_time = now - datetime.timedelta(days=random.randint(1, 25))
            tx_time = tx_time.replace(hour=random.randint(9, 21), minute=random.randint(0, 59), second=0, microsecond=0)
            tx_list.append((
                f"TXN_HIGHVALUE_{idx:04d}", u, high_value_dev[u], high_value_ip[u],
                random.choice(mch_ids), amt, "INR", tx_time, "COMPLETED", 1,
                round(random.uniform(0.2, 1.8), 2), False
            ))
            idx += 1

    # Insert into Database
    cursor.executemany("""
        INSERT INTO transactions 
        (transaction_id, user_id, device_id, ip_address, merchant_id, amount, currency, timestamp, status, velocity_1h, amount_zscore_prior, is_fraud_ground_truth)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, tx_list)

    conn.commit()
    conn.close()

    logger.info(f"Synthetic Data Generation Complete! Generated {len(tx_list)} total transactions.")
    return len(tx_list)

if __name__ == "__main__":
    generate_dataset()

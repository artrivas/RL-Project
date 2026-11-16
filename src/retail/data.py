"""Streaming M5 selection and selected-SKU audit. No fitted test statistics."""
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import math
from pathlib import Path
from .config import STORES

SPLITS = {"train": (1, 1211), "validation": (1212, 1576), "test": (1577, 1941)}
FILES = ("sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv")

@dataclass(frozen=True)
class RetailData:
    item_id: str
    sales: tuple  # store-major, d_1 at index 0
    prices: tuple
    calendar: tuple
    manifest: dict

    @property
    def days(self):
        return len(self.calendar)

    def __post_init__(self):
        if len(self.sales) != 3 or len(self.prices) != 3 or not self.calendar:
            raise ValueError("data must contain three stores and a nonempty calendar")
        for series in (*self.sales, *self.prices):
            if len(series) != self.days:
                raise ValueError("misaligned daily series")
        if any(type(x) is not int or x < 0 for series in self.sales for x in series):
            raise ValueError("sales must be nonnegative integers")
        if any(not math.isfinite(x) or x <= 0 for series in self.prices for x in series):
            raise ValueError("prices must be finite and positive")

def _rows(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required) <= set(fields):
            raise ValueError(f"{path}: missing or duplicate columns")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}: malformed CSV row")
            yield row

def _units(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid sales count: {value!r}") from exc
    if result < 0:
        raise ValueError("negative sales")
    return result

def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_m5(directory, item_id=None):
    """Auto-selection maximizes train sales among items present in all stores.

    Validate all calendar days and all values/price joins for the chosen item.
    This is deliberately not advertised as an audit of every SKU in M5.
    """
    automatic_selection = item_id is None
    root = Path(directory)
    days = tuple(f"d_{d}" for d in range(1, 1942))
    required = ("item_id", "store_id", *days)
    totals, seen = {}, set()
    for row in _rows(root / FILES[0], required):
        store = row["store_id"]
        if store not in STORES:
            continue
        key = row["item_id"], store
        if key in seen:
            raise ValueError(f"duplicate sales key: {key}")
        seen.add(key)
        # Selection deliberately never reads validation/test values.
        total = sum(_units(row[d]) for d in days[:1211])
        totals.setdefault(row["item_id"], {})[store] = total
    eligible = {item: sum(values.values()) for item, values in totals.items()
                if set(values) == set(STORES) and min(values.values()) > 0}
    if item_id is None:
        if not eligible:
            raise ValueError("no SKU has positive train sales in all three stores")
        item_id = min(eligible, key=lambda item: (-eligible[item], item))
    if item_id not in eligible:
        raise ValueError("requested SKU is missing a store or has zero train sales")
    sales = {}
    for row in _rows(root / FILES[0], required):
        if row["item_id"] == item_id and row["store_id"] in STORES:
            sales[row["store_id"]] = tuple(_units(row[d]) for d in days)

    calendar = {}
    columns = ("d", "date", "wm_yr_wk", "snap_CA", "event_name_1", "event_name_2")
    for row in _rows(root / FILES[1], columns):
        key = row["d"]
        if key in calendar:
            raise ValueError(f"duplicate calendar key: {key}")
        dt = date.fromisoformat(row["date"])
        snap = int(row["snap_CA"])
        if snap not in (0, 1):
            raise ValueError("snap_CA must be binary")
        calendar[key] = {"date": dt.isoformat(), "week": row["wm_yr_wk"],
                         "weekday": dt.weekday(), "month": dt.month,
                         "snap": snap, "event": int(bool(row["event_name_1"] or row["event_name_2"]))}
    if not set(days) <= calendar.keys():
        raise ValueError("missing calendar days")
    start = date(2011, 1, 29)
    for i, day in enumerate(days):
        if calendar[day]["date"] != (start + timedelta(days=i)).isoformat():
            raise ValueError(f"unexpected date for {day}")
    weekly = {}
    for row in _rows(root / FILES[2], ("store_id", "item_id", "wm_yr_wk", "sell_price")):
        if row["item_id"] != item_id or row["store_id"] not in STORES:
            continue
        key = row["store_id"], row["wm_yr_wk"]
        if key in weekly:
            raise ValueError(f"duplicate price key: {key}")
        value = float(row["sell_price"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid price: {key}")
        weekly[key] = value
    try:
        prices = tuple(tuple(weekly[(s, calendar[d]["week"])] for d in days) for s in STORES)
    except KeyError as exc:
        raise ValueError(f"missing price; no future-price backfill allowed: {exc}") from exc
    manifest = {"source": "M5", "item_id": item_id, "stores": STORES,
                "selection": "train_sales_desc_item_id_asc" if automatic_selection else "explicit",
                "splits": SPLITS, "audit_scope": "selected_sku_all_days",
                "sha256": {name: fingerprint(root / name) for name in FILES},
                "historical_inventory_observed": False,
                "reward": "simulated_operating_profit_proxy"}
    return RetailData(item_id, tuple(sales[s] for s in STORES), prices,
                      tuple(calendar[d] for d in days), manifest)

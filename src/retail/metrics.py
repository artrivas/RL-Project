"""Metrics over unshaped simulator outcomes."""
def summarize(transitions):
    rows = list(transitions)
    if not rows:
        raise ValueError("cannot summarize an empty rollout")
    demand = sum(sum(r["demand"]) for r in rows)
    sold = sum(sum(r["sales"]) for r in rows)
    eligible = sum(d > 0 for r in rows for d in r["demand"])
    stockouts = sum(x > 0 for r in rows for x in r["lost"])
    return {"days": len(rows), "base_return": sum(r["base_reward"] for r in rows),
            "demand_units": demand, "sold_units": sold, "lost_units": demand-sold,
            "eligible_store_days": eligible, "stockout_store_days": stockouts,
            "stockout_rate": stockouts/eligible if eligible else None,
            "fill_rate": sold/demand if demand else None}

def relative_stockout_reduction(candidate, baseline):
    if baseline is None or baseline == 0 or candidate is None:
        return None
    return 1-candidate/baseline


def discounted_return(rewards, gamma):
    rewards=list(rewards)
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0,1]")
    total=0.0
    discount=1.0
    for r in rewards:
        total+=discount*float(r)
        discount*=gamma
    return float(total)

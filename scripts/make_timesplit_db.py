#!/usr/bin/env python3
"""Copy training.db and drop all rows on/after a cutoff date.

Produces the train-side database for an honest out-of-sample benchmark:
train on the copy, evaluate with benchmark_model.py --holdout-start <cutoff>
against the full database.
"""

import argparse
import shutil
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="training.db")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--cutoff", required=True, help="YYYY-MM-DD; rows >= cutoff are removed")
    args = ap.parse_args()

    shutil.copy(args.src, args.dst)
    conn = sqlite3.connect(args.dst)
    n = conn.execute("SELECT COUNT(*) FROM market_data WHERE date >= ?", (args.cutoff,)).fetchone()[
        0
    ]
    conn.execute("DELETE FROM market_data WHERE date >= ?", (args.cutoff,))
    conn.commit()
    conn.execute("VACUUM")
    print(f"removed {n} rows on/after {args.cutoff}")
    for at in ("forex", "stock"):
        r = conn.execute(
            "SELECT COUNT(*), MAX(date) FROM market_data WHERE asset_type=?", (at,)
        ).fetchone()
        print(f"  {at}: rows={r[0]} max_date={r[1]}")
    conn.close()


if __name__ == "__main__":
    main()

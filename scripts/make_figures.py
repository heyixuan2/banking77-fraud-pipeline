"""One file per figure, for the Quarto report to include."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

d = json.load(open("eda_out.json"))
daily = pl.read_parquet("fig_daily.parquet"); lens = pl.read_parquet("fig_len.parquet")
intents = pl.read_parquet("fig_intents.parquet"); drift = pl.read_parquet("fig_drift.parquet")
hour = pl.read_parquet("fig_hour.parquet"); part = pl.read_parquet("fig_part.parquet")
bm = json.load(open("bench.json"))
get = lambda e, q: next(r["seconds"] for r in bm if r["engine"] == e and r["query"] == q)

INK, MUTED, ACC = "#1a1a1a", "#5a5a5a", "#c05621"
C = {"washington_dc": "#2b6cb0", "addis_ababa": "#c05621", "yuzhno_sakhalinsk": "#2f855a"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#cccccc", "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white"})
os.makedirs("figs", exist_ok=True)

def finish(fig, ax, name):
    for sp in ("top", "right"):
        (ax if not isinstance(ax, list) else ax[0]).spines[sp].set_visible(False)
    fig.savefig(f"figs/{name}.png", dpi=150)
    plt.close(fig)

# 1 daily volume
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for b in C:
    s = daily.filter(pl.col("branch") == b).sort("date")
    ax.plot(s["date"], s["n"] / 1000, lw=1.0, color=C[b], label=b.replace("_", " "))
ax.set_ylabel("thousands of messages / day"); ax.set_ylim(bottom=0)
ax.legend(frameon=False, ncol=3, fontsize=8); ax.tick_params(axis="x", rotation=25)
finish(fig, ax, "daily")

# 2 local clock
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for b in C:
    s = hour.filter(pl.col("branch") == b).sort("local_h")
    ax.plot(s["local_h"], s["n"] / 1000, lw=1.6, color=C[b], label=b.replace("_", " "))
ax.set_xlabel("hour of day, in the branch's own time zone")
ax.set_ylabel("thousands of messages"); ax.set_xticks([0, 6, 12, 18, 23])
ax.legend(frameon=False, fontsize=8)
finish(fig, ax, "clock")

# 3 review coverage
fig, ax = plt.subplots(figsize=(7.2, 2.4))
bs = [x["branch"] for x in d["by_branch"]]
ax.barh(range(len(bs)), [x["n"] / 1e6 for x in d["by_branch"]], color="#dde3ea", label="unreviewed (label = NA)")
ax.barh(range(len(bs)), [x["labelled"] / 1e6 for x in d["by_branch"]], color="#2b6cb0", label="reviewed")
ax.set_yticks(range(len(bs))); ax.set_yticklabels([b.replace("_", " ") for b in bs])
ax.set_xlabel("millions of messages"); ax.legend(frameon=False, fontsize=8, loc="lower right")
finish(fig, ax, "coverage")

# 4 message length
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.hist(lens["len_chars"], bins=140, range=(0, 1250), color="#2b6cb0", log=True)
ax.axvline(999, color="#c53030", lw=1.1, ls="--")
ax.text(1010, ax.get_ylim()[1] * 0.12, "999 characters", fontsize=8, color="#c53030")
ax.set_xlabel("characters per message"); ax.set_ylabel("rows (log scale)")
finish(fig, ax, "length")

# 5 intent imbalance
fig, ax = plt.subplots(figsize=(7.2, 2.8))
srt = intents.sort("n", descending=True)
ax.bar(range(srt.height), srt["n"] / 1000, color="#2b6cb0", width=1.0)
ax.set_xlabel("the 77 intents, most to least frequent")
ax.set_ylabel("thousands of reviewed messages")
finish(fig, ax, "intents")

# 6 drift
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.plot(drift["week"], drift["pct"], marker="o", ms=4, lw=1.6, color=ACC)
ax.set_ylabel("% of reviewed rows on intent 54\nusing the new wording")
ax.tick_params(axis="x", rotation=25)
finish(fig, ax, "drift")

# 7 engines
fig, ax = plt.subplots(figsize=(7.2, 2.9))
eng = ["Polars", "DuckDB", "Athena"]; W = 0.34
full = [get(e, "full scan") for e in eng]; pru = [get(e, "3-week slice") for e in eng]
x = range(len(eng))
ax.bar([i - W/2 for i in x], full, W, color="#a0aec0", label="all 90 days")
ax.bar([i + W/2 for i in x], pru, W, color="#2b6cb0", label="three-week slice")
for i, (f, pz) in enumerate(zip(full, pru)):
    ax.text(i - W/2, f + .08, f"{f:.2f}s", ha="center", fontsize=8)
    ax.text(i + W/2, pz + .08, f"{pz:.2f}s", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(eng); ax.set_ylabel("seconds")
ax.legend(frameon=False, fontsize=8)
finish(fig, ax, "engines")

# 8 partitions
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.hist(part["n"] / 1000, bins=34, color="#2f855a")
ax.axvline(d["part"]["median"] / 1000, color=INK, lw=1.1, ls="--")
ax.text(d["part"]["median"] / 1000 + 4, ax.get_ylim()[1] * .9, f"median {d['part']['median']:,}", fontsize=8)
ax.set_xlabel("thousands of rows in a date/branch partition"); ax.set_ylabel("partitions")
finish(fig, ax, "partitions")

print("figs/:", sorted(os.listdir("figs")))

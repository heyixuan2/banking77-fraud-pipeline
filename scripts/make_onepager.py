"""Compose the one-page EDA progress report as a PDF."""
import json
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import polars as pl

d = json.load(open("eda_out.json"))
daily = pl.read_parquet("fig_daily.parquet")
lens = pl.read_parquet("fig_len.parquet")
intents = pl.read_parquet("fig_intents.parquet")
drift = pl.read_parquet("fig_drift.parquet")

INK, MUTED = "#1a1a1a", "#666666"
C = {"washington_dc": "#2b6cb0", "addis_ababa": "#c05621", "yuzhno_sakhalinsk": "#2f855a"}
plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#cccccc",
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 8, "axes.titleweight": "bold"})

fig = plt.figure(figsize=(8.5, 11))
gs = GridSpec(4, 2, figure=fig, height_ratios=[0.90, 1, 1, 1],
              hspace=0.55, wspace=0.25,
              left=0.09, right=0.96, top=0.96, bottom=0.05)

# ---- header -------------------------------------------------------------
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
ax.text(0, 1.0, "Catching Fraud in the Chat Box — EDA Progress Report",
        fontsize=14, fontweight="bold", va="top")
ax.text(0, 0.80, "Yixuan He, Lanbin Fan, Zizheng Wang, Ken Qin   ·   "
                 "DSAN 6000 Final Project   ·   22 November 2026",
        fontsize=8, color=MUTED, va="top")
body = (
    "Banking77's released files hold 13,083 labelled messages (~1 MB), far too small to exercise a Spark cluster, and they "
    "carry a label for every row — the opposite of what a bank's live chatbot stream looks like. We therefore replay those "
    "messages into a simulated production log: 90 days of traffic across three branches, with the timestamps, the sparse "
    "review coverage, and the adversarial behaviour a fraud reviewer would actually face. The log now sits in S3 as "
    "Hive-partitioned Parquet and is the input for every later stage of the project. Everything below was computed by "
    "scanning that S3 copy directly with Polars — the data was never pulled onto the instance."
)
ax.text(0, 0.66, "\n".join(textwrap.wrap(body, 118)), fontsize=7.6, va="top", linespacing=1.55)

stats = (f"{d['rows']:,} rows  ·  {d['cols']} columns  ·  270 partitions  ·  1.99 GB Parquet  ·  "
         f"{d['days']} days  ·  s3://dsan6000-banking77-team/raw/")
ax.text(0, -0.17, stats, fontsize=7.6, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f2f5f9", ec="#ccd6e0"))

# ---- 1. daily volume ----------------------------------------------------
ax1 = fig.add_subplot(gs[1, 0])
for b in ["washington_dc", "addis_ababa", "yuzhno_sakhalinsk"]:
    s = daily.filter(pl.col("branch") == b).sort("date")
    ax1.plot(s["date"], s["n"], lw=0.9, color=C[b], label=b.replace("_", " "))
ax1.set_title("1  Daily message volume by branch")
ax1.set_ylabel("messages / day")
ax1.set_ylim(bottom=0); ax1.legend(frameon=False, fontsize=6, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.02))
ax1.tick_params(axis="x", rotation=30, labelsize=6)
for sp in ("top", "right"): ax1.spines[sp].set_visible(False)

# ---- 2. review coverage -------------------------------------------------
ax2 = fig.add_subplot(gs[1, 1])
bs = [b["branch"] for b in d["by_branch"]]
tot = [b["n"] / 1e6 for b in d["by_branch"]]
lab = [b["labelled"] / 1e6 for b in d["by_branch"]]
y = range(len(bs))
ax2.barh(y, tot, color="#dde3ea", label="unreviewed (label = NA)")
ax2.barh(y, lab, color="#2b6cb0", label="reviewed")
ax2.set_yticks(list(y)); ax2.set_yticklabels([b.replace("_", " ") for b in bs], fontsize=6)
ax2.set_xlabel("millions of messages")
ax2.set_title(f"2  Review coverage — only {d['labelled_pct']}% carries a label")
ax2.legend(frameon=False, fontsize=6, loc="lower right")
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)

# ---- 3. message length --------------------------------------------------
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(lens["len_chars"], bins=120, range=(0, 1250), color="#2b6cb0", log=True)
ax3.axvline(999, color="#c53030", lw=1, ls="--")
ax3.text(1005, ax3.get_ylim()[1] * 0.25, f"999 chars\n{d['over_999']:,} rows above",
         fontsize=6, color="#c53030")
ax3.set_title("3  Message length — token-flooding sits in the right tail")
ax3.set_xlabel("characters"); ax3.set_ylabel("rows (log)")
for sp in ("top", "right"): ax3.spines[sp].set_visible(False)

# ---- 4. intent imbalance ------------------------------------------------
ax4 = fig.add_subplot(gs[2, 1])
srt = intents.sort("n", descending=True)
ax4.bar(range(srt.height), srt["n"], color="#2b6cb0", width=1.0)
ax4.set_title(f"4  Intent imbalance on the reviewed subset ({d['imbalance_ratio']}x)")
ax4.set_xlabel("77 intents, most to least frequent"); ax4.set_ylabel("reviewed messages")
ax4.text(0.97, 0.93, srt["intent"][0], ha="right", transform=ax4.transAxes, fontsize=5.5, color=MUTED)
ax4.text(0.97, 0.84, "…", ha="right", transform=ax4.transAxes, fontsize=5.5, color=MUTED)
ax4.text(0.97, 0.75, srt["intent"][-1], ha="right", transform=ax4.transAxes, fontsize=5.5, color=MUTED)
for sp in ("top", "right"): ax4.spines[sp].set_visible(False)

# ---- 5. drift -----------------------------------------------------------
ax5 = fig.add_subplot(gs[3, 0])
ax5.plot(drift["week"], drift["pct"], marker="o", ms=3, lw=1.2, color="#c05621")
ax5.set_title("5  Vocabulary drift on intent 54")
ax5.set_ylabel("% of reviewed rows\nusing the new wording")
ax5.tick_params(axis="x", rotation=30, labelsize=6)
for sp in ("top", "right"): ax5.spines[sp].set_visible(False)

# ---- 6. ground truth + next steps --------------------------------------
ax6 = fig.add_subplot(gs[3, 1]); ax6.axis("off")
ax6.text(0, 1.0, "6  Planted ground truth", fontsize=8, fontweight="bold", va="top")
rows = [(i["injected_type"], i["n"]) for i in d["injected"] if i["injected_type"]]
yy = 0.86
for name, n in rows:
    ax6.text(0, yy, name.replace("_", " "), fontsize=7, va="top")
    ax6.text(1.0, yy, f"{n:,}", fontsize=7, va="top", ha="right")
    yy -= 0.09
ax6.text(0, yy - 0.02,
         "Banking77 ships no fraud labels, so these rows are planted by the\n"
         "generator and recorded in injected_type. They are the evaluation\n"
         "target, and are the only rows whose fraud status we know.",
         fontsize=6.3, va="top", color=MUTED, linespacing=1.5)
ax6.text(0, yy - 0.30, "Next: Spark cluster run (due 29 Nov)", fontsize=8,
         fontweight="bold", va="top")
ax6.text(0, yy - 0.39,
         "Same log, 1 / 2 / 4 workers against 1% / 10% / 100% slices,\n"
         "recording runtime, shuffle size and partition counts.",
         fontsize=6.3, va="top", color=MUTED, linespacing=1.5)

fig.savefig("eda_report.pdf", dpi=200)
fig.savefig("eda_report.png", dpi=170)
print("wrote eda_report.pdf / eda_report.png")

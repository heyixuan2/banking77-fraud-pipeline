"""Compose the two-page EDA progress report."""
import json, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import polars as pl

d = json.load(open("eda_out.json"))
daily = pl.read_parquet("fig_daily.parquet"); lens = pl.read_parquet("fig_len.parquet")
intents = pl.read_parquet("fig_intents.parquet"); drift = pl.read_parquet("fig_drift.parquet")
hour = pl.read_parquet("fig_hour.parquet"); part = pl.read_parquet("fig_part.parquet")
bench = pl.read_parquet("fig_tokens.parquet"); bm = json.load(open("bench.json"))

INK, MUTED, ACC = "#1a1a1a", "#666666", "#c05621"
C = {"washington_dc": "#2b6cb0", "addis_ababa": "#c05621", "yuzhno_sakhalinsk": "#2f855a"}
plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#cccccc", "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 8, "axes.titleweight": "bold"})
def clean(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

pdf = PdfPages("eda_report.pdf")

# ===================== PAGE 1 =====================
fig = plt.figure(figsize=(8.5, 11))
gs = GridSpec(4, 2, figure=fig, height_ratios=[0.85, 1, 1, 1], hspace=0.55,
              wspace=0.30, left=0.125, right=0.955, top=0.955, bottom=0.075)
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
ax.text(0, 1.0, "Catching Fraud in the Chat Box — EDA Progress Report",
        fontsize=14, fontweight="bold", va="top")
ax.text(0, 0.82, "Yixuan He, Lanbin Fan, Zizheng Wang, Ken Qin   ·   DSAN 6000 Final Project"
                 "   ·   22 November 2026   ·   Page 1 of 2: the data",
        fontsize=8, color=MUTED, va="top")
body = ("Banking77 ships 13,083 messages in about 1 MB, and every one of them carries a label. A bank's live chatbot "
        "stream looks nothing like that — it is large, and almost none of it has been reviewed. So we replay those 13,083 "
        "messages into a 90-day log across three branches, each running on its own business hours, with review coverage set "
        "to 5% and three adversarial patterns planted in. The log sits in S3 as Hive-partitioned Parquet. Polars read it "
        "straight from the bucket to produce every number here; nothing was copied onto the instance.")
ax.text(0, 0.64, "\n".join(textwrap.wrap(body, 118)), fontsize=7.6, va="top", linespacing=1.55)
ax.text(0, -0.20, f"{d['rows']:,} rows  ·  {d['cols']} columns  ·  {d['part']['n']} partitions  ·  1.99 GB Parquet  ·  "
                  f"{d['days']} days  ·  s3://dsan6000-banking77-team/raw/",
        fontsize=7.6, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f2f5f9", ec="#ccd6e0"))

a = fig.add_subplot(gs[1, 0])
for b in C:
    s = daily.filter(pl.col("branch") == b).sort("date")
    a.plot(s["date"], s["n"], lw=0.9, color=C[b], label=b.replace("_", " "))
a.set_title("1  Daily volume by branch"); a.set_ylabel("messages / day"); a.set_ylim(bottom=0)
a.legend(frameon=False, fontsize=6, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.02))
a.tick_params(axis="x", rotation=30, labelsize=6); clean(a)

a = fig.add_subplot(gs[1, 1])
for b in C:
    s = hour.filter(pl.col("branch") == b).sort("local_h")
    a.plot(s["local_h"], s["n"] / 1000, lw=1.2, color=C[b], label=b.replace("_", " "))
a.set_title("2  Traffic follows each branch's local clock")
a.set_xlabel("hour of day, branch-local"); a.set_ylabel("thousands of messages")
a.set_xticks([0, 6, 12, 18, 23]); a.legend(frameon=False, fontsize=6); clean(a)

a = fig.add_subplot(gs[2, 0])
bs = [x["branch"] for x in d["by_branch"]]
a.barh(range(len(bs)), [x["n"] / 1e6 for x in d["by_branch"]], color="#dde3ea", label="unreviewed (label = NA)")
a.barh(range(len(bs)), [x["labelled"] / 1e6 for x in d["by_branch"]], color="#2b6cb0", label="reviewed")
a.set_yticks(range(len(bs))); a.set_yticklabels([b.replace("_", " ") for b in bs], fontsize=6)
a.set_xlabel("millions of messages"); a.set_title(f"3  Review coverage — {d['labelled_pct']}% carries a label")
a.legend(frameon=False, fontsize=6, loc="lower right"); clean(a)

a = fig.add_subplot(gs[2, 1])
a.hist(lens["len_chars"], bins=120, range=(0, 1250), color="#2b6cb0", log=True)
a.axvline(999, color="#c53030", lw=1, ls="--")
a.text(1005, a.get_ylim()[1] * 0.2, f"999 chars\n{d['over_999']:,} rows above", fontsize=6, color="#c53030")
a.set_title("4  Message length — flooding sits in the tail")
a.set_xlabel("characters"); a.set_ylabel("rows (log)"); clean(a)

a = fig.add_subplot(gs[3, 0])
srt = intents.sort("n", descending=True)
a.bar(range(srt.height), srt["n"], color="#2b6cb0", width=1.0)
a.set_title(f"5  Intent imbalance, reviewed subset ({d['imbalance_ratio']}x)")
a.set_xlabel("77 intents, most to least frequent"); a.set_ylabel("reviewed messages")
a.text(0.97, 0.92, srt["intent"][0], ha="right", transform=a.transAxes, fontsize=5.5, color=MUTED)
a.text(0.97, 0.80, srt["intent"][-1], ha="right", transform=a.transAxes, fontsize=5.5, color=MUTED); clean(a)

a = fig.add_subplot(gs[3, 1])
a.plot(drift["week"], drift["pct"], marker="o", ms=3, lw=1.2, color=ACC)
a.set_title("6  Vocabulary drift on intent 54")
a.set_ylabel("% of reviewed rows\nusing the new wording")
a.tick_params(axis="x", rotation=30, labelsize=6); clean(a)
pdf.savefig(fig); plt.close(fig)

# ===================== PAGE 2 =====================
fig = plt.figure(figsize=(8.5, 11))
gs = GridSpec(4, 2, figure=fig, height_ratios=[0.5, 1, 1, 1], hspace=0.55,
              wspace=0.30, left=0.125, right=0.955, top=0.955, bottom=0.075)
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
ax.text(0, 1.0, "Page 2 — querying the lake, and what is wrong with the data",
        fontsize=12, fontweight="bold", va="top")
body2 = ("We ran one query — messages and reviewed messages per branch — three ways over the same S3 Parquet: once "
         "across all 90 days, once across a three-week slice. The pair separates how fast an engine is from how much work "
         "partition pruning saves it. The caveats at the bottom are limits of our simulation, not of the tools, and they "
         "bound what the modelling stage can claim.")
ax.text(0, 0.62, "\n".join(textwrap.wrap(body2, 118)), fontsize=7.6, va="top", linespacing=1.55)

a = fig.add_subplot(gs[1, :])
eng = ["Polars", "DuckDB", "Athena"]; W = 0.35
full = [next(r["seconds"] for r in bm if r["engine"] == e and r["query"] == "full scan") for e in eng]
pru = [next(r["seconds"] for r in bm if r["engine"] == e and r["query"] == "3-week slice") for e in eng]
x = range(len(eng))
a.bar([i - W/2 for i in x], full, W, color="#a0aec0", label="all 90 days")
a.bar([i + W/2 for i in x], pru, W, color="#2b6cb0", label="3-week slice (partition pruning)")
for i, (f, p) in enumerate(zip(full, pru)):
    a.text(i - W/2, f + .1, f"{f:.2f}s", ha="center", fontsize=6.5)
    a.text(i + W/2, p + .1, f"{p:.2f}s", ha="center", fontsize=6.5)
a.set_xticks(list(x)); a.set_xticklabels(eng); a.set_ylabel("seconds")
a.set_title("7  Same query, three engines — wall-clock"); a.legend(frameon=False, fontsize=6.5); clean(a)

a = fig.add_subplot(gs[2, 0]); a.axis("off")
a.text(0, 1.0, "8  What Athena actually read", fontsize=8, fontweight="bold", va="top")
sc_f = next(r["scanned_mb"] for r in bm if r["engine"] == "Athena" and r["query"] == "full scan")
sc_p = next(r["scanned_mb"] for r in bm if r["engine"] == "Athena" and r["query"] == "3-week slice")
a.text(0, 0.80, f"all 90 days        {sc_f:,.1f} MB scanned", fontsize=7.5, va="top", family="monospace")
a.text(0, 0.68, f"3-week slice        {sc_p:,.1f} MB scanned", fontsize=7.5, va="top", family="monospace")
a.text(0, 0.56, f"dataset on disk   {1990:,} MB", fontsize=7.5, va="top", family="monospace", color=MUTED)
a.text(0, 0.36, "\n".join(textwrap.wrap(
    "Counting all 44.5M rows cost 5.9 MB of reads against a 1.99 GB file set, because Parquet hands the engine only the "
    "columns the query names. Narrowing to three weeks cut that by another 4.2x. Athena's clock hardly moved: about 3.9s "
    "of it is start-up it pays no matter how small the question. That overhead is why Polars and DuckDB win here, and why "
    "they would lose on a dataset 100x larger.", 56)),
    fontsize=6.5, va="top", color=MUTED, linespacing=1.5)

a = fig.add_subplot(gs[2, 1])
a.hist(part["n"] / 1000, bins=30, color="#2f855a")
a.set_title(f"9  Partition sizes — {d['part_skew']}x skew")
a.set_xlabel("thousands of rows per date/branch partition"); a.set_ylabel("partitions")
a.axvline(d["part"]["median"] / 1000, color=INK, lw=1, ls="--")
a.text(d["part"]["median"] / 1000, a.get_ylim()[1] * .9, f"  median {d['part']['median']:,}", fontsize=6); clean(a)

a = fig.add_subplot(gs[3, :]); a.axis("off")
a.text(0, 1.0, "10  Caveats the modelling stage has to live with", fontsize=8, fontweight="bold", va="top")
items = [
    (f"Only {d['unique_text']:,} distinct message texts across {d['rows']:,} rows — each repeats about {d['dup_factor']}x.",
     "Split the rows at random and near-identical messages land on both sides, which would make the model look better "
     "than it is. We will split by message cohort, or deduplicate before evaluating."),
    ("Fraud labels are ours, not Banking77's.",
     "Precision and recall on the planted rows tell us whether the pipeline catches patterns we designed ourselves. They "
     "say nothing about real fraud, and the final report will state that plainly."),
    (f"Partitions range {d['part']['min']:,} to {d['part']['max']:,} rows ({d['part_skew']}x).",
     "Weekday-weekend rhythm and branch size drive the spread. It matters next week: with partitions this uneven, some "
     "Spark tasks finish early and sit idle while others run long."),
]
y = 0.80
for head, tail in items:
    a.text(0.01, y, "•", fontsize=8, va="top")
    a.text(0.035, y, head, fontsize=7.3, va="top", fontweight="bold")
    a.text(0.035, y - 0.11, "\n".join(textwrap.wrap(tail, 112)), fontsize=6.8, va="top",
           color=MUTED, linespacing=1.5)
    y -= 0.29
nxt = ("Next, due 29 November: the Spark cluster report. Same log, 1% / 10% / 100% slices against 1, 2 and 4 workers, "
       "recording runtime, shuffle size and partition counts.")
a.text(0.01, y - 0.02, "\n".join(textwrap.wrap(nxt, 108)), fontsize=7.3, va="top",
       fontweight="bold", linespacing=1.5)
pdf.savefig(fig); plt.close(fig)
pdf.close()
print("wrote eda_report.pdf (2 pages)")

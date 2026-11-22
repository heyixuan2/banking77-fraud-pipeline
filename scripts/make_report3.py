"""Two-page EDA progress report, each figure carrying its finding."""
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
bm = json.load(open("bench.json"))
get = lambda e, q: next(r["seconds"] for r in bm if r["engine"] == e and r["query"] == q)

INK, MUTED, ACC = "#1a1a1a", "#5a5a5a", "#c05621"
C = {"washington_dc": "#2b6cb0", "addis_ababa": "#c05621", "yuzhno_sakhalinsk": "#2f855a"}
plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#cccccc", "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 8, "axes.titleweight": "bold"})

def clean(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

def cap(ax, text, width=60):
    """Finding, printed under the axes."""
    ax.text(0, -0.44, "\n".join(textwrap.wrap(text, width)), transform=ax.transAxes,
            fontsize=6.4, va="top", color=MUTED, linespacing=1.55)

pdf = PdfPages("eda_report.pdf")

# ============================== PAGE 1 ==============================
fig = plt.figure(figsize=(8.5, 11))
gs = GridSpec(4, 2, figure=fig, height_ratios=[0.62, 1, 1, 1], hspace=1.15,
              wspace=0.32, left=0.125, right=0.955, top=0.955, bottom=0.165)

ax = fig.add_subplot(gs[0, :]); ax.axis("off")
ax.text(0, 1.0, "Catching Fraud in the Chat Box — EDA Progress Report",
        fontsize=14, fontweight="bold", va="top")
ax.text(0, 0.78, "Yixuan He, Lanbin Fan, Zizheng Wang, Ken Qin   ·   DSAN 6000 Final Project"
                 "   ·   22 November 2026   ·   Page 1 of 2: the data",
        fontsize=8, color=MUTED, va="top")
body = ("Banking77 ships 13,083 messages in about 1 MB, and every one carries a label. A bank's live chatbot stream "
        "looks nothing like that — it is large, and almost none of it has been reviewed. So we replay those messages "
        "into a 90-day log across three branches, each on its own business hours, with 5% review coverage and three "
        "adversarial patterns planted in. Polars read it straight from S3; nothing was copied onto the instance.")
ax.text(0, 0.56, "\n".join(textwrap.wrap(body, 116)), fontsize=7.6, va="top", linespacing=1.5)
ax.text(0, -0.40, f"{d['rows']:,} rows  ·  {d['cols']} columns  ·  {d['part']['n']} partitions  ·  1.99 GB Parquet  ·  "
                  f"{d['days']} days  ·  s3://dsan6000-banking77-team/raw/",
        fontsize=7.6, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f2f5f9", ec="#ccd6e0"))

a = fig.add_subplot(gs[1, 0])
for b in C:
    s = daily.filter(pl.col("branch") == b).sort("date")
    a.plot(s["date"], s["n"], lw=0.9, color=C[b], label=b.replace("_", " "))
a.set_title("1  Daily volume by branch"); a.set_ylabel("messages / day"); a.set_ylim(bottom=0)
a.legend(frameon=False, fontsize=6, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.03))
a.tick_params(axis="x", rotation=30, labelsize=6); clean(a)
cap(a, "Volume holds steady with a clear weekend dip. Washington DC carries half the traffic, "
       "Addis Ababa a third, Yuzhno-Sakhalinsk a fifth.")

a = fig.add_subplot(gs[1, 1])
for b in C:
    s = hour.filter(pl.col("branch") == b).sort("local_h")
    a.plot(s["local_h"], s["n"] / 1000, lw=1.2, color=C[b], label=b.replace("_", " "))
a.set_title("2  Traffic follows each branch's local clock")
a.set_xlabel("hour of day, branch-local"); a.set_ylabel("thousands of messages")
a.set_xticks([0, 6, 12, 18, 23]); a.legend(frameon=False, fontsize=6); clean(a)
cap(a, "All three peak near 13:00 in their own time zone. In UTC those peaks land at 18:00, 10:00 and "
       "02:00, so one branch is busiest while another sleeps.")

a = fig.add_subplot(gs[2, 0])
bs = [x["branch"] for x in d["by_branch"]]
a.barh(range(len(bs)), [x["n"] / 1e6 for x in d["by_branch"]], color="#dde3ea", label="unreviewed (label = NA)")
a.barh(range(len(bs)), [x["labelled"] / 1e6 for x in d["by_branch"]], color="#2b6cb0", label="reviewed")
a.set_yticks(range(len(bs))); a.set_yticklabels([b.replace("_", " ") for b in bs], fontsize=6)
a.set_xlabel("millions of messages"); a.set_title(f"3  Review coverage — {d['labelled_pct']}% carries a label")
a.legend(frameon=False, fontsize=6, loc="lower right"); clean(a)
cap(a, f"{d['labelled']/1e6:.2f}M of {d['rows']/1e6:.1f}M rows have an intent label. The share is the same in every "
       "branch, so no branch is better covered than another. Everything else is what the model must rank.")

a = fig.add_subplot(gs[2, 1])
a.hist(lens["len_chars"], bins=120, range=(0, 1250), color="#2b6cb0", log=True)
a.axvline(999, color="#c53030", lw=1, ls="--")
a.text(1005, a.get_ylim()[1] * 0.15, "999", fontsize=6, color="#c53030")
a.set_title("4  Message length — flooding sits in the tail")
a.set_xlabel("characters"); a.set_ylabel("rows (log)"); clean(a)
cap(a, f"Half the messages are under 54 characters and the ordinary tail dies out near 450. The "
       f"{d['over_999']:,} rows past 999 are the planted token-flooding, with nothing in between.")

a = fig.add_subplot(gs[3, 0])
srt = intents.sort("n", descending=True)
a.bar(range(srt.height), srt["n"], color="#2b6cb0", width=1.0)
a.set_title(f"5  Intent imbalance, reviewed subset ({d['imbalance_ratio']}x)")
a.set_xlabel("77 intents, most to least frequent"); a.set_ylabel("reviewed messages"); clean(a)
cap(a, f"The most common intent ({srt['intent'][0]}, {srt['n'][0]:,}) has {d['imbalance_ratio']}x the messages of the "
       f"rarest ({srt['intent'][-1]}, {srt['n'][-1]:,}). Mild, but enough that accuracy alone will mislead.")

a = fig.add_subplot(gs[3, 1])
a.plot(drift["week"], drift["pct"], marker="o", ms=3, lw=1.2, color=ACC)
a.set_title("6  Vocabulary drift on intent 54")
a.set_ylabel("% of reviewed rows\nusing the new wording")
a.tick_params(axis="x", rotation=30, labelsize=6); clean(a)
cap(a, "Until late October no reviewed message on supported_cards_and_currencies uses the new wording. Five weeks "
       "later 69% do. A model frozen in September would degrade on this intent and nowhere else.")
pdf.savefig(fig); plt.close(fig)

# ============================== PAGE 2 ==============================
fig = plt.figure(figsize=(8.5, 11))
gs = GridSpec(4, 2, figure=fig, height_ratios=[0.52, 1, 1, 1.05], hspace=0.95,
              wspace=0.32, left=0.125, right=0.955, top=0.955, bottom=0.075)
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
ax.text(0, 1.0, "Page 2 — querying the lake, and what is wrong with the data",
        fontsize=12, fontweight="bold", va="top")
body2 = ("We ran one query — messages and reviewed messages per branch — three ways over the same S3 Parquet: once "
         "across all 90 days, once across a three-week slice. The pair separates how fast an engine is from how much "
         "work partition pruning saves it.")
ax.text(0, 0.66, "\n".join(textwrap.wrap(body2, 116)), fontsize=7.6, va="top", linespacing=1.5)

a = fig.add_subplot(gs[1, :])
eng = ["Polars", "DuckDB", "Athena"]; W = 0.35
full = [get(e, "full scan") for e in eng]; pru = [get(e, "3-week slice") for e in eng]
x = range(len(eng))
a.bar([i - W/2 for i in x], full, W, color="#a0aec0", label="all 90 days")
a.bar([i + W/2 for i in x], pru, W, color="#2b6cb0", label="3-week slice (partition pruning)")
for i, (f, pz) in enumerate(zip(full, pru)):
    a.text(i - W/2, f + .1, f"{f:.2f}s", ha="center", fontsize=6.5)
    a.text(i + W/2, pz + .1, f"{pz:.2f}s", ha="center", fontsize=6.5)
a.set_xticks(list(x)); a.set_xticklabels(eng); a.set_ylabel("seconds")
a.set_title("7  Same query, three engines — wall-clock"); a.legend(frameon=False, fontsize=6.5); clean(a)
a.text(0, -0.30, "\n".join(textwrap.wrap(
    f"Pruning to three weeks cuts Polars by {full[0]/pru[0]:.1f}x and DuckDB by {full[1]/pru[1]:.1f}x. Athena does not "
    "move, because what it spends here is start-up, not scanning. DuckDB is the fastest of the three at this size.", 128)),
    transform=a.transAxes, fontsize=6.6, va="top", color=MUTED, linespacing=1.55)

a = fig.add_subplot(gs[2, 0]); a.axis("off")
a.text(0, 1.0, "8  What Athena actually read", fontsize=8, fontweight="bold", va="top")
sc_f = next(r["scanned_mb"] for r in bm if r["engine"] == "Athena" and r["query"] == "full scan")
sc_p = next(r["scanned_mb"] for r in bm if r["engine"] == "Athena" and r["query"] == "3-week slice")
for j, (k, v) in enumerate([("all 90 days", f"{sc_f:,.1f} MB"), ("3-week slice", f"{sc_p:,.1f} MB"),
                            ("files on disk", "1,990 MB")]):
    a.text(0, 0.80 - j * 0.12, f"{k:<16}{v}", fontsize=7.4, va="top", family="monospace",
           color=MUTED if j == 2 else INK)
a.text(0, 0.38, "\n".join(textwrap.wrap(
    "Counting all 44.5M rows cost 5.9 MB of reads, because Parquet hands the engine only the two columns the query "
    "names. Narrowing to three weeks cut that by another 4.2x. Athena still pays about 3.9s of start-up no matter how "
    "small the question, which is why the local engines win here and would lose on data 100x larger.", 52)),
    fontsize=6.6, va="top", color=MUTED, linespacing=1.55)

a = fig.add_subplot(gs[2, 1])
a.hist(part["n"] / 1000, bins=30, color="#2f855a")
a.set_title(f"9  Partition sizes — {d['part_skew']}x skew")
a.set_xlabel("thousands of rows per date/branch partition"); a.set_ylabel("partitions")
a.axvline(d["part"]["median"] / 1000, color=INK, lw=1, ls="--")
a.text(d["part"]["median"] / 1000, a.get_ylim()[1] * .92, f"  median {d['part']['median']:,}", fontsize=6); clean(a)
cap(a, f"The 270 partitions run from {d['part']['min']:,} to {d['part']['max']:,} rows. The three clusters are the "
       "three branches; within each, weekends form the lower lobe.")

a = fig.add_subplot(gs[3, :]); a.axis("off")
a.text(0, 1.0, "10  Caveats the modelling stage has to live with", fontsize=8, fontweight="bold", va="top")
items = [
    (f"Only {d['unique_text']:,} distinct message texts across {d['rows']:,} rows — each repeats about {d['dup_factor']}x.",
     "Split the rows at random and near-identical messages land on both sides, which would make the model look better "
     "than it is. We will split by message cohort, or deduplicate before evaluating."),
    ("Fraud labels are ours, not Banking77's.",
     "Precision and recall on the planted rows tell us whether the pipeline catches patterns we designed ourselves. "
     "They say nothing about real fraud, and the final report will state that plainly."),
    (f"Partitions range {d['part']['min']:,} to {d['part']['max']:,} rows ({d['part_skew']}x).",
     "Weekday-weekend rhythm and branch size drive the spread, so the imbalance is structural rather than accidental."),
]
y = 0.84
for head, tail in items:
    a.text(0.008, y, "•", fontsize=8, va="top")
    a.text(0.032, y, head, fontsize=7.3, va="top", fontweight="bold")
    a.text(0.032, y - 0.105, "\n".join(textwrap.wrap(tail, 104)), fontsize=6.8, va="top",
           color=MUTED, linespacing=1.5)
    y -= 0.30
pdf.savefig(fig); plt.close(fig)
pdf.close()
print("wrote eda_report.pdf (2 pages, captioned)")

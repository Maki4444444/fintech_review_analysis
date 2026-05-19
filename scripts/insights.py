"""
scripts/insights.py
Omega Consultancy — Ethiopian Fintech Review Analysis
Task 4: Insights & Recommendations
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "clean", "reviews_final.csv")
OUTPUT_DIR  = os.path.join("notebooks", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BANK_COLORS = {
    "Commercial Bank of Ethiopia": "#1B4F8A",
    "Bank of Abyssinia":           "#E87722",
    "Dashen Bank":                 "#2E8B57",
}
SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral":  "#f39c12",
    "negative": "#e74c3c",
}

BANKS = ["Commercial Bank of Ethiopia", "Bank of Abyssinia", "Dashen Bank"]
BANK_SHORT = {
    "Commercial Bank of Ethiopia": "CBE",
    "Bank of Abyssinia":           "BOA",
    "Dashen Bank":                 "Dashen",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["bank"] = df["bank"].str.strip()
    df["vader_label"] = df["vader_label"].str.lower().str.strip()
    df["distilbert_label"] = df["distilbert_label"].str.lower().str.strip()
    df["year_month"] = df["date"].dt.to_period("M")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. DRIVERS & PAIN POINTS
# ══════════════════════════════════════════════════════════════════════════════

def get_drivers_and_pain_points(df: pd.DataFrame) -> dict:
    """
    For each bank return the top 2 themes by positive reviews (drivers)
    and top 2 themes by negative reviews (pain points), with review counts.
    """
    results = {}
    for bank in BANKS:
        sub = df[df["bank"] == bank]
        pos = (
            sub[sub["vader_label"] == "positive"]
            .groupby("identified_theme")
            .size()
            .sort_values(ascending=False)
        )
        neg = (
            sub[sub["vader_label"] == "negative"]
            .groupby("identified_theme")
            .size()
            .sort_values(ascending=False)
        )
        results[bank] = {
            "drivers":     pos.head(2).to_dict(),
            "pain_points": neg.head(2).to_dict(),
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. BANK COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def bank_comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("bank")
        .agg(
            avg_rating=("rating", "mean"),
            avg_vader_score=("vader_score", "mean"),
            total_reviews=("review_id", "count"),
            pct_positive=("vader_label", lambda x: (x == "positive").mean() * 100),
            pct_negative=("vader_label", lambda x: (x == "negative").mean() * 100),
        )
        .reset_index()
        .round(3)
    )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 4. RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

RECOMMENDATIONS = {
    "Commercial Bank of Ethiopia": [
        {
            "title": "Stabilise the transfer & payment flow",
            "detail": (
                "Slow and failed transactions are the single largest driver of 1-star reviews. "
                "Implement async transaction processing with real-time push notifications so users "
                "know immediately whether a transfer succeeded or is pending — eliminating the "
                "'did it go through?' anxiety that leads to duplicate submissions and complaints."
            ),
        },
        {
            "title": "Invest in login reliability & biometric onboarding",
            "detail": (
                "OTP delivery failures and session timeouts disproportionately affect first-time "
                "users. Introduce a fallback OTP channel (email + SMS) and surface fingerprint/face "
                "login prominently during onboarding to reduce drop-off at the authentication step."
            ),
        },
    ],
    "Bank of Abyssinia": [
        {
            "title": "Overhaul app stability — prioritise crash-free sessions",
            "detail": (
                "BOA has the lowest average rating (3.4★) and the highest share of stability "
                "complaints. Commission a crash-analytics integration (Firebase Crashlytics or "
                "equivalent), triage the top-5 crash signatures, and ship hotfixes before the next "
                "feature cycle. Target a crash-free session rate above 99%."
            ),
        },
        {
            "title": "Rebuild customer-support responsiveness inside the app",
            "detail": (
                "Users frequently mention that complaints go unanswered. Embed a live-chat or "
                "AI-chatbot triage layer within the app to deflect common queries (balance, "
                "transaction status, OTP issues) and route complex cases to human agents with "
                "full context — reducing resolution time and review-store frustration."
            ),
        },
    ],
    "Dashen Bank": [
        {
            "title": "Capitalise on UI satisfaction — extend it to advanced features",
            "detail": (
                "Dashen scores well on interface and ease-of-use. Leverage this goodwill by "
                "rolling out the most-requested advanced features (scheduled transfers, spend "
                "analytics, card freeze/unfreeze) within the same polished UI — converting "
                "satisfied users into power users and raising retention."
            ),
        },
        {
            "title": "Reduce network-error friction for low-connectivity users",
            "detail": (
                "A meaningful share of negative reviews cite connectivity and timeout errors, "
                "reflecting Ethiopia's variable network conditions. Implement offline queueing "
                "for initiated transfers and a lightweight mode that degrades gracefully on "
                "2G/3G — a competitive differentiator in the Ethiopian market."
            ),
        },
    ],
}


def print_recommendations():
    for bank, recs in RECOMMENDATIONS.items():
        print(f"\n{'═'*60}")
        print(f"  {BANK_SHORT[bank]} — {bank}")
        print(f"{'═'*60}")
        dp = get_drivers_and_pain_points(load_data())
        print(f"  Drivers:     {list(dp[bank]['drivers'].keys())}")
        print(f"  Pain Points: {list(dp[bank]['pain_points'].keys())}")
        print("  Recommendations:")
        for i, r in enumerate(recs, 1):
            print(f"    {i}. {r['title']}")
            print(f"       {r['detail'][:120]}...")


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi":      150,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":     "DejaVu Sans",
    })


# ── Plot 1: Sentiment Distribution by Bank ────────────────────────────────────
def plot_sentiment_distribution(df: pd.DataFrame, save: bool = True):
    set_style()
    order = ["positive", "neutral", "negative"]
    counts = (
        df.groupby(["bank", "vader_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(pct))
    for sentiment in order:
        vals = pct[sentiment].values
        bars = ax.bar(
            [BANK_SHORT[b] for b in pct.index],
            vals,
            bottom=bottom,
            color=SENTIMENT_COLORS[sentiment],
            label=sentiment.capitalize(),
            width=0.5,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, val in zip(bars, vals):
            if val > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                )
        bottom += vals

    ax.set_ylabel("Share of Reviews (%)")
    ax.set_title("Sentiment Distribution by Bank\n(VADER labels)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "01_sentiment_distribution.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Plot 2: Rating Distribution per Bank (Boxplot) ────────────────────────────
def plot_rating_distribution(df: pd.DataFrame, save: bool = True):
    set_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    data_by_bank = [df[df["bank"] == b]["rating"].values for b in BANKS]
    bp = ax.boxplot(
        data_by_bank,
        labels=[BANK_SHORT[b] for b in BANKS],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.4),
        widths=0.45,
    )
    for patch, bank in zip(bp["boxes"], BANKS):
        patch.set_facecolor(BANK_COLORS[bank])
        patch.set_alpha(0.85)

    # overlay jittered points
    for i, (bank, bdata) in enumerate(zip(BANKS, data_by_bank), 1):
        jitter = np.random.uniform(-0.15, 0.15, size=len(bdata))
        ax.scatter(
            i + jitter, bdata,
            alpha=0.18, s=12,
            color=BANK_COLORS[bank], zorder=3,
        )

    ax.set_ylabel("Star Rating")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("Rating Distribution per Bank", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "02_rating_distribution.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Plot 3: Top Theme Frequency per Bank (Horizontal bars) ────────────────────
def plot_theme_frequency(df: pd.DataFrame, top_n: int = 7, save: bool = True):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("Top Themes by Review Frequency per Bank", fontsize=13, fontweight="bold", y=1.02)

    for ax, bank in zip(axes, BANKS):
        sub = df[df["bank"] == bank]
        theme_counts = (
            sub["identified_theme"]
            .value_counts()
            .head(top_n)
            .sort_values()
        )
        bars = ax.barh(
            theme_counts.index,
            theme_counts.values,
            color=BANK_COLORS[bank],
            alpha=0.85,
            edgecolor="white",
        )
        for bar in bars:
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())),
                va="center", fontsize=8.5,
            )
        ax.set_title(BANK_SHORT[bank], fontsize=11, fontweight="bold", color=BANK_COLORS[bank])
        ax.set_xlabel("Number of Reviews")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "03_theme_frequency.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Plot 4: Sentiment Trend Over Time ─────────────────────────────────────────
def plot_sentiment_trend(df: pd.DataFrame, save: bool = True):
    set_style()
    monthly = (
        df.groupby(["year_month", "bank"])["vader_score"]
        .mean()
        .reset_index()
    )
    monthly["year_month_dt"] = monthly["year_month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(11, 5))
    for bank in BANKS:
        sub = monthly[monthly["bank"] == bank].sort_values("year_month_dt")
        ax.plot(
            sub["year_month_dt"],
            sub["vader_score"],
            label=BANK_SHORT[bank],
            color=BANK_COLORS[bank],
            linewidth=2.2,
            marker="o",
            markersize=5,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_ylabel("Avg VADER Sentiment Score")
    ax.set_xlabel("")
    ax.set_title("Monthly Sentiment Trend by Bank", fontsize=13, fontweight="bold")
    ax.legend(title="Bank")
    fig.autofmt_xdate()
    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "04_sentiment_trend.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Plot 5: Bank Comparison Dashboard ─────────────────────────────────────────
def plot_comparison_dashboard(df: pd.DataFrame, save: bool = True):
    set_style()
    summary = bank_comparison_summary(df)
    summary["bank_short"] = summary["bank"].map(BANK_SHORT)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Bank Comparison Dashboard", fontsize=13, fontweight="bold")

    # avg rating
    bars1 = ax1.bar(
        summary["bank_short"],
        summary["avg_rating"],
        color=[BANK_COLORS[b] for b in summary["bank"]],
        width=0.5, alpha=0.88, edgecolor="white",
    )
    ax1.set_ylim(0, 5.5)
    ax1.set_ylabel("Average Star Rating")
    ax1.set_title("Avg Rating")
    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{bar.get_height():.2f}★",
            ha="center", fontsize=10, fontweight="bold",
        )

    # avg vader score
    bars2 = ax2.bar(
        summary["bank_short"],
        summary["avg_vader_score"],
        color=[BANK_COLORS[b] for b in summary["bank"]],
        width=0.5, alpha=0.88, edgecolor="white",
    )
    ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Average VADER Score (−1 to +1)")
    ax2.set_title("Avg Sentiment Score")
    for bar in bars2:
        ypos = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            ypos + 0.01 if ypos >= 0 else ypos - 0.03,
            f"{ypos:.3f}",
            ha="center", fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "05_comparison_dashboard.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df):,} reviews across {df['bank'].nunique()} banks.\n")

    print("── Drivers & Pain Points ──")
    dp = get_drivers_and_pain_points(df)
    for bank, info in dp.items():
        print(f"\n{BANK_SHORT[bank]}:")
        print(f"  Drivers:     {info['drivers']}")
        print(f"  Pain Points: {info['pain_points']}")

    print("\n── Bank Comparison Summary ──")
    print(bank_comparison_summary(df).to_string(index=False))

    print("\n── Generating Visualisations ──")
    plot_sentiment_distribution(df)
    plot_rating_distribution(df)
    plot_theme_frequency(df)
    plot_sentiment_trend(df)
    plot_comparison_dashboard(df)

    print("\n── Recommendations ──")
    for bank, recs in RECOMMENDATIONS.items():
        print(f"\n{BANK_SHORT[bank]}:")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['title']}")

    print("\nDone. All outputs saved to notebooks/outputs/")


if __name__ == "__main__":
    main()
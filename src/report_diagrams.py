"""Generate the diagrams and mock-up charts used in the four weekly reports (reports/*.docx).
Project: Predicting 30-day hospital readmission risk (hypothetical).
All numbers in mock-up charts are synthetic / illustrative."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "diagrams")
os.makedirs(OUT, exist_ok=True)

NAVY, TEAL, AMBER, CORAL, SLATE, LIGHT = "#1F3A5F", "#2A9D8F", "#E9C46A", "#E76F51", "#5C6B7A", "#EEF2F6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11})
rng = np.random.default_rng(42)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc=NAVY, tc="white", fs=9.5, bold=True, style="round,pad=0.02,rounding_size=0.08"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=fc, ec="none"))
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p1, p2, color=SLATE, style="-|>", ls="-", rad=0.0, lw=1.6):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- WEEK 1
def w1_lifecycle():
    fig, ax = canvas(10, 4.2, (0, 10), (0, 4.2))
    steps = [("1. Problem\nDefinition", NAVY), ("2. Data\nCollection", NAVY), ("3. Data\nCleaning", TEAL),
             ("4. Exploratory\nAnalysis", TEAL), ("5. Feature\nEngineering", TEAL), ("6. Modelling", CORAL),
             ("7. Evaluation", CORAL), ("8. Reporting &\nHandover", NAVY)]
    xs = [1.15, 3.7, 6.25, 8.8]
    pos = [(xs[i], 3.0) for i in range(4)] + [(xs[3 - i], 1.2) for i in range(4)]
    for (t, c), (x, y) in zip(steps, pos):
        box(ax, x, y, 2.0, 1.0, t, fc=c)
    for i in range(3):
        arrow(ax, (pos[i][0] + 1.0, 3.0), (pos[i + 1][0] - 1.0, 3.0))
    arrow(ax, (8.8, 2.5), (8.8, 1.7))
    for i in range(4, 7):
        arrow(ax, (pos[i][0] - 1.0, 1.2), (pos[i + 1][0] + 1.0, 1.2))
    arrow(ax, (3.7, 1.72), (6.25, 2.48), color=CORAL, ls="--", rad=-0.25)
    ax.text(3.2, 2.15, "iterate if metrics\nmiss targets", color=CORAL, fontsize=8.5, ha="center")
    save(fig, "w1_lifecycle.png")


def w1_architecture():
    fig, ax = canvas(10.5, 4.8, (0, 10.5), (0, 4.8))
    srcs = ["Admissions &\nDischarge (ADT)", "Diagnoses &\nProcedures (ICD-10)", "Lab Results &\nVitals", "Medications", "Demographics &\nSocial factors"]
    for i, s in enumerate(srcs):
        box(ax, 1.15, 4.2 - i * 0.85, 2.0, 0.7, s, fc=LIGHT, tc=NAVY, fs=8.3)
        arrow(ax, (2.15, 4.2 - i * 0.85), (3.05, 2.5))
    box(ax, 3.95, 2.5, 1.7, 1.2, "Ingestion\n& Validation\n(pandas)", fc=NAVY, fs=8.8)
    box(ax, 5.95, 2.5, 1.7, 1.2, "Clean\nAnalytical\nTable (parquet)", fc=TEAL, fs=8.8)
    box(ax, 7.95, 3.5, 1.7, 0.9, "Feature Store\n& EDA", fc=TEAL, fs=8.8)
    box(ax, 7.95, 1.5, 1.7, 0.9, "Model Training\n(scikit-learn)", fc=CORAL, fs=8.8)
    box(ax, 9.75, 2.5, 1.3, 1.2, "Risk Scores\n& Report", fc=NAVY, fs=8.8)
    arrow(ax, (4.8, 2.5), (5.1, 2.5)); arrow(ax, (6.8, 2.8), (7.1, 3.4)); arrow(ax, (7.95, 3.05), (7.95, 1.95))
    arrow(ax, (8.8, 1.6), (9.1, 2.3)); arrow(ax, (8.8, 3.4), (9.1, 2.7))
    ax.text(5.25, 0.35, "De-identified data only  |  version-controlled code (Git)  |  reproducible environment (conda / requirements.txt)",
            ha="center", fontsize=8.5, color=SLATE, style="italic")
    save(fig, "w1_architecture.png")


def w1_gantt():
    tasks = [("Problem framing & literature review", 0, 3), ("Data sourcing & data dictionary", 3, 3),
             ("Data cleaning & validation", 6, 5), ("Exploratory data analysis", 11, 5),
             ("Feature engineering", 16, 4), ("Baseline + advanced modelling", 20, 6),
             ("Evaluation & fairness checks", 26, 3), ("Report, visuals & presentation", 29, 4)]
    colors = [NAVY, NAVY, TEAL, TEAL, TEAL, CORAL, CORAL, NAVY]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, ((t, s, d), c) in enumerate(zip(tasks, colors)):
        ax.barh(i, d, left=s, color=c, height=0.55)
        ax.text(s + d + 0.3, i, f"{d} h", va="center", fontsize=8.5, color=SLATE)
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t for t, _, _ in tasks]); ax.invert_yaxis()
    ax.set_xlabel("Cumulative effort (hours)"); ax.set_xlim(0, 35)
    for m, lab in [(6, "M1 data ready"), (16, "M2 EDA done"), (29, "M3 model frozen"), (33, "M4 delivered")]:
        ax.axvline(m, color=AMBER, ls="--", lw=1)
        ax.text(m, -0.9, lab, fontsize=7.8, ha="center", color="#8a6d00")
    ax.set_title("Project timeline: 33 planned hours (+2 h contingency)", pad=18)
    save(fig, "w1_gantt.png")


def w1_risk():
    fig, ax = plt.subplots(figsize=(7, 5))
    grid = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("r", ["#DFF3EE", AMBER, CORAL])
    ax.imshow(grid, cmap=cmap, origin="lower", extent=(0, 3, 0, 3), alpha=0.85)
    risks = [(2.5, 2.5, "R1 Missing /\nincomplete records"), (1.5, 2.55, "R2 Class\nimbalance"),
             (2.5, 1.5, "R3 Data\nleakage"), (0.5, 1.5, "R4 Scope\ncreep"),
             (1.5, 0.5, "R5 Tool /\nenv. issues"), (1.5, 1.5, "R6 Model\nbias")]
    for x, y, t in risks:
        ax.text(x, y, t, ha="center", va="center", fontsize=8.5, color=NAVY, fontweight="bold")
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_yticks([0.5, 1.5, 2.5]); ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Impact"); ax.set_ylabel("Likelihood"); ax.set_title("Risk assessment matrix")
    for s in ax.spines.values(): s.set_visible(False)
    save(fig, "w1_risk.png")


# ---------------------------------------------------------------- WEEK 2
def w2_workflow():
    fig, ax = canvas(10.5, 3.6, (0, 10.5), (0, 3.6))
    steps = ["Load &\nInspect", "Data Quality\nAudit", "Clean &\nTransform", "Univariate", "Bivariate", "Multivariate", "Document\n& Report"]
    cols = [NAVY, NAVY, TEAL, TEAL, TEAL, TEAL, CORAL]
    for i, (s, c) in enumerate(zip(steps, cols)):
        x = 0.8 + i * 1.48
        box(ax, x, 1.9, 1.25, 1.0, s, fc=c, fs=8.6)
        if i < len(steps) - 1:
            arrow(ax, (x + 0.63, 1.9), (x + 0.85, 1.9))
    subs = [".shape .dtypes\n.head() .info()", "missing %, dupes,\nranges, outliers", "impute, cap,\nencode, cast",
            "distributions,\nsummary stats", "correlation,\ngroup compares", "heatmaps, PCA,\ninteractions", "notebook +\nfindings log"]
    for i, s in enumerate(subs):
        ax.text(0.8 + i * 1.48, 0.85, s, ha="center", va="center", fontsize=7.6, color=SLATE)
    arrow(ax, (9.68, 2.45), (3.76, 2.45), color=CORAL, ls="--", rad=0.25)
    ax.text(6.7, 3.4, "new questions loop back to cleaning", fontsize=8, color=CORAL, ha="center")
    save(fig, "w2_workflow.png")


def w2_chart_selector():
    fig, ax = canvas(10.5, 5, (0, 10.5), (0, 5))
    box(ax, 5.25, 4.4, 3.2, 0.7, "What question am I answering?", fc=NAVY)
    q = [(1.45, "Distribution\n(one variable)"), (4.0, "Comparison\n(across groups)"), (6.5, "Relationship\n(two+ variables)"), (9.05, "Change over\ntime")]
    a = [["Histogram / KDE", "Box / violin plot", "Bar of counts"],
         ["Grouped bar chart", "Box plot by group", "Dot / lollipop"],
         ["Scatter (+trend)", "Correlation heatmap", "Pair plot"],
         ["Line chart", "Area chart", "Rolling average"]]
    for (x, t), opts in zip(q, a):
        arrow(ax, (5.25, 4.05), (x, 3.55))
        box(ax, x, 3.15, 2.2, 0.75, t, fc=TEAL, fs=9)
        for j, o in enumerate(opts):
            box(ax, x, 2.15 - j * 0.68, 2.2, 0.52, o, fc=LIGHT, tc=NAVY, fs=8.6, bold=False)
        arrow(ax, (x, 2.77), (x, 2.43))
    save(fig, "w2_chart_selector.png")


def w2_gallery():
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    age = np.clip(rng.normal(66, 14, 1500), 18, 99)
    los = rng.gamma(2.2, 2.2, 1500)
    readm = rng.random(1500) < (0.08 + 0.004 * (age - 50).clip(0) + 0.01 * los).clip(0, 0.6)
    ax = axes[0, 0]; ax.hist(age, bins=30, color=NAVY, alpha=0.9); ax.set_title("Histogram: patient age"); ax.set_xlabel("Age (years)")
    ax = axes[0, 1]
    bp = ax.boxplot([los[~readm], los[readm]], tick_labels=["Not readmitted", "Readmitted"], patch_artist=True, widths=0.5)
    for p, c in zip(bp["boxes"], [TEAL, CORAL]): p.set_facecolor(c)
    ax.set_title("Box plot: length of stay by outcome"); ax.set_ylabel("Days")
    ax = axes[0, 2]; ax.scatter(age, los, s=6, alpha=0.35, c=np.where(readm, CORAL, TEAL))
    ax.set_title("Scatter: age vs length of stay"); ax.set_xlabel("Age"); ax.set_ylabel("LOS (days)")
    ax = axes[1, 0]
    names = ["Age", "LOS", "Prior adm.", "Meds", "Comorb.", "HbA1c"]
    c = np.array([[1, .22, .31, .28, .45, .12], [.22, 1, .35, .40, .38, .09], [.31, .35, 1, .33, .41, .15],
                  [.28, .40, .33, 1, .52, .21], [.45, .38, .41, .52, 1, .26], [.12, .09, .15, .21, .26, 1]])
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels(names, fontsize=8); ax.set_title("Heatmap: correlation matrix")
    for i in range(6):
        for j in range(6): ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax = axes[1, 1]
    cats = ["Home", "Home\nhealth", "Skilled\nnursing", "Rehab"]; rates = [9.8, 14.2, 21.5, 17.1]
    ax.bar(cats, rates, color=[TEAL, TEAL, CORAL, AMBER]); ax.set_title("Bar: readmission % by discharge"); ax.set_ylabel("%")
    ax = axes[1, 2]
    m = np.arange(1, 25); y = 14 + 1.5 * np.sin(m / 2) + rng.normal(0, .5, 24) - m * 0.06
    ax.plot(m, y, color=NAVY, lw=1.8, marker="o", ms=3); ax.set_title("Line: monthly readmission rate"); ax.set_xlabel("Month"); ax.set_ylabel("%")
    fig.suptitle("Illustrative visualisation gallery (synthetic data)", fontweight="bold", color=SLATE)
    fig.tight_layout()
    save(fig, "w2_gallery.png")


def w2_missing():
    fig, ax = canvas(10.5, 4.6, (0, 10.5), (0, 4.6))
    box(ax, 5.25, 4.1, 3.4, 0.65, "Column has missing values", fc=NAVY)
    box(ax, 5.25, 3.05, 3.4, 0.7, "Why is it missing?\n(MCAR / MAR / MNAR)", fc=TEAL, fs=9)
    arrow(ax, (5.25, 3.77), (5.25, 3.42))
    opts = [(1.6, "> 60% missing", "Drop column or keep\nonly a 'missing' flag"),
            (4.1, "< 5%, random", "Drop rows or\nsimple impute"),
            (6.6, "5-60%, MAR", "Median / mode or\nKNN / iterative imputer"),
            (9.1, "Informative\n(MNAR)", "Impute + add\nis_missing indicator")]
    for x, cond, act in opts:
        arrow(ax, (5.25, 2.68), (x, 2.15))
        box(ax, x, 1.8, 2.2, 0.62, cond, fc=AMBER, tc=NAVY, fs=8.8)
        box(ax, x, 0.75, 2.2, 0.85, act, fc=LIGHT, tc=NAVY, fs=8.3, bold=False)
        arrow(ax, (x, 1.48), (x, 1.2))
    save(fig, "w2_missing.png")


def w2_outliers():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.concatenate([rng.gamma(2.2, 2.2, 800), [38, 42, 47, 55]])
    q1, q3 = np.percentile(x, [25, 75]); hi = q3 + 1.5 * (q3 - q1)
    axes[0].hist(x, bins=50, color=TEAL); axes[0].axvline(hi, color=CORAL, ls="--")
    axes[0].text(hi + 1, axes[0].get_ylim()[1] * 0.8, f"IQR fence\n= {hi:.1f} days", color=CORAL, fontsize=8.5)
    axes[0].set_title("Detect: IQR rule on length of stay"); axes[0].set_xlabel("Days")
    axes[1].hist(np.clip(x, None, hi), bins=40, color=NAVY); axes[1].set_title("Treat: winsorised (capped) at fence"); axes[1].set_xlabel("Days")
    fig.tight_layout(); save(fig, "w2_outliers.png")


# ---------------------------------------------------------------- WEEK 3
def w3_pipeline():
    fig, ax = canvas(11, 4.6, (0, 11), (0, 4.6))
    top = [("Raw\nData", NAVY), ("Train / Val / Test\nsplit (70/15/15)", NAVY), ("Preprocessing\nPipeline", TEAL),
           ("Feature\nSelection", TEAL), ("Model Training\n(5-fold CV)", CORAL)]
    for i, (t, c) in enumerate(top):
        x = 1.0 + i * 2.2
        box(ax, x, 3.6, 1.8, 0.95, t, fc=c, fs=8.8)
        if i < 4: arrow(ax, (x + 0.9, 3.6), (x + 1.3, 3.6))
    bot = [("Monitoring &\nRetraining", NAVY), ("Deployment\n(API / batch)", NAVY), ("Final Test-set\nEvaluation", CORAL),
           ("Threshold &\nCalibration", CORAL), ("Hyperparameter\nTuning", CORAL)]
    for i, (t, c) in enumerate(bot):
        x = 1.0 + i * 2.2
        box(ax, x, 1.5, 1.8, 0.95, t, fc=c, fs=8.8)
        if i < 4: arrow(ax, (x + 1.3, 1.5), (x + 0.9, 1.5))
    arrow(ax, (9.8, 3.12), (9.8, 1.98))
    arrow(ax, (1.0, 1.98), (1.0, 3.12), color=CORAL, ls="--")
    ax.text(1.15, 2.55, "drift\ndetected", fontsize=8, color=CORAL)
    ax.text(5.5, 0.55, "Test set is touched exactly once - after all tuning decisions are frozen", ha="center",
            fontsize=8.8, color=SLATE, style="italic")
    save(fig, "w3_pipeline.png")


def w3_cv():
    fig, ax = canvas(10, 3.8, (0, 10), (0, 3.8))
    for k in range(5):
        y = 3.2 - k * 0.6
        ax.text(0.55, y, f"Fold {k + 1}", va="center", fontsize=9, color=NAVY, fontweight="bold")
        for j in range(5):
            c = CORAL if j == k else TEAL
            ax.add_patch(FancyBboxPatch((1.3 + j * 1.35, y - 0.22), 1.25, 0.44, boxstyle="round,pad=0.01", fc=c, ec="none"))
            ax.text(1.3 + j * 1.35 + 0.62, y, "Validate" if j == k else "Train", ha="center", va="center", color="white", fontsize=8.3)
    ax.add_patch(FancyBboxPatch((8.25, 0.4), 1.4, 2.98, boxstyle="round,pad=0.01", fc=NAVY, ec="none"))
    ax.text(8.95, 1.9, "Held-out\nTEST\n(15%)\n\nnever used\nin CV", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.text(4.6, 0.12, "Stratified: each fold keeps the ~15% readmission rate  |  score = mean ± std across 5 folds",
            ha="center", fontsize=8.3, color=SLATE)
    save(fig, "w3_cv.png")


def w3_models():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    models = ["Logistic Regression\n(baseline)", "Decision Tree", "Random Forest", "XGBoost /\nLightGBM"]
    auc = [0.68, 0.64, 0.73, 0.76]; err = [0.012, 0.02, 0.011, 0.010]
    ax.barh(models, auc, xerr=err, color=[SLATE, SLATE, TEAL, CORAL], height=0.55, capsize=4)
    for i, v in enumerate(auc): ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(0.70, color=AMBER, ls="--"); ax.text(0.702, 3.45, "target ROC-AUC 0.70", fontsize=8, color="#8a6d00")
    ax.set_xlim(0.5, 0.85); ax.set_xlabel("Expected cross-validated ROC-AUC (planning estimate)"); ax.invert_yaxis()
    ax.set_title("Planned model comparison - expected ranges from published readmission studies")
    save(fig, "w3_models.png")


def w3_confusion():
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    m = np.array([[2210, 340], [190, 260]])
    ax.imshow(m, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("t", ["#F4F7FA", TEAL]))
    labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{m[i, j]}", ha="center", va="center", fontsize=10, fontweight="bold", color=NAVY)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: No", "Pred: Readmit"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: No", "Actual: Readmit"])
    ax.set_title("Illustrative confusion matrix (n = 3,000)")
    for s in ax.spines.values(): s.set_visible(False)
    save(fig, "w3_confusion.png")


def w3_mlops():
    fig, ax = canvas(8.5, 6, (-0.25, 8.25), (-0.25, 5.75))
    c, r = (4, 2.75), 2.25
    stages = ["Train &\nRegister", "Validate\n& Approve", "Deploy\n(FastAPI)", "Score\nPatients", "Monitor\nDrift & AUC", "Retrain\nTrigger"]
    cols = [CORAL, CORAL, NAVY, NAVY, TEAL, TEAL]
    pts = []
    for i, (s, col) in enumerate(zip(stages, cols)):
        a = np.pi / 2 - i * 2 * np.pi / 6
        x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a); pts.append((x, y))
        box(ax, x, y, 1.6, 0.8, s, fc=col, fs=8.6)
    for i in range(6):
        p1, p2 = np.array(pts[i]), np.array(pts[(i + 1) % 6]); d = (p2 - p1) / np.linalg.norm(p2 - p1)
        arrow(ax, tuple(p1 + d * 0.75), tuple(p2 - d * 0.75), rad=-0.15)
    ax.text(4, 2.75, "MLOps\nlifecycle", ha="center", va="center", fontsize=11, fontweight="bold", color=SLATE)
    save(fig, "w3_mlops.png")


def w3_roc_pr():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    f = np.linspace(0, 1, 200)
    for g, lab, col in [(0.45, "Gradient boosting (AUC≈0.76)", CORAL), (0.6, "Logistic reg. (AUC≈0.68)", SLATE)]:
        axes[0].plot(f, f ** g, color=col, lw=2, label=lab)
    axes[0].plot([0, 1], [0, 1], "k:", lw=1, label="Random (0.50)")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate (recall)"); axes[0].set_title("ROC curve (illustrative)"); axes[0].legend(fontsize=8)
    rec = np.linspace(0.01, 1, 200)
    axes[1].plot(rec, 0.15 + 0.5 * (1 - rec) ** 1.6, color=CORAL, lw=2, label="Gradient boosting")
    axes[1].plot(rec, 0.15 + 0.32 * (1 - rec) ** 1.6, color=SLATE, lw=2, label="Logistic reg.")
    axes[1].axhline(0.15, color="k", ls=":", lw=1, label="Prevalence (15%)")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-Recall curve (illustrative)"); axes[1].legend(fontsize=8)
    fig.tight_layout(); save(fig, "w3_roc_pr.png")


# ---------------------------------------------------------------- WEEK 4
def w4_kpis():
    fig, ax = canvas(11, 2.1, (0, 11), (0, 2.1))
    kpis = [("15.2%", "Baseline 30-day\nreadmission rate", NAVY), ("0.76", "Model ROC-AUC\n(held-out test)", TEAL),
            ("58%", "Readmissions caught\nin top-20% risk tier", CORAL), ("~220", "Readmissions avoidable\nper year (est.)", AMBER)]
    for i, (v, l, c) in enumerate(kpis):
        x = 1.4 + i * 2.75
        ax.add_patch(FancyBboxPatch((x - 1.25, 0.15), 2.5, 1.8, boxstyle="round,pad=0.02,rounding_size=0.1", fc=LIGHT, ec=c, lw=2.5))
        ax.text(x, 1.35, v, ha="center", va="center", fontsize=22, fontweight="bold", color=c if c != AMBER else "#B8860B")
        ax.text(x, 0.6, l, ha="center", va="center", fontsize=8.8, color=NAVY)
    save(fig, "w4_kpis.png")


def w4_drivers():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    feats = ["Prior admissions (12 mo)", "Number of chronic conditions", "Length of stay", "Discharged to nursing facility",
             "Number of medications", "Age 75+", "No follow-up booked at discharge", "Abnormal labs at discharge"]
    imp = [0.21, 0.17, 0.13, 0.11, 0.10, 0.09, 0.08, 0.06]
    cols = [CORAL if i < 3 else TEAL for i in range(len(feats))]
    ax.barh(feats[::-1], imp[::-1], color=cols[::-1], height=0.6)
    for i, v in enumerate(imp[::-1]): ax.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_xlabel("Mean |SHAP value| (relative importance)"); ax.set_title("Top drivers of readmission risk (mock-up)")
    save(fig, "w4_drivers.png")


def w4_deciles():
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    d = np.arange(1, 11); r = np.array([2.0, 3.0, 4.0, 5.5, 7.0, 9.0, 12.0, 21.5, 34.0, 54.0])
    ax.bar(d, r, color=[TEAL] * 8 + [CORAL] * 2)
    ax.axhline(15.2, color=NAVY, ls="--"); ax.text(0.6, 16.8, "hospital average 15.2%", color=NAVY, fontsize=8.5)
    for x, v in zip(d, r): ax.text(x, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8.5)
    ax.set_xticks(d); ax.set_xlabel("Predicted risk decile (1 = lowest, 10 = highest)"); ax.set_ylabel("Actual readmission rate")
    ax.set_title("Risk stratification: top 2 deciles hold 58% of all readmissions (mock-up)")
    save(fig, "w4_deciles.png")


def w4_trend():
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    m = np.arange(24); base = 15.5 + 0.8 * np.sin(2 * np.pi * m / 12 + 1) + rng.normal(0, 0.35, 24)
    base[[13]] += 3.0
    ax.plot(m, base, color=NAVY, lw=2, marker="o", ms=3.5)
    ax.annotate("Anomaly: Feb spike\n(flu season +\nstaff shortage)", (13, base[13]), (16.5, 18.4), fontsize=8.5, color=CORAL,
                arrowprops=dict(arrowstyle="->", color=CORAL))
    ax.axvspan(10.5, 14.5, color=AMBER, alpha=0.15); ax.text(12.5, 13.1, "winter", ha="center", fontsize=8, color="#8a6d00")
    ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels(["Jan Y1", "Apr", "Jul", "Oct", "Jan Y2", "Apr", "Jul", "Oct"])
    ax.set_ylabel("Readmission rate (%)"); ax.set_ylim(12.5, 20.5); ax.set_title("Monthly readmission trend with seasonal pattern (mock-up)")
    save(fig, "w4_trend.png")


def w4_cohort():
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    groups = ["Heart failure", "COPD", "Pneumonia", "Diabetes", "Hip/knee\nreplacement"]
    nofu = [24.1, 21.7, 17.2, 16.0, 6.1]; fu = [16.3, 15.1, 12.4, 11.2, 4.8]
    x = np.arange(len(groups)); w = 0.38
    ax.bar(x - w / 2, nofu, w, color=CORAL, label="No follow-up within 7 days")
    ax.bar(x + w / 2, fu, w, color=TEAL, label="Follow-up within 7 days")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Readmission rate (%)"); ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("Early follow-up is associated with lower readmission in every cohort (mock-up)")
    save(fig, "w4_cohort.png")


def w4_impact():
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    labels = ["Avoided readmission\ncosts (220 x $13k)", "Follow-up programme\ncost (5,000 x $150)", "Net annual\nsaving"]
    bottoms = [0, 2.11, 0]; heights = [2.86, 0.75, 2.11]; cols = [TEAL, CORAL, NAVY]
    txt = ["+$2.86M", "-$0.75M", "$2.11M"]
    ax.bar(labels, heights, bottom=bottoms, color=cols, width=0.5)
    for i, (b, h) in enumerate(zip(bottoms, heights)):
        ax.text(i, b + h + 0.06, txt[i], ha="center", fontsize=10, fontweight="bold")
    ax.plot([0.25, 0.75], [2.86, 2.86], color=SLATE, lw=0.8, ls=":"); ax.plot([1.25, 1.75], [2.11, 2.11], color=SLATE, lw=0.8, ls=":")
    ax.set_ylabel("USD millions / year"); ax.set_ylim(0, 3.4)
    ax.set_title("Projected annual financial impact: about 3.8x return on programme cost (illustrative)")
    save(fig, "w4_impact.png")


def w4_storyboard():
    fig, ax = canvas(11, 2.8, (0, 11), (0, 2.8))
    beats = [("1. Hook", "1 in 7 patients\ncomes back\nwithin 30 days", NAVY), ("2. Problem", "Cost, penalties,\npatient harm", NAVY),
             ("3. Insight", "Risk is\nconcentrated &\npredictable", TEAL), ("4. Evidence", "Decile chart +\ntop drivers", TEAL),
             ("5. Action", "Target top 20%\nwith follow-up\nbundle", CORAL), ("6. Ask", "Approve a\n6-month pilot", CORAL)]
    for i, (h, t, c) in enumerate(beats):
        x = 0.95 + i * 1.82
        box(ax, x, 2.25, 1.6, 0.5, h, fc=c, fs=9)
        box(ax, x, 1.15, 1.6, 1.3, t, fc=LIGHT, tc=NAVY, fs=8.3, bold=False)
        if i < 5: arrow(ax, (x + 0.82, 1.15), (x + 1.0, 1.15))
    save(fig, "w4_storyboard.png")


def w4_pyramid():
    fig, ax = canvas(7.5, 4.2, (0, 7.5), (0, 4.2))
    layers = [("Executives", "1 slide: the decision & ROI", CORAL, 3.4), ("Clinical managers", "Who to target, workflow change", TEAL, 4.6),
              ("Care teams", "Patient-level risk list & reasons", NAVY, 5.8), ("Data / IT team", "Full methodology & code", SLATE, 6.8)]
    for i, (who, what, c, w) in enumerate(layers):
        y = 3.55 - i * 0.92
        box(ax, 3.75, y, w, 0.78, f"{who}\n{what}", fc=c, fs=8.5, style="round,pad=0.01,rounding_size=0.04")
    save(fig, "w4_pyramid.png")


if __name__ == "__main__":
    for fn in [w1_lifecycle, w1_architecture, w1_gantt, w1_risk, w2_workflow, w2_chart_selector, w2_gallery, w2_missing,
               w2_outliers, w3_pipeline, w3_cv, w3_models, w3_confusion, w3_mlops, w3_roc_pr, w4_kpis, w4_drivers,
               w4_deciles, w4_trend, w4_cohort, w4_impact, w4_storyboard, w4_pyramid]:
        fn()
    print("figures:", sorted(os.listdir(OUT)))

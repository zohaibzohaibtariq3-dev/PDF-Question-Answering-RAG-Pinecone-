"""Generates docs/architecture.png and docs/architecture.svg (run: python docs/make_diagram.py)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 13)
ax.set_ylim(0, 9.6)
ax.axis("off")

BLUE, GREEN, ORANGE, GREY = "#dbeafe", "#dcfce7", "#ffedd5", "#f1f5f9"
EDGE = "#334155"


def box(x, y, w, h, title, sub="", fc=BLUE, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=EDGE, lw=1.4))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8, color="#475569")
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold" if bold else "normal")


def arrow(p1, p2, label="", rad=0.0, color=EDGE):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=1.6,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.18, label, ha="center",
                fontsize=8.5, color=color, fontweight="bold")


ax.text(6.5, 9.25, "RAG System Architecture — PDF Q&A with Pinecone", ha="center",
        fontsize=15, fontweight="bold")

# column headers
ax.text(2.3, 8.7, "INDEXING PIPELINE (per PDF)", ha="center", fontsize=10.5, fontweight="bold", color="#1d4ed8")
ax.text(10.7, 8.7, "QUERY PIPELINE (per question)", ha="center", fontsize=10.5, fontweight="bold", color="#15803d")

W, H = 3.8, 0.95
LX, RX = 0.4, 8.8
ys = [7.55, 6.3, 5.05, 3.8, 2.55]

left = [
    ("1. PDF Upload", "Streamlit · ≤ 20 MB · validation", "app.py / pdf_loader.py"),
    ("2. Text Extraction", "pypdf · per-page text", "pdf_loader.py"),
    ("3. Cleaning + Chunking", "artifact cleanup · recursive splitter", "text_processing.py"),
    ("4. Embedding Generation", "all-MiniLM-L6-v2 · 384-d · normalised", "embedder.py"),
    ("5. Upsert + Metadata", "page · doc name · chunk id · text", "vector_store.py"),
]
right = [
    ("1. User Question", "empty-query validation", "app.py"),
    ("2. Query Embedding", "same model as indexing", "embedder.py"),
    ("3. Semantic Retrieval", "top-k · cosine · threshold · page filter", "retriever.py"),
    ("4. Grounded Generation", "Groq LLM · temp 0 · context-only prompt", "generator.py"),
    ("5. Answer + Sources", "page · excerpt · similarity · confidence", "pipeline.py / app.py"),
]
for (t, s, f), y in zip(left, ys):
    box(LX, y, W, H, t, s, BLUE)
    ax.text(LX + W - 0.08, y + 0.08, f, ha="right", fontsize=7, style="italic", color="#64748b")
for (t, s, f), y in zip(right, ys):
    box(RX, y, W, H, t, s, GREEN)
    ax.text(RX + W - 0.08, y + 0.08, f, ha="right", fontsize=7, style="italic", color="#64748b")

for col_x in (LX, RX):
    for i in range(4):
        arrow((col_x + W / 2, ys[i]), (col_x + W / 2, ys[i + 1] + H))

# Pinecone block
ax.add_patch(FancyBboxPatch((5.0, 2.6), 3.0, 3.4, boxstyle="round,pad=0.02,rounding_size=0.2",
                            fc=ORANGE, ec="#c2410c", lw=2))
ax.text(6.5, 5.65, "Pinecone", ha="center", fontsize=13, fontweight="bold", color="#9a3412")
ax.text(6.5, 5.28, "serverless index · cosine", ha="center", fontsize=8.5, color="#7c2d12")
for i, name in enumerate(["namespace: doc-A", "namespace: doc-B", "namespace: doc-C"]):
    ax.add_patch(FancyBboxPatch((5.25, 4.35 - i * 0.55), 2.5, 0.42, boxstyle="round,pad=0.01,rounding_size=0.08",
                                fc="white", ec="#c2410c", lw=1))
    ax.text(6.5, 4.56 - i * 0.55, name, ha="center", va="center", fontsize=8.5)
ax.text(6.5, 2.8, "metadata: page · doc · chunk id · text", ha="center", fontsize=7.5, color="#7c2d12")

arrow((LX + W, ys[4] + H / 2), (5.0, 3.3), "upsert", rad=0.15, color="#c2410c")
arrow((8.0, 5.3), (RX, ys[2] + H / 2), "query (top-k)", rad=-0.1, color="#c2410c")
# embedding of query is used for the search
arrow((RX, ys[1] + H / 2), (7.2, 6.0), "", rad=0.2, color="#64748b")
ax.text(8.0, 6.95, "query vector", fontsize=8, color="#64748b", ha="center")
# retrieved chunks to generation

# generation guard note
ax.add_patch(FancyBboxPatch((8.8, 1.2), 3.8, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc="#fef9c3", ec="#a16207", lw=1.2))
ax.text(10.7, 1.9, "Hallucination guard", ha="center", fontsize=9.5, fontweight="bold", color="#854d0e")
ax.text(10.7, 1.5, "no chunk ≥ threshold → fixed fallback,\nLLM is never called", ha="center", fontsize=8, color="#713f12")

# bottom strip
ax.add_patch(FancyBboxPatch((0.4, 0.2), 12.2, 0.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc=GREY, ec="#94a3b8", lw=1))
ax.text(6.5, 0.55, "Cross-cutting: .env secrets (config.py) · error handling (exceptions.py) · "
        "query log JSONL (query_logger.py) · session history (Streamlit)",
        ha="center", va="center", fontsize=8.8, color="#334155")

fig.savefig("docs/architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("docs/architecture.svg", bbox_inches="tight", facecolor="white")
print("saved")

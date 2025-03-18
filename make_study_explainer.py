import matplotlib.pyplot as plt


def fig_explainer():
    colors = {
        "chemo": "#0072B2",  # Blue
        "survey": "#D55E00"  # Vermilion
    }
    plt.rcParams["font.family"] = "Arial"

    # Data
    chemo_sessions = ["Chemo #1", "Chemo #2",
                      "Chemo #3", "Chemo #4", "Chemo #5", "Chemo #6"]
    chemo_positions = [1, 2, 3, 4, 5, 6]

    surveys = {
        "Survey 0\n(Pre-surgery)": 0.2,
        "Survey 1": 0.8,
        "Survey 2": 1.2,
        "Survey 3": 2.8,
        "Survey 4": 3.2,
        "Survey 5": 5.8,
        "Survey 6": 6.2,
        "Survey 7\n(6 months\nafter chemo)": 7,
        "Survey 8\n(12 months\nafter chemo)": 8,
    }

    fig, ax = plt.subplots(figsize=(12, 3))

    # Timeline for chemo sessions
    ax.hlines(1, 0, 8.5, colors="gray", linestyles="dashed", zorder=1)
    for pos in chemo_positions:
        ax.add_patch(plt.Rectangle((pos - 0.1, 0.95), 0.2, 0.1,
                     color=colors["chemo"], label="Chemotherapy Sessions", zorder=2))

    for pos, label in zip(chemo_positions, chemo_sessions):
        ax.text(pos, 1.1, label, ha="center",
                va="bottom", fontsize=14, color=colors["chemo"])

    for survey, position in surveys.items():
        ax.scatter(position, 1, color=colors["survey"], s=50, zorder=3)
        ax.text(position, 0.95, survey, rotation=45, ha="right",
                va="top", fontsize=12, color=colors["survey"])

    ax.set_ylim(0.5, 1.5)
    ax.set_xlim(0, 8.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.set_title("Timeline of Chemotherapy Sessions and Surveys", fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/survey_details.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    fig_explainer()

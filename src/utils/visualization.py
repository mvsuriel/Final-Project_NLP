"""Figure builders for the evaluation scoreboard. Each function takes the dataframe(s) it
plots plus the save_fig / format_model_name display hooks as explicit parameters."""

import json
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import load_predictions


def save_figure(fig, name, fig_dir):
    """Write a Figure to results/figures/ as PNG and PDF, then show. dpi=150 on both formats
    so a rasterized panel (e.g. an imshow) is not downsampled in the PDF."""
    for ext in ('png', 'pdf'):
        fig.savefig(fig_dir / f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.show()


def plot_dataset_overview(test_df: pd.DataFrame, *, save_fig: Callable[[str], None],
                          title: str = 'Class Distribution (Test Set)',
                          fig_name: str = 'class_distribution') -> None:
    """Plot the class distribution and print dataset size and balance. `title` labels the
    chart and `fig_name` names the saved figure when a caller passes a non-test split (e.g.
    the full dataset), so the two views do not overwrite each other on disk."""
    print(f"Total samples: {len(test_df)}")
    print(f"Number of classes: {test_df['status'].nunique()}")

    class_counts = test_df['status'].value_counts()
    class_dist = pd.DataFrame({
        'Count': class_counts,
        'Percentage': (class_counts / len(test_df) * 100).round(2),
    })
    print(f"\nClass distribution:")
    print(class_dist)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=class_counts.values, y=class_counts.index, palette="viridis", hue=class_counts.index, legend=False)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Samples', fontweight='bold', labelpad=10)
    plt.ylabel('Mental Health Condition', fontweight='bold', labelpad=10)

    for i, p in enumerate(ax.patches):
        count = int(p.get_width())
        percentage = class_dist['Percentage'].iloc[i]
        ax.annotate(f"  {count} ({percentage}%)",
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=11, color='#333333')

    # Pad the x-axis so labels are not cut off.
    plt.xlim(0, max(class_counts.values) * 1.2)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    save_fig(fig_name)


def plot_text_length_distributions(test_df: pd.DataFrame, *, save_fig: Callable[[str], None]) -> None:
    """Plot the character-length and word-count distributions for the test set."""
    # Copy so we never mutate the shared guard test_df.
    test_df = test_df.copy()
    test_df['text_length'] = test_df['text'].str.len()
    test_df['word_count'] = test_df['text'].str.split().str.len()

    print("Text length statistics (characters):")
    print(test_df['text_length'].describe().round(2))
    print("\nWord count statistics:")
    print(test_df['word_count'].describe().round(2))

    plt.figure(figsize=(10, 5))
    sns.histplot(test_df['text_length'], bins=50, color='steelblue', kde=True, edgecolor='white')
    plt.axvline(test_df['text_length'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {test_df['text_length'].mean():.0f}")
    plt.title('Distribution of Text Length (characters)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of characters', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    save_fig('text_length_chars_distribution')

    plt.figure(figsize=(10, 5))
    sns.histplot(test_df['word_count'], bins=50, color='mediumseagreen', kde=True, edgecolor='white')
    plt.axvline(test_df['word_count'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {test_df['word_count'].mean():.0f}")
    plt.title('Distribution of Word Count', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of words', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    save_fig('text_length_words_distribution')

    print("\nAverage word count per class:")
    print(test_df.groupby('status')['word_count'].mean().round(1).sort_values(ascending=False))


def plot_full_text_length_distributions(df: pd.DataFrame, *, save_fig: Callable[[str], None]) -> None:
    """Plot character-length and word-count distributions over the full dataset (side-by-side)."""
    # Copy so we never mutate the caller's df.
    df = df.copy()
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()

    print("Text length statistics (characters):")
    print(df['text_length'].describe().round(2))
    print("\nWord count statistics:")
    print(df['word_count'].describe().round(2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title, xlabel in (
        (axes[0], 'text_length', 'Distribution of Text Length (characters)', 'Number of characters'),
        (axes[1], 'word_count', 'Distribution of Word Count', 'Number of words'),
    ):
        ax.hist(df[col], bins=50, color='steelblue', edgecolor='black')
        ax.axvline(df[col].mean(), color='red', linestyle='--', label=f"Mean: {df[col].mean():.0f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.legend()
    plt.tight_layout()
    save_fig('text_length_distribution')

    print("\nAverage word count per class:")
    print(df.groupby('status')['word_count'].mean().round(1).sort_values(ascending=False))


def _baseline_bar(summary: pd.DataFrame, *, save_fig: Callable[[str], None],
                  format_model_name: Callable[..., str] = None) -> None:
    """Part 1 baseline bar chart (accuracy + macro F1). With format_model_name it labels the
    x-axis; without it the experiment names are tidied inline."""
    x = np.arange(len(summary))
    width = 0.38
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width/2, summary['accuracy'], width, label='Accuracy', color='steelblue', edgecolor='none')
    plt.bar(x + width/2, summary['macro_f1'], width, label='Macro F1', color='lightsteelblue', edgecolor='none')

    for xi, acc, f1m in zip(x, summary['accuracy'], summary['macro_f1']):
        if pd.notna(acc):
            plt.text(xi - width/2, acc + 0.02, f"{acc:.3f}", ha='center', fontsize=9)
        if pd.notna(f1m):
            plt.text(xi + width/2, f1m + 0.02, f"{f1m:.3f}", ha='center', fontsize=9)

    if format_model_name is not None:
        plt.xticks(x, [format_model_name(e) for e in summary['experiment']], rotation=15, ha='right')
    else:
        plt.xticks(x, [e.replace('part1_', '').replace('_', ' ') for e in summary['experiment']])
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.title('Part 1 - Baseline Comparison (test set)', pad=15, fontweight='bold')
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    save_fig('part1_baseline_comparison')


def plot_baseline_comparison(load_all_metrics: Callable[[], list], *, save_fig: Callable[[str], None],
                             format_model_name: Callable[..., str]) -> None:
    """Plot the Part 1 baselines: accuracy and macro F1 side by side. Sourced through
    load_all_metrics() so it respects the same provenance guard as the ladder."""
    summary = pd.DataFrame([{
        'experiment': m['name'],
        'accuracy': m.get('accuracy'),
        'macro_f1': m.get('macro_f1'),
        'kappa': m.get('kappa'),
        'suicidal_recall': m.get('per_class', {}).get('suicidal', {}).get('recall'),
        'accuracy_ci95': m.get('accuracy_ci95'),
    } for m in load_all_metrics() if m['name'].startswith('part1')]).sort_values('accuracy').reset_index(drop=True)
    print(summary.to_string(index=False))
    _baseline_bar(summary, save_fig=save_fig, format_model_name=format_model_name)


def plot_random_baseline(y_test: np.ndarray, classes: Sequence[str], *, save_fig: Callable[[str], None],
                         seed: int, n_boot: int) -> None:
    """Plot the random-classifier accuracy and macro-F1 distributions."""
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(seed)

    # Simulate on the test set's class proportions.
    class_proportions = np.array([np.mean(y_test == c) for c in classes])
    class_proportions = class_proportions / class_proportions.sum()

    accuracies, f1_scores_sim = [], []
    for _ in range(n_boot):
        random_preds = rng.choice(classes, size=len(y_test), p=class_proportions)
        accuracies.append(accuracy_score(y_test, random_preds))
        f1_scores_sim.append(f1_score(y_test, random_preds, average='macro'))

    plt.figure(figsize=(10, 5))
    sns.histplot(accuracies, bins=30, color='steelblue', kde=True, edgecolor='white')
    plt.axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(accuracies):.4f}")
    plt.title('Random Classifier Accuracy Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Accuracy', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    save_fig('random_baseline_accuracy_distribution')

    plt.figure(figsize=(10, 5))
    sns.histplot(f1_scores_sim, bins=30, color='mediumseagreen', kde=True, edgecolor='white')
    plt.axvline(np.mean(f1_scores_sim), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(f1_scores_sim):.4f}")
    plt.title('Random Classifier Macro F1 Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Macro F1', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    save_fig('random_baseline_f1_distribution')


def plot_rulebased_confusion(y_test: np.ndarray, classes: Sequence[str],
                             load_all_metrics: Callable[[], list], *,
                             save_fig: Callable[[str], None]) -> None:
    """Plot the rule-based confusion matrix and per-class F1, loading its predictions and
    metrics here. Drawing is shared with plot_rulebased_diagnostics. Skips if either is absent."""
    try:
        y_pred_rules = load_predictions('part1_rule_based')['y_pred'].values
        results_rules = next((m for m in load_all_metrics() if m['name'] == 'part1_rule_based'), None)
        if results_rules:
            plot_rulebased_diagnostics(y_test, y_pred_rules, results_rules, classes, save_fig=save_fig)
    except Exception as e:
        print(f"Could not generate baseline diagnostics: {e}")


def plot_confusion_from_predictions(name: str, classes: Sequence[str], *,
                                    save_fig: Callable[[str], None], title=None) -> None:
    """Row-normalised confusion matrix (seaborn heatmap) from one model's saved test
    predictions (results/predictions/{name}.csv). Each row is a true class and sums to 1, so a
    cell is the proportion of that class sent to each predicted class. Skips quietly if the
    model has no saved predictions."""
    from sklearn.metrics import confusion_matrix

    try:
        preds = load_predictions(name)
    except FileNotFoundError:
        print(f"No saved predictions for {name} - skipped.")
        return

    cm_norm = confusion_matrix(preds['y_true'].values, preds['y_pred'].values,
                               labels=list(classes), normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Proportion of True Class'})
    plt.title(f"Normalized Confusion Matrix: {title or name.replace('_', ' ')}")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_fig(f'{name}_confusion_matrix')


def plot_results_ladder(ladder: pd.DataFrame, *, save_fig: Callable[[str], None],
                        format_model_name: Callable[..., str]) -> None:
    """Plot the results ladder: accuracy with 95% CI whiskers and Cohen's kappa."""
    plot_df = ladder.dropna(subset=['accuracy']).sort_values(by=['part', 'accuracy']).reset_index(drop=True)
    ypos = np.arange(len(plot_df))
    err_low  = (plot_df['accuracy'] - plot_df['acc_ci95_low']).fillna(0)
    err_high = (plot_df['acc_ci95_high'] - plot_df['accuracy']).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(plot_df) + 2))

    # Color bars by part.
    parts = plot_df['part'].unique()
    palette = sns.color_palette("muted", len(parts))
    color_map = {part: palette[i] for i, part in enumerate(parts)}
    bar_colors = plot_df['part'].map(color_map).tolist()

    ax.barh(ypos, plot_df['accuracy'], xerr=[err_low, err_high], color=bar_colors,
            edgecolor='none', capsize=4, label='Accuracy (95% CI)', height=0.6, alpha=0.8)

    kappa_rows = plot_df.dropna(subset=['kappa'])
    ax.scatter(kappa_rows['kappa'], kappa_rows.index, marker='o', s=80, color=sns.color_palette("flare")[2],
               zorder=3, label="Cohen's kappa", edgecolor='white')

    for yi, acc in zip(ypos, plot_df['accuracy']):
        ax.text(acc + 0.02, yi, f'{acc:.3f}', va='center', fontsize=10, color='#333333')

    ax.set_yticks(ypos)
    formatted_names = [format_model_name(e) for e in plot_df['experiment']]
    ax.set_yticklabels(formatted_names)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Score on the frozen test set', labelpad=10, fontweight='bold')
    ax.set_title('Results Ladder: Accuracy and Agreement on Frozen Test Set (Grouped by Part)', pad=15, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', bbox_to_anchor=(0, -0.2), ncol=2, frameon=False)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    save_fig('results_ladder')


def plot_f1_vs_recall(ladder: pd.DataFrame, *, save_fig: Callable[[str], None],
                      format_model_name: Callable[..., str]) -> None:
    """Plot macro F1 against suicidal recall as a grouped bar chart."""
    # Accuracy tracks macro F1 closely here, so we show macro F1 and suicidal recall only.
    plot_df = ladder.dropna(subset=['macro_f1', 'suicidal_recall']).reset_index(drop=True)

    melted_df = plot_df.melt(id_vars=['experiment', 'part'],
                             value_vars=['macro_f1', 'suicidal_recall'],
                             var_name='Metric',
                             value_name='Score')

    melted_df['Metric'] = melted_df['Metric'].replace({'macro_f1': 'Macro F1', 'suicidal_recall': 'Suicidal Recall'})

    # Sort models by macro F1 for a consistent order.
    sorted_plot_df = plot_df.sort_values(by='macro_f1', ascending=True).reset_index(drop=True)
    sorted_models_labels = [format_model_name(row['experiment']) for index, row in sorted_plot_df.iterrows()]

    melted_df['model_label'] = melted_df.apply(lambda row: format_model_name(row['experiment']), axis=1)
    melted_df['model_label'] = pd.Categorical(melted_df['model_label'], categories=sorted_models_labels, ordered=True)
    melted_df = melted_df.sort_values('model_label')

    fig, ax = plt.subplots(figsize=(14, 0.5 * len(sorted_models_labels) + 2))

    sns.barplot(
        data=melted_df,
        x='Score',
        y='model_label',
        hue='Metric',
        palette={'Macro F1': 'steelblue', 'Suicidal Recall': 'indianred'},
        ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Score', fontweight='bold', labelpad=10)
    ax.set_ylabel('Model', fontweight='bold', labelpad=10)
    ax.set_title('Model Performance: Macro F1 vs. Suicidal Recall', pad=15, fontsize=14, fontweight='bold')
    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    save_fig('macro_f1_vs_suicidal_recall_bar_chart')


def plot_top3_vs_soa(ladder: pd.DataFrame, *, save_fig: Callable[[str], None],
                     format_model_name: Callable[..., str]) -> None:
    """Plot the top 3 models against the state-of-the-art reference."""
    soa_df = ladder[ladder['experiment'] == 'part0_soa_reference']
    our_models_df = ladder[(ladder['experiment'] != 'part0_soa_reference')].dropna(subset=['accuracy'])

    top3_df = our_models_df.sort_values('accuracy', ascending=False).head(3)

    comparison_df = pd.concat([soa_df, top3_df]).reset_index(drop=True)

    metrics = ['accuracy', 'macro_f1', 'suicidal_recall']
    metric_labels = ['Accuracy', 'Macro F1', 'Suicidal Recall']

    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    # Red for the SOA, viridis for the rest.
    colors = ['#e74c3c'] + list(sns.color_palette("viridis", 3).as_hex())

    for i, row in comparison_df.iterrows():
        model_name = format_model_name(row['experiment'])
        # Missing metrics (e.g. SOA without suicidal recall) become zero.
        values = [row[m] if pd.notna(row[m]) else 0 for m in metrics]

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i], edgecolor='white', alpha=0.9)

        # Label non-zero bars only.
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                            textcoords='offset points', color='#333333')

    ax.set_ylabel('Score', fontweight='bold', labelpad=10)
    ax.set_title('Top 3 Models vs State-of-the-Art', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    sns.despine(top=True, right=True)
    plt.tight_layout()
    save_fig('top_3_models_vs_soa_comparison')


def plot_recall_heatmap(load_all_metrics: Callable[[], list], classes: Sequence[str],
                        y_test: np.ndarray, *, save_fig: Callable[[str], None],
                        format_model_name: Callable[..., str]) -> None:
    """Plot the per-class recall heatmap across every experiment that reports per-class metrics."""
    sns.set_theme(style="white", font_scale=1.1)

    def row_label(m):
        return format_model_name(m['name'], m.get('n'), len(y_test))

    pc = {row_label(m): {c: m['per_class'].get(c, {}).get('recall', np.nan) for c in classes}
          for m in load_all_metrics() if 'per_class' in m}
    heat = pd.DataFrame(pc).T.loc[:, classes]

    # Sort alphabetically so models group by part.
    heat = heat.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(11, 0.7 * len(heat) + 2))

    sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=1, linecolor='white', cbar_kws={'label': 'Recall Score', 'shrink': 0.8}, ax=ax)

    ax.set_title('Recall per True Class (Grouped by Part)', pad=20, fontsize=15, fontweight='bold')
    plt.xticks(rotation=35, ha='right')
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.tight_layout()
    save_fig('per_class_recall_heatmap')


def plot_suicidal_recall(ladder: pd.DataFrame, y_test: np.ndarray, *, save_fig: Callable[[str], None],
                         format_model_name: Callable[..., str]) -> None:
    """Plot suicidal recall per experiment, the metric later parts must not regress."""
    sui = ladder.dropna(subset=['suicidal_recall']).reset_index(drop=True)
    full_test = (sui['n'] == len(y_test)).to_numpy()

    # Red < 0.5, green at/above, grey if scored on a subset (n < full test).
    colors = ['#d65f5f' if v < 0.5 else '#5b9e73' for v in sui['suicidal_recall']]
    colors = [c if full else '#d3d3d3' for c, full in zip(colors, full_test)]
    ylabels = [format_model_name(e, n, len(y_test)) for e, n in zip(sui['experiment'], sui['n'])]

    fig, ax = plt.subplots(figsize=(9, 0.65 * len(sui) + 2))

    bars = ax.barh(np.arange(len(sui)), sui['suicidal_recall'], color=colors, edgecolor='none', height=0.6, alpha=0.9)
    ax.bar_label(bars, fmt='%.2f', padding=6, fontsize=10, color='#333333')

    ax.set_yticks(np.arange(len(sui)))
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, 1.05)

    ax.axvline(0.5, color='#444444', ls='--', lw=1.5, alpha=0.6, zorder=0)

    ax.set_xlabel('Recall on the Suicidal Class', labelpad=10, fontweight='bold')
    ax.set_title('Suicidal Recall by Model (Red < 50%)', pad=15, fontsize=14, fontweight='bold')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    save_fig('suicidal_recall_by_model')


# --- Part 1 figures.

def plot_wordclouds_by_class(train_df: pd.DataFrame, classes: Sequence[str], *,
                             save_fig: Callable[[str], None]) -> None:
    """Plot a 3x3 grid of per-class word clouds from the training split."""
    from wordcloud import WordCloud

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for ax, label in zip(axes, classes):
        text = ' '.join(train_df.loc[train_df['status'] == label, 'text'])
        wc = WordCloud(width=400, height=300, background_color='white',
                       max_words=50, colormap='Blues').generate(text)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(label.upper(), fontsize=14, fontweight='bold')

    for ax in axes:
        ax.axis('off')

    plt.suptitle('Word Clouds per Mental Health Class (training split)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig('wordclouds_by_class')


def plot_random_baseline_distribution(accuracies: Sequence[float], f1_scores_sim: Sequence[float], *,
                                      save_fig: Callable[[str], None]) -> None:
    """Plot the accuracy and macro-F1 distributions of the simulated random classifier."""
    plt.figure(figsize=(12, 4))
    for pos, (vals, title, xlabel) in enumerate((
        (accuracies, 'Random Classifier Accuracy Distribution', 'Accuracy'),
        (f1_scores_sim, 'Random Classifier Macro F1 Distribution', 'Macro F1'),
    ), start=1):
        plt.subplot(1, 2, pos)
        plt.hist(vals, bins=30, color='steelblue', edgecolor='black')
        plt.axvline(np.mean(vals), color='red', linestyle='--', label=f"Mean: {np.mean(vals):.4f}")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    save_fig('random_baseline_distribution')


def plot_rulebased_diagnostics(y_test: np.ndarray, y_pred_rules: np.ndarray, results_rules: dict,
                               classes: Sequence[str], *, save_fig: Callable[[str], None]) -> None:
    """Plot the hand-crafted rule-based classifier's confusion matrix and per-class F1."""
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_rules,
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
        normalize='true',
        values_format='.0%',
    )
    plt.title('Rule-Based Classifier - Confusion Matrix (row-normalised)')
    plt.tight_layout()
    save_fig('rule_based_confusion_matrix')

    per_class_f1 = pd.Series({c: results_rules['per_class'][c]['f1'] for c in classes}).sort_values()
    plt.figure(figsize=(8, 4.5))
    plt.barh(per_class_f1.index, per_class_f1.values, color='steelblue', edgecolor='black')
    for i, v in enumerate(per_class_f1.values):
        plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=10)
    plt.xlim(0, 1)
    plt.xlabel('F1 score')
    plt.title('Rule-Based Classifier - F1 per Class')
    plt.tight_layout()
    save_fig('rule_based_per_class_f1')


def plot_mined_lexicon_k_curve(k_grid: Sequence[int], val_curve: Sequence[float], k_final: int, *,
                               save_fig: Callable[[str], None]) -> None:
    """Plot the mined-lexicon validation-accuracy curve over K with the chosen K marked."""
    plt.figure(figsize=(7, 4))
    plt.plot(k_grid, val_curve, marker='o', color='steelblue')
    plt.axvline(k_final, color='red', linestyle='--', label=f'chosen K={k_final}')
    plt.xscale('log')
    plt.xlabel('K (terms per class, log scale)')
    plt.ylabel('Validation accuracy')
    plt.title('Mined-Lexicon Classifier - choosing K on validation')
    plt.legend()
    plt.tight_layout()
    save_fig('mined_lexicon_k_curve')


def plot_part1_baseline_comparison(summary: pd.DataFrame, *,
                                   save_fig: Callable[[str], None]) -> None:
    """Plot Part 1 baseline accuracy and macro F1 from an in-notebook summary table."""
    _baseline_bar(summary, save_fig=save_fig)


# --- Part 2 figures. These cross-arm/diagnostic plots save via fig_dir, not save_fig.

def plot_part2_augmentation_arms(arms: Sequence[tuple], arm_metrics: Sequence[dict], *,
                                 metrics_dir, fig_dir) -> None:
    """Plot the Part 2b augmentation arms (A/B/C) against the mined-lexicon accuracy line."""
    bars = {'accuracy': [m['accuracy'] for m in arm_metrics],
            'macro F1': [m['macro_f1'] for m in arm_metrics],
            'suicidal recall': [m['per_class']['suicidal']['recall'] for m in arm_metrics]}
    x = np.arange(len(arms))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, vals) in enumerate(bars.items()):
        rects = ax.bar(x + (i - 1) * width, vals, width, label=metric)
        ax.bar_label(rects, fmt='%.3f', fontsize=8)
    lex_acc = json.loads((metrics_dir / 'part1_rule_based_mined.json').read_text())['accuracy']
    ax.axhline(lex_acc, color='red', linestyle='--', linewidth=1,
               label=f'Part 1 mined lexicon accuracy ({lex_acc:.3f})')
    ax.set_xticks(x, [label for label, _ in arms])
    ax.set_ylim(0, 1)
    ax.set_ylabel('score')
    ax.set_title('Part 2b - augmentation arms vs the 32-only baseline')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    save_figure(fig, 'part2_augmentation_arms', fig_dir)


def plot_part2_llmgen_confusion_compare(y_test: np.ndarray, classes: Sequence[str],
                                        run_a_name: str, run_c_name: str, n_synth_per_class: int, *,
                                        metrics_dir, fig_dir) -> None:
    """Plot the row-normalised confusion matrices of arm A (32-only) and arm C (32+LLM-gen)."""
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, name, ttl in zip(axes, (run_a_name, run_c_name),
                             ('A: 32 only', f'C: 32 + {n_synth_per_class}/class LLM-gen')):
        cm = confusion_matrix(y_test, load_predictions(name)['y_pred'], labels=list(classes))
        cm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm, vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        acc = json.loads((metrics_dir / f'{name}.json').read_text())['accuracy']
        ax.set_title(f'{ttl} (acc {acc:.3f})', fontsize=10)
        for r in range(len(classes)):
            for c_ in range(len(classes)):
                ax.text(c_, r, f'{cm[r, c_]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if cm[r, c_] > 0.5 else 'black')
    fig.colorbar(im, ax=axes, shrink=0.8, label='row share')
    fig.suptitle('Part 2d - what the synthetic data changed (row-normalised confusion)')
    save_figure(fig, 'part2_llmgen_confusion_compare', fig_dir)


def plot_part2_synth_length_hist(real_w, synth_w, *, fig_dir) -> None:
    """Plot real vs LLM-generated post-length (word-count) densities."""
    fig2, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, float(real_w.quantile(0.99)), 60)
    ax.hist(real_w, bins=bins, density=True, alpha=0.55, label=f'real train (n={len(real_w):,})')
    ax.hist(synth_w, bins=bins, density=True, alpha=0.55, label=f'synthetic (n={len(synth_w):,})')
    ax.set_xlabel('words per post')
    ax.set_ylabel('density')
    ax.set_title('Part 2d - real vs LLM-generated post lengths')
    ax.legend()
    fig2.tight_layout()
    save_figure(fig2, 'part2_synth_length_hist', fig_dir)


def plot_part2_bias_audit(audit: pd.DataFrame, avail: Sequence[str], slices: dict, *,
                          fig_dir) -> None:
    """Plot accuracy and suicidal recall per gendered-mention slice, one group per model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(slices))
    width = 0.8 / max(len(avail), 1)

    for j, name in enumerate(avail):
        sub = audit[audit['model'] == name].set_index('slice').loc[list(slices)]
        offset = (j - (len(avail) - 1) / 2) * width
        axes[0].bar(x + offset, sub['accuracy'], width, yerr=sub['acc_95half'], capsize=3, label=name, alpha=0.85)
        axes[1].bar(x + offset, sub['suicidal_recall'], width, label=name, alpha=0.85)

    for ax, ttl in zip(axes, ('Accuracy by Group Mention (95% CI)', 'Suicidal Recall by Group Mention')):
        ax.set_xticks(x, list(slices), rotation=20, ha='right')
        ax.set_title(ttl, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)

    axes[0].legend(fontsize=9, loc='lower right')
    fig.suptitle('Bias Audit: Group-Mention Sensitivity', fontsize=14, fontweight='bold')
    sns.despine()
    fig.tight_layout()

    save_figure(fig, 'part2_bias_audit', fig_dir)


# --- Part 3 figures.

def plot_part3_learning_curve(cdf: pd.DataFrame, overlay: Sequence[dict], lexicon: dict,
                              zeroshot: dict, soa: dict, *, fig_dir) -> None:
    """Plot the Part 3 learning curve (accuracy/macro-F1 and suicidal recall vs train size)
    with Part 2 model points and the Part 1 / zero-shot / SOA reference lines overlaid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax1.plot(cdf['n'], cdf['accuracy'], 'o-', color='tab:blue', label='accuracy (Part 3)')
    ax1.plot(cdf['n'], cdf['macro_f1'], 's--', color='tab:cyan', label='macro-F1 (Part 3)')
    ax2.plot(cdf['n'], cdf['suicidal_recall'], 'o-', color='tab:red', label='Part 3 curve')

    for o in overlay:
        ax1.scatter(o['n'], o['accuracy'], marker='D', color='tab:orange', zorder=3)
        ax1.annotate(o['label'], (o['n'], o['accuracy']), fontsize=7,
                     xytext=(4, 4), textcoords='offset points')
        ax2.scatter(o['n'], o['suicidal_recall'], marker='D', color='tab:orange', zorder=3)
        ax2.annotate(o['label'], (o['n'], o['suicidal_recall']), fontsize=7,
                     xytext=(4, 4), textcoords='offset points')
    if overlay:
        ax1.scatter([], [], marker='D', color='tab:orange', label='Part 2 models (overlay)')
        ax2.scatter([], [], marker='D', color='tab:orange', label='Part 2 models (overlay)')

    if lexicon:
        ax1.axhline(lexicon['accuracy'], ls=':', color='grey')
        ax1.annotate(f"mined lexicon {lexicon['accuracy']:.1%} (Part 1)",
                     (cdf['n'].min(), lexicon['accuracy']), fontsize=8,
                     xytext=(2, 3), textcoords='offset points', color='grey')
        sui_bar = lexicon['per_class']['suicidal']['recall']
        ax2.axhline(sui_bar, ls=':', color='grey')
        ax2.annotate(f'lexicon suicidal recall {sui_bar:.2f} (the 0.45 bar)',
                     (cdf['n'].min(), sui_bar), fontsize=8,
                     xytext=(2, 3), textcoords='offset points', color='grey')
    if zeroshot:
        ax1.axhline(zeroshot['accuracy'], ls='--', color='tab:purple', alpha=0.6)
        ax1.annotate(f"zero-shot Gemma {zeroshot['accuracy']:.1%} (2c, n=2,100 subset)",
                     (cdf['n'].min(), zeroshot['accuracy']), fontsize=8,
                     xytext=(2, 3), textcoords='offset points', color='tab:purple')

    # Published BERT SOA on the same 7-class task (Sevinç 2025, Table I), accuracy only:
    # the paper reports one aggregate score, so there is no per-class line.
    if soa:
        ax1.axhline(soa['accuracy'], ls='-.', color='black', alpha=0.7)
        ax1.annotate(f"published BERT SOA {soa['accuracy']:.1%} (Sevinç 2025)",
                     (cdf['n'].min(), soa['accuracy']), fontsize=8,
                     xytext=(2, 3), textcoords='offset points', color='black')

    for ax, ylab in ((ax1, 'score'), (ax2, 'suicidal recall')):
        ax.set_xscale('log')
        ax.set_xlabel('labeled training rows (log scale)')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='lower right')
    ax1.set_title('Accuracy / macro-F1 vs training-set size')
    ax2.set_title('Suicidal recall (ethics-critical) vs training-set size')
    fig.suptitle('Part 3 - learning curve with the Part 2 techniques overlaid')
    # Keep the recipe caveat on the figure (see PART3_PLAN.md section 3c).
    fig.text(0.5, -0.03,
             'Part 2 points: 40-epoch few-shot recipe. Curve points: 3-epoch full-data recipe. '
             'Each regime uses its appropriate recipe; see PART3_PLAN.md section 3c.',
             ha='center', fontsize=8, style='italic')
    fig.tight_layout()
    save_figure(fig, 'part3_learning_curve', fig_dir)
    print('Saved part3_learning_curve (png+pdf).')


def plot_part3_synth_vs_distill_recall(plot_df: pd.DataFrame) -> None:
    """Plot suicidal recall for the synth vs distill stacks at each real-data fraction.
    Shown, not written to disk."""
    fig, ax = plt.subplots(figsize=(8, 5))

    pct_values = sorted(plot_df['pct'].unique())
    n_pct = len(pct_values)
    ind = np.arange(n_pct)

    bar_width = 0.35

    synth_recall = plot_df[plot_df['technique'] == 'synth'].set_index('pct').loc[pct_values]['suicidal_recall'].values
    distill_recall = plot_df[plot_df['technique'] == 'distill'].set_index('pct').loc[pct_values]['suicidal_recall'].values

    rects1 = ax.bar(ind - bar_width/2, synth_recall, bar_width, label='Synth', color='skyblue')
    rects2 = ax.bar(ind + bar_width/2, distill_recall, bar_width, label='Distill', color='lightcoral')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    ax.set_title('Suicidal Recall Comparison: Synth vs Distill Methods')
    ax.set_xlabel('Percentage of Real Data')
    ax.set_ylabel('Suicidal Recall')
    ax.set_xticks(ind)
    ax.set_xticklabels([f'{pct}%' for pct in pct_values])
    ax.legend(title='Technique')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(plot_df['suicidal_recall']) * 1.1 + 0.1)
    plt.tight_layout()
    plt.show()

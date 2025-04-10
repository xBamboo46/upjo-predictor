def format_value(x, format_str="%0.03f"):
    try:
        return format_str % x
    except:
        return str(x)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shap
from shap import Explanation
from shap.plots._labels import labels
from shap.plots._style import get_style

feature_display_names = {
    "回缩速度": "Recoil Speed",
    "曲线面积": "Curve Area",
    "45min下降百分比": "45min Decline (%)"
}

def plot_custom_waterfall(shap_values: Explanation, max_display=10, show=True):
    style = get_style()
    if show is False:
        plt.ioff()

    if not isinstance(shap_values, Explanation):
        raise TypeError("The waterfall plot requires an `Explanation` object.")

    if len(shap_values.shape) != 1:
        raise ValueError("The waterfall plot only supports a single explanation.")

    base_values = float(shap_values.base_values)
    features = shap_values.display_data if shap_values.display_data is not None else shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    if isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(len(values))])

    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts, pos_inds, pos_widths, pos_low, pos_high = [], [], [], [], []
    neg_lefts, neg_inds, neg_widths, neg_low, neg_high = [], [], [], [], []
    loc = base_values + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    num_individual = num_features if num_features == len(values) else num_features - 1

    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)

        name_en = feature_display_names.get(feature_names[order[i]], feature_names[order[i]])
        val = features[order[i]]
        val_str = format_value(float(val), "%0.03f") if np.issubdtype(type(val), np.number) else str(val)
        yticklabels[rng[i]] = f"{name_en} = {val_str}"

        if num_individual != num_features or i + 4 < num_individual:
            plt.plot(
                [loc, loc],
                [rng[i] - 1 - 0.4, rng[i] + 0.4],
                color=style.vlines_color,
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )

    if num_features < len(values):
        yticklabels[0] = f"{len(shap_values) - num_features + 1} other features"
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = (
        pos_lefts
        + list(np.array(pos_lefts) + np.array(pos_widths))
        + neg_lefts
        + list(np.array(neg_lefts) + np.array(neg_widths))
    )
    dataw = np.max(points) - np.min(points)

    label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(
        pos_inds,
        np.array(pos_widths) + label_padding + 0.02 * dataw,
        left=np.array(pos_lefts) - 0.01 * dataw,
        color=style.primary_color_positive,
        alpha=0,
    )
    label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(
        neg_inds,
        np.array(neg_widths) + label_padding - 0.02 * dataw,
        left=np.array(neg_lefts) + 0.01 * dataw,
        color=style.primary_color_negative,
        alpha=0,
    )

    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        plt.arrow(
            pos_lefts[i],
            pos_inds[i],
            max(dist - hl_scaled, 0.000001),
            0,
            head_length=min(dist, hl_scaled),
            color=style.primary_color_positive,
            width=bar_width,
            head_width=bar_width,
        )
        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i],
                pos_inds[i],
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=style.secondary_color_positive,
            )

        # ✅ 动态调整位置：短箭头显示在右边
        if dist < 0.15 * dataw:
            plt.text(
                pos_lefts[i] + dist + 0.01 * dataw,
                pos_inds[i],
                f"{dist:+.3f}",
                ha="left", va="center",
                color=style.primary_color_positive,
                fontsize=12,
            )
        else:
            plt.text(
                pos_lefts[i] + 0.5 * dist,
                pos_inds[i],
                f"{dist:+.3f}",
                ha="center", va="center",
                color=style.text_color,
                fontsize=12,
            )

    for i in range(len(neg_inds)):
        dist = neg_widths[i]
        plt.arrow(
            neg_lefts[i],
            neg_inds[i],
            -max(-dist - hl_scaled, 0.000001),
            0,
            head_length=min(-dist, hl_scaled),
            color=style.primary_color_negative,
            width=bar_width,
            head_width=bar_width,
        )
        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i],
                neg_inds[i],
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=style.secondary_color_negative,
            )

        # ✅ 动态调整位置：短箭头显示在左边
        if -dist < 0.15 * dataw:
            plt.text(
                neg_lefts[i] + dist - 0.01 * dataw,
                neg_inds[i],
                f"{dist:+.3f}",
                ha="right", va="center",
                color=style.primary_color_negative,
                fontsize=12,
            )
        else:
            plt.text(
                neg_lefts[i] + 0.5 * dist,
                neg_inds[i],
                f"{dist:+.3f}",
                ha="center", va="center",
                color=style.text_color,
                fontsize=12,
            )

    plt.yticks(list(range(num_features)), yticklabels[:-1], fontsize=13)
    for i in range(num_features):
        plt.axhline(i, color=style.hlines_color, lw=0.5, dashes=(1, 5), zorder=-1)

    plt.axvline(base_values, 0, 1 / num_features, color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)

    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)

    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks([base_values, base_values + 1e-8])
    ax2.set_xticklabels(["\n$E[f(X)]$", "\n$ = " + format_value(base_values, "%0.03f") + "$"], fontsize=12, ha="left")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    ax3.set_xticks([fx, fx + 1e-8])
    ax3.set_xticklabels(["$f(x)$", "$ = " + format_value(fx, "%0.03f") + "$"], fontsize=12, ha="left")

    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans))
    tick_labels[1].set_color(style.tick_labels_color)

    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-20 / 72.0, 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(22 / 72.0, -1 / 72.0, fig.dpi_scale_trans))
    tick_labels[1].set_color(style.tick_labels_color)

    for i in range(num_features):
        ax.yaxis.get_majorticklabels()[i].set_color(style.tick_labels_color)

    if show:
        plt.show()
    else:
        return plt.gcf()

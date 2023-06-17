import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

entity_order = ["advertiser", "analytic provider", "social media", "content provider", "auth provider", "email service provider"]
datatype_order = ["government identifier", "contact information", "software identifier", "hardware identifier", "protected classification",  "biometric information", "geolocation",  "internet activity"]

shorter_data_mapping = {
    'government identifier': 'government id.',
    'contact information': 'contact info.',
    'software identifier': 'software id.',
    'hardware identifier': 'hardware id.',
    'protected classification': 'protected class.',
    'biometric information': 'biometric info.',
}

shorter_entity_mapping = {
    "analytic provider": "analytic\nprovider",
    "content provider": "content\nprovider",
    "email service provider": "email service\nprovider"
}


def plot_collection(entity_stats):
    total_collection = entity_stats["total"]

    fig = plt.figure(figsize=(4.7, 3.3))
    ax = total_collection.loc[datatype_order[::-1]].rename(shorter_data_mapping).plot.barh()
    ax.bar_label(ax.containers[0], padding = 4)

    ax.set_xlim(0, 3800)
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_xlabel("No. of Policies")
    ax.set_ylabel("Category of Data Types", labelpad=2)
    fig.set_tight_layout(True)


def plot_sharing(entity_stats):
    tmp_df = entity_stats.loc[datatype_order, entity_order].T.copy()
    tmp_df.rename(columns=shorter_data_mapping, index=shorter_entity_mapping, inplace=True)
    #tmp_df2 = ext_entity_stats.copy()

    labels = tmp_df.applymap(lambda x: ("{:,}".format(int(x))))
    #labels = labels + "\n" + tmp_df2.applymap(lambda x: ("({:,})".format(int(x))))

    fig = plt.figure(figsize=(5, 4.5))
    fontsize = 'medium'

    ax = sns.heatmap(tmp_df, cmap='YlOrRd', annot=labels, square=True, annot_kws={'size': "medium"},
                    vmin=0.0, vmax=600,
                    cbar_kws={'shrink': 0.56, 'aspect': 50, 'pad': 0.01}, fmt="")

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=fontsize)
    colorbar.ax.set_ylabel('No. of Policies', size=fontsize, labelpad=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", size=fontsize)

    ax.set_ylabel("Category of Entities", labelpad=0)
    ax.set_xlabel("Category of Data Types", labelpad=0)
    fig.set_tight_layout(True)


def plot_purposes(purpose_stats):
    tmp_df = purpose_stats.loc[datatype_order, :].T
    tmp_df.rename(columns=shorter_data_mapping, inplace=True)

    labels = tmp_df.applymap(lambda x: ("{:,}".format(int(x))))

    fig = plt.figure(figsize=(5, 4.5))
    fontsize = 'medium'

    ax = sns.heatmap(tmp_df, cmap='YlOrRd', annot=labels, square=True, annot_kws={'size': "medium"},
                    vmin=0.0, vmax=1500.0,
                    cbar_kws={'shrink': 0.48, 'aspect': 50, 'pad': 0.01}, fmt="")

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=fontsize)
    colorbar.ax.set_ylabel('No. of Policies', size=fontsize, labelpad=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", size=fontsize)

    ax.set_ylabel("Purpose", labelpad=-5)
    ax.set_xlabel("Category of Data Types", labelpad=0)
    fig.set_tight_layout(True)


if __name__ == "__main__":
    matplotlib.use('Agg')
    matplotlib.rc('font', family='DejaVu Sans', stretch="condensed")

    # python evals/edges/plot.py /.../poligraph-stats.csv poligraph-summarization.pdf
    result_path, out_pdf_path = sys.argv[1:]

    entity_df = pd.read_csv(os.path.join(result_path, 'entity_stats.csv'), index_col=0)
    purpose_df = pd.read_csv(os.path.join(result_path, 'purpose_stats.csv'), index_col=0)

    with PdfPages(out_pdf_path) as pdf_backend:
        plot_collection(entity_df)
        pdf_backend.savefig()

        plot_sharing(entity_df)
        pdf_backend.savefig()

        plot_purposes(purpose_df)
        pdf_backend.savefig()

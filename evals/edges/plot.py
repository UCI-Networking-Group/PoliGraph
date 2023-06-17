import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def plot_collect_edges(df):
    # Statistics of common COLLECT edges
    entities_mapping = {
        "we": "we",
        "UNSPECIFIED_ACTOR": "(unspecified\nthird party)",
        "advertiser": "advertiser",
        "service provider": "service\nprovider",
        "analytic provider": "analytic\nprovider",
        "Google": "Google",
        "business partner": "business\npartner",
        "social media": "social\nmedia",
    }

    dtypes_mapping = {
        "UNSPECIFIED_DATA": "(unspecified data)",
        "personal information": "personal\ninformation",
        "cookie / pixel tag": "cookie /\npixel tag",
        "email address": "email address",
        "non-personal information": "non-personal\ninformation",
        "geolocation": "geolocation",
        "personal identifier": "personal\nidentifier",
        "person name": "person name",
        #"ip address": "ip address",
        #"device identifier": "device\nidentifier",
    }

    tmp_df = df[df.u.isin(entities_mapping) & df.v.isin(dtypes_mapping) & (df.rel == "COLLECT")]
    pv_table = pd.pivot_table(tmp_df, values="count", index="u", columns="v")
    pv_table = pv_table.loc[entities_mapping.keys(), dtypes_mapping.keys()]
    pv_table_purpose = pd.pivot_table(tmp_df, values="purpose_count", index="u", columns="v")
    pv_table_purpose = pv_table_purpose.loc[entities_mapping.keys(), dtypes_mapping.keys()]

    pv_table.rename(columns=dtypes_mapping, index=entities_mapping, inplace=True)
    pv_table_purpose.rename(columns=dtypes_mapping, index=entities_mapping, inplace=True)

    fig = plt.figure(figsize=(5, 6))
    fontsize = 'medium'

    labels = pv_table.fillna(0).applymap(lambda x: ("{:,}".format(int(x))))
    labels = labels + "\n" + pv_table_purpose.fillna(0).applymap(lambda x: ("({:,})".format(int(x))))

    ax = sns.heatmap(pv_table, cmap='YlOrRd', annot=labels, square=True, vmin=0, vmax=3000,
        annot_kws={'size': "small"},
        cbar_kws={'shrink': 0.58, 'aspect': 60, 'pad': 0.01}, fmt="")

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=fontsize, pad=0)
    colorbar.ax.set_ylabel('No. of Policies', size=fontsize, labelpad=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", size=fontsize)

    ax.tick_params(axis='x', which='major', pad=0)
    ax.tick_params(axis='y', which='major', pad=2)

    ax.set_xlabel("Data Type", labelpad=-10)
    ax.set_ylabel("Entity", labelpad=-12)
    fig.set_tight_layout(True)


def plot_subsume_edges(df):
    # Statistics of common SUBSUME edges
    # yq '.links[] | select( .key == "SUBSUM" ) | .source'
    upper_mapping = {
        "personal information": "personal\ninformation",
        "contact information": "contact\ninformation",
        "personal identifier": "personal\nidentifier",
        "identifier": "identifier",
        "device identifier": "device\nidentifier",
        "geolocation": "geolocation",
        "non-personal information": "non-personal\ninformation",
        "cookie / pixel tag": "cookie /\npixel tag",
    }

    # yq '.links[] | select( .key == "SUBSUM" ) | .target'
    lower_mapping = {
        "email address": "email address",
        "person name": "person name",
        "postal address": "postal address",
        "phone number": "phone number",
        "ip address": "ip address",
        "geolocation": "geolocation",
        #"date of birth": "date of birth",
        "advertising id": "advertising id",
        "precise geolocation": "precise\ngeolocation",
        #"anonymous identifier": "anonymous\nidentifier",
    }

    tmp_df = df[df.u.isin(upper_mapping) & df.v.isin(lower_mapping) & (df.rel == "SUBSUM")]
    pv_table = pd.pivot_table(tmp_df, values="count", index="u", columns="v")
    pv_table = pv_table.loc[upper_mapping.keys(), lower_mapping.keys()]
    pv_table.rename(columns=lower_mapping, index=upper_mapping, inplace=True)

    fig = plt.figure(figsize=(6, 4))
    fontsize = 'medium'

    labels = pv_table.fillna(0).applymap(lambda x: ("{:,}".format(int(x))))
    ax = sns.heatmap(pv_table, cmap='YlOrRd', annot=labels, square=True, vmin=0, vmax=500,
        annot_kws={'size': "medium"},
        cbar_kws={'shrink': 1.0, 'aspect': 50, 'pad': 0.01}, fmt="")

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=fontsize)
    colorbar.ax.set_ylabel('No. of Policies', size=fontsize, labelpad=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", size=fontsize)

    ax.tick_params(axis='both', which='major', pad=2)
    ax.set_xlabel("Hyponym", labelpad=-5)
    ax.set_ylabel("Hypernym", labelpad=-12)
    fig.set_tight_layout(True)


if __name__ == "__main__":
    matplotlib.use('Agg')
    matplotlib.rc('font', family='DejaVu Sans', stretch="condensed")

    # python evals/edges/plot.py /.../poligraph-stats.csv plot-edges.pdf
    df_path, out_pdf_path = sys.argv[1:]
    df = pd.read_csv(df_path)

    with PdfPages(out_pdf_path) as pdf_backend:
        plot_collect_edges(df)
        pdf_backend.savefig()

        plot_subsume_edges(df)
        pdf_backend.savefig()

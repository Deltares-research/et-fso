# -*- coding: utf-8 -*-
"""
@author: imhof_rn

Plot KGE per basin and method for UK.
"""

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec


# -------------------------------------------------------------------------------- #
# The initial settings
# -------------------------------------------------------------------------------- #
# Provide the csv file with the benchmark results
benchmark_csv_filename = "p:/11209205-034-et-fso/Verification/Results_Benchmark_Methods.csv"

# TODO: add this to the script as well
# Provide the csv file with the FSO results
# fso_csv_filename = "p:/11209205-034-et-fso/Verification/Results_FSO_validation.csv"

# Also provide the csv with the lat-lon locations of the discharge gauges
gauge_info_csv = "p:/11206558-across/GB/CAMELS-GB/CAMELS_GB_topographic_attributes.csv"

# Define the list of methods to loop trough
methods = ["ksathorfrac100","ksathorfrac_AXA","ksathorfrac_RF","ksathorfrac_BRT"]

# Define which KGE is used (options are kge_orig and kge_modified)
used_kge_metric = "kge_orig"

# Define the output filename where the resulting map will be stored
output_filename = "p:/11209205-034-et-fso/Verification/Figs/KGE_comparison_map_kge_orig.csv"
# output_filename = "p:/11209205-034-et-fso/Verification/Figs/KGE_comparison_map_kge_modified.csv"

# Finally, also provide the text files of the color maps used in Crameri et al. (2020),
# Nature Communications
imola_colormap = "c:/Users/imhof_rn/OneDrive - Stichting Deltares/Documents/PhD/ScientificColourMaps6/imola/DiscretePalettes/imola50.txt"


# -------------------------------------------------------------------------------- #
# Open all results
# -------------------------------------------------------------------------------- #
# Read the benchmark results
df_benchmark_results = pd.read_csv(benchmark_csv_filename)
#TODO: read the FSO results
# df_fso_results = pd.read_csv(fso_csv_filename)

# Also open the csv with the gauge info
df_gauge_info = pd.read_csv(gauge_info_csv)
# Only keep the rows (the gauge_id) that are also present in df_benchmark_results
#Create a boolean mask
mask = df_gauge_info["gauge_id"].isin(df_benchmark_results["basin"])
# Apply the mask to df_gauge_info
df_gauge_info = df_gauge_info[mask]                     

# Get the langitude and latitude values
lats = np.array(df_gauge_info["gauge_lat"])
lons = np.array(df_gauge_info["gauge_lon"])

# Only keep the requested methods and the requested KGE metric
results_for_plotting = dict()
for method in methods:
    results_for_plotting[method] = np.array(
        df_benchmark_results[f"{method}_{used_kge_metric}"]
    )
#TODO: remove this once the results are there..
results_for_plotting["ksathorfrac_FSO"] = results_for_plotting[method].copy()
# results_for_plotting["ksathorfrac_FSO"] = np.array(df_fso_results[used_kge_metric])

methods.append("ksathorfrac_FSO")


# --------------------------------------------------------------------------- #
# Open the colormap(s)
# --------------------------------------------------------------------------- #
# Open the text file in read mode
file = open(imola_colormap, "r")

# Create an empty list to store the last column values
imola_colormap_list = []

# Loop through each line in the file
for line in file:
    # Split the line by whitespace and get the last element
    last_element = line.split()[-1]
    # Append the last element to the list
    if last_element.startswith("#"):
        imola_colormap_list.append(last_element)

# Close the file
file.close()

# Convert the list to a numpy array
imola_colormap_list = np.array(imola_colormap_list)

# We only use 7 out of 50 colors. 
# Use slicing to get every 7th row starting from 0
imola_colormap_final = imola_colormap_list[::7]


# --------------------------------------------------------------------------- #
# Plot it
# --------------------------------------------------------------------------- #
# Set up the figure and subplots grid
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 3, figure=fig) 

# The color and normalization settings of the map
# colors = ['#a50026', '#d73027', '#f46d43', '#e0f3f8', '#abd9e9', "#2171b5", "#08306b"]
cmap = ListedColormap(imola_colormap_final)

# define the bins and normalize
bounds = [-10, -0.4, 0, 0.2, 0.4, 0.6, 0.8, 1]
# bounds = [-10, -0.4, -0.25, 0, 0.25, 0.5, 0.75, 1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Define the labels
labels = ['< -0.4', '-0.4 - 0.0', '0.0 - 0.2', '0.2 - 0.4', '0.4 - 0.6', '0.6 - 0.8', '> 0.8']
# labels = ['< -0.4', '-0.4 - -0.25', '-0.25 - 0.0', '0.0 - 0.25', '0.25 - 0.5', '0.5 - 0.75', '> 0.75'] # Difference label

# Loop through methods and create subplots
for i, method in enumerate(methods):
    ax = fig.add_subplot(gs[i // 3, i % 3])  # Adjust subplot indexing based on grid size

    # Define a Basemap for the UK for each subplot
    m = Basemap(projection='merc', llcrnrlat=49, urcrnrlat=59, llcrnrlon=-11, urcrnrlon=2,
                lat_ts=51, resolution='i', ax=ax) # Pass the axes object to Basemap
    # Resolution = h for higher resolution

    # Draw map details
    m.drawcoastlines(linewidth=0.3,  zorder=3)
    m.drawcountries(linewidth=0.4,  zorder=3)
    m.fillcontinents(color=(0.92, 0.92, 0.92), lake_color='white')
    m.drawmapboundary(fill_color='lightblue')

    # # draw parallels and meridians
    # parallels = np.arange(49., 59., 1.0)
    # meridians = np.arange(-11, 2., 1.0)
    # m.drawparallels(
    #     parallels, 
    #     labels=[True, False, False, False], 
    #     fontsize=17, 
    #     linewidth=0.75, 
    #     color='gray', 
    #     dashes=[1, 3],
    #     )
    # m.drawmeridians(
    #     meridians, 
    #     labels=[False, False, False, True], 
    #     fontsize=17, 
    #     linewidth=0.75, 
    #     color='gray', 
    #     dashes=[1, 3]
    #     )

    # Convert gauge locations to map projection coordinates
    x, y = m(lons, lats)

    ###
    # create the new map
    ###
    # Plot points colored by KGE value for each method
    sc = ax.scatter(
        x, y, c=results_for_plotting[method], cmap=cmap, norm=norm, s=10, zorder=2
        )

    # Set the axis labels
    # ax.set_xlabel('Longitude', fontsize=22, labelpad=30)
    # ax.set_ylabel('Latitude', fontsize=22, labelpad=40);

    # Add a title to each subplot
    ax.set_title(f'Results for {method}')

    # Create the custom legend using Line2D
    legend_elements = []
    for i in range(len(imola_colormap_final)):
        legend_elements.append(Line2D([0], [0], color='w', marker='o', 
                                    markerfacecolor=imola_colormap_final[i], markersize=10, 
                                    label=labels[i])
                            )

    legend1 = ax.legend(
        handles=legend_elements, 
        loc="center", 
        bbox_to_anchor=(1.12, 0.74), 
        ncol=1, 
        frameon=False, 
        title=r"KGE value", 
        title_fontsize=22,
        labelspacing=1.0,
        fontsize=17,
        )      
    #TODO: make the legend work for the entire figure instead of for each sub figure
# plt.savefig(outfile)
# plt.close()                
plt.show()
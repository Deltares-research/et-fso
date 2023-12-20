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
benchmark_csv_filename = "p:/11209205-034-et-fso/Verification/Results_Benchmark_Methods_Optimization1_14trainingbasins.csv"

# Also provide the csv with the lat-lon locations of the discharge gauges
gauge_info_csv = "p:/11206558-across/GB/CAMELS-GB/CAMELS_GB_topographic_attributes.csv"

# Define the list of methods to loop trough
methods = ["ksathorfrac100","ksathorfrac_AXA","ksathorfrac_RF","ksathorfrac_BRT","Test1_14trainingbasins"]

# Define which KGE is used (options are kge_orig and kge_modified)
used_kge_metric = "kge_orig" #"kge_modified"

# Give the shapefile of the UK
uk_shape = "c:/Users/imhof_rn/OneDrive - Stichting Deltares/Documents/SITO/Enabling_Technologies/FSO/ET-FSO-2023/UK_shape"

# Define the output filename where the resulting map will be stored
outputfile = "p:/11209205-034-et-fso/Verification/Figs/KGE_comparison_map_kge_origTest1_14trainingbasins.png"
# outputfile = "p:/11209205-034-et-fso/Verification/Figs/KGE_comparison_map_kge_modified_Test1_14trainingbasins.png"

# Finally, also provide the text files of the color maps used in Crameri et al. (2020),
# Nature Communications
imola_colormap = "c:/Users/imhof_rn/OneDrive - Stichting Deltares/Documents/PhD/ScientificColourMaps6/imola/DiscretePalettes/imola50.txt"


# -------------------------------------------------------------------------------- #
# Open all results
# -------------------------------------------------------------------------------- #
# Read the benchmark results
df_benchmark_results = pd.read_csv(benchmark_csv_filename)

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

# For our information, plot the average KGE value
for method in methods:
    print(f"Mean KGE {method} is: ")
    print(np.nanmean(results_for_plotting[method]))

# -------------------------------------------------------------------------------- #
# Also calculate the difference between the methods and the KsatHorFrac 100 run
# Add it to the results and methods list
# -------------------------------------------------------------------------------- #
methods_new = []
for method in methods:
    if method != "ksathorfrac100":
        results_for_plotting[f"difference_{method}"] = results_for_plotting[method] - results_for_plotting["ksathorfrac100"]
        methods_new.append(f"difference_{method}")

for method in methods_new:
    methods.append(method)


# -------------------------------------------------------------------------------- #
# Open the colormap(s)
# -------------------------------------------------------------------------------- #
###
# Open the imola map
###
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
# Use slicing to get every 8th row starting from 0
imola_colormap_final = imola_colormap_list[::8]


# --------------------------------------------------------------------------- #
# Plot it
# --------------------------------------------------------------------------- #
# Set up the figure and subplots grid
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 5, figure=fig) 

# The color and normalization settings of the map
colors_diff = ['#f4a582','#fddbc7','#f7f7f7','#92c5de','#4393c3','#2166ac','#053061']
colors_kge = imola_colormap_final
cmap_kge = ListedColormap(colors_kge)
cmap_diff = ListedColormap(colors_diff)

# define the bins and normalize
bounds_kge = [-10, -0.4, 0, 0.2, 0.4, 0.6, 0.8, 1]
bounds_diff = [-10, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 1]
norm_kge = mpl.colors.BoundaryNorm(bounds_kge, cmap_kge.N)
norm_diff = mpl.colors.BoundaryNorm(bounds_diff, cmap_diff.N)

# Define the labels
labels_kge = [
    '< -0.4', '-0.4 - 0.0', '0.0 - 0.2', '0.2 - 0.4', '0.4 - 0.6', '0.6 - 0.8', '> 0.8'
    ]
labels_diff = ['< -0.2', '-0.2 - -0.1', '-0.1 - 0.1', '0.1 - 0.2', '0.2 - 0.3', '0.3 - 0.4', '> 0.4']

# Loop through methods and create subplots
for i, method in enumerate(methods):
    # Add one to skip the first column in the second row of the plot (because we don't plot
    # the difference with ksathorfrac 100).
    if i >= 5: 
        i = i+1
    ax = fig.add_subplot(gs[i // 5, i % 5])  # Adjust subplot indexing based on grid size

    # Define a Basemap for the UK for each subplot
    m = Basemap(
        projection='merc', 
        llcrnrlat=49.9, 
        urcrnrlat=59, 
        llcrnrlon=-8, 
        urcrnrlon=1.8,
        lat_ts=51, 
        resolution='i', 
        ax=ax
        )

    # Draw map details
    m.readshapefile(
        uk_shape, "uk", color="grey", linewidth=0.5,
        )

    # Convert gauge locations to map projection coordinates
    x, y = m(lons, lats)

    ###
    # create the new map
    ###
    if i < 5:
        # Plot points colored by KGE value for each method
        sc = ax.scatter(
            x, y, c=results_for_plotting[method], cmap=cmap_kge, norm=norm_kge, 
            s=18, zorder=2, edgecolors="grey", linewidth=0.5,
            )
    else:
        # Plot the difference in KGE for each method
        sc = ax.scatter(
            x, y, c=results_for_plotting[method], cmap=cmap_diff, norm=norm_diff, 
            s=18, zorder=2, edgecolors="grey", linewidth=0.5,
            )

    # Add a title to each subplot
    if i < 5:
        if method == "Test2_30trainingbasins":
            ax.set_title(f'Results for FSO', fontsize=15)
        else:
            ax.set_title(f'Results for {method}', fontsize=15)
    else:
        ax.set_title(f'Difference', fontsize=15)

    # Turn off the frames around the map
    ax.set_frame_on(False)

# Create the custom legend using Line2D
legend_elements_kge = []
for i in range(len(colors_kge)):
    legend_elements_kge.append(Line2D([0], [0], color='white', marker='o', 
                                markerfacecolor=colors_kge[i], 
                                markersize=10, label=labels_kge[i])
                        )

legend_elements_diff = []
for i in range(len(colors_diff)):
    legend_elements_diff.append(Line2D([0], [0], color='white', marker='o', 
                                markerfacecolor=colors_diff[i], 
                                markersize=10, label=labels_diff[i])
                        )


fig.legend(
    handles=legend_elements_kge, 
    loc="upper left", 
    bbox_to_anchor=(0.02, 0.9), 
    ncol=1, 
    frameon=False, 
    title=r"KGE", 
    title_fontsize=15, 
    labelspacing=1.0, 
    fontsize=12,
    )      

fig.legend(
    handles=legend_elements_diff, 
    loc="lower left", 
    bbox_to_anchor=(0.02, 0.1), 
    ncol=1, 
    frameon=False, 
    title=r"KGE difference", 
    title_fontsize=15, 
    labelspacing=1.0, 
    fontsize=12,
    )     

plt.savefig(outputfile)
plt.close()                
# plt.show()
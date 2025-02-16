import dcor
import numpy as np
import matplotlib.pyplot as plt


def dcor_plot_MC(dfsMC, dfProcessesMC, isMCList, bins, outFile):

    # Store distance correlation values and bin ranges
    dcor_results = {}

    for idx, df in enumerate(dfsMC):
        process_name = dfProcessesMC.process[isMCList[idx]]
        print(process_name)

        # Initialize a list to store the distance correlation coefficients for this process
        dcor_results[process_name] = []

        for b_low, b_high in zip(bins[:-1], bins[1:]):
            # Mask for dijet_mass in bin range
            m = (df.dijet_mass > b_low) & (df.dijet_mass < b_high)

            # Ensure proper data type for PNN1 and PNN2
            pnn1 = np.array(df.PNN1[m], dtype=np.float64).flatten()
            pnn2 = np.array(df.PNN2[m], dtype=np.float64).flatten()

            # Compute distance correlation
            if len(pnn1) > 1 and len(pnn2) > 1:  # Avoid computing if data is insufficient
                dcor_coef = dcor.distance_correlation(pnn1, pnn2)
            else:
                dcor_coef = 0.0  # Assign zero if not enough data points

            print("     %.1f < mjj < %.1f : %.5f" % (b_low, b_high, dcor_coef))

            # Append the result to the list for the process
            dcor_results[process_name].append(dcor_coef)

    # Plot all processes in subplots
    num_processes = len(dcor_results)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_processes + cols - 1) // cols  # Calculate required rows
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5), sharey=True, sharex=True)
    axes = axes.flatten()  # Flatten in case we don't fill all grid cells

    for i, (process_name, dcor_values) in enumerate(dcor_results.items()):
        ax = axes[i]
        ax.bar(
            range(len(dcor_values)),
            dcor_values,
            tick_label=[f"{b_low:.0f}-{b_high:.0f}" for b_low, b_high in zip(bins[:-1], bins[1:])],
        )
        if i >= len(dcor_results)-3:
            ax.set_xlabel("Mass Bins (mjj ranges)")
        if i %3==0:
            ax.set_ylabel("Distance Correlation")
        ax.set_title(f"{process_name}")
        ax.set_ylim(0, 0.1)
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(outFile, bbox_inches='tight')



def dcor_plot_Data(dfsData, processNames, isDataList, bins, outFile, nEvents=5000):


    dcor_data_values = []
    for idx, df in enumerate(dfsData):
        print(processNames[isDataList[idx]])
        for b_low, b_high in zip(bins[:-1], bins[1:]):
            m = (df.dijet_mass > b_low) & (df.dijet_mass < b_high)
            dcor_coef = dcor.distance_correlation(np.array(df.PNN1[m].values[:nEvents], dtype=np.float64), np.array(df.PNN2[m].values[:nEvents], dtype=np.float64))
            dcor_data_values.append(dcor_coef)
            print("     %.1f < mjj < %.1f : %.5f"%(b_low, b_high, dcor_coef))


    nbins = len(bins) - 1
    n_datataking = len(dcor_data_values) // nbins

    # Reshape dcor_data_values into a 2D list: each row corresponds to a data-taking period
    dcor_data_values_reshaped = [
        dcor_data_values[i * nbins:(i + 1) * nbins] for i in range(n_datataking)
    ]

    # Plot grouped bar plot
    bar_width = 0.2  # Width of each bar
    x = np.arange(nbins)  # x positions for bins

    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors for different data-taking periods
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:n_datataking]
    labels = [f"{processNames[i]}" for i in range(n_datataking)]

    for i, dcor_values in enumerate(dcor_data_values_reshaped):
        # Offset the x positions for each data-taking period
        ax.bar(x + i * bar_width, dcor_values, bar_width, label=labels[i], color=colors[i])
    ax.legend()

    # Customize the plot
    ax.set_xlabel("Mass Bins (mjj ranges)")
    ax.set_ylabel("Distance Correlation")
    #ax.set_title("Distance Correlation per Mass Bin for Each Data-Taking Period")
    ax.set_xticks(x + bar_width * (n_datataking - 1) / 2)  # Center x-ticks
    ax.set_xticklabels([f"{b_low:.0f}-{b_high:.0f}" for b_low, b_high in zip(bins[:-1], bins[1:])], rotation=45)
    if outFile is not None:
        fig.savefig(outFile, bbox_inches='tight')
    return np.array(dcor_data_values)






def dpearson_plot_Data(dfsData, processNames, isDataList, bins, outFile):
    from scipy.stats import pearsonr

    dpearson_data_values = []
    for idx, df in enumerate(dfsData):
        print(processNames[isDataList[idx]])
        print("Bin Mass \t : Pearson")
        for b_low, b_high in zip(bins[:-1], bins[1:]):
            m = (df.dijet_mass > b_low) & (df.dijet_mass < b_high)
            dpearson_coef = pearsonr(np.array(df.PNN1[m], dtype=np.float64), np.array(df.PNN2[m], dtype=np.float64))[0]
            dpearson_data_values.append(dpearson_coef)
            print("     %.1f < mjj < %.1f : %.5f"%(b_low, b_high, dpearson_coef))

    nbins = len(bins) - 1
    n_datataking = len(dpearson_data_values) // nbins

    # Reshape dpearson_data_values into a 2D list: each row corresponds to a data-taking period
    dpearson_data_values_reshaped = [
        dpearson_data_values[i * nbins:(i + 1) * nbins] for i in range(n_datataking)
    ]

    # Plot grouped bar plot
    bar_width = 0.2  # Width of each bar
    x = np.arange(nbins)  # x positions for bins

    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors for different data-taking periods
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:n_datataking]
    labels = [f"{processNames[i]}" for i in range(n_datataking)]

    for i, dpearson_values in enumerate(dpearson_data_values_reshaped):
        # Offset the x positions for each data-taking period
        ax.bar(x + i * bar_width, dpearson_values, bar_width, label=labels[i], color=colors[i])
    ax.legend()

    # Customize the plot
    ax.set_xlabel("Mass Bins (mjj ranges)")
    ax.set_ylabel("Pearson R")
    #ax.set_title("Distance Correlation per Mass Bin for Each Data-Taking Period")
    ax.set_xticks(x + bar_width * (n_datataking - 1) / 2)  # Center x-ticks
    ax.set_xticklabels([f"{b_low:.0f}-{b_high:.0f}" for b_low, b_high in zip(bins[:-1], bins[1:])], rotation=45)
    if outFile is not None:
        fig.savefig(outFile, bbox_inches='tight')
    return np.array(dpearson_values)


import pandas as pd
from scipy.interpolate import splev, splrep, interp1d
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

Tc_ = lambda time: 277.36424309532794 - 0.023973977538587 * time - 0.761568634472077 * time**2 + \
    0.025594287611410 * time**3 - 0.000307496975562 * time**4 + 0.000001264230759 * time**5


# Volume MRI + Histo
root = Path(os.path.realpath(__file__)).parent.parent
volumes = pd.read_csv(root / 'data/VolumeArea17.csv')
# what we do is to average some ages
volumes = volumes.append(volumes.loc[volumes.Age.isin([70, 72])].mean(), ignore_index=True)
volumes = volumes.append(volumes.loc[volumes.Age.isin([78, 79])].mean(), ignore_index=True)
volumes = volumes.append(volumes.loc[volumes.Age.isin([84, 86])].mean(), ignore_index=True)
volumes = volumes.loc[~volumes.Age.isin([70, 72, 78, 79, 84, 86])]

col = volumes.loc[:, ["Ratio_Histo_VZ", "Ratio_MRI_VZ"]]
volumes['Ratio_VZ'] = col.mean(axis=1)
col = volumes.loc[:, ["Ratio_Histo_ISVZ", "Ratio_MRI_ISVZ"]]
volumes['Ratio_ISVZ'] = col.mean(axis=1) * volumes.loc[0, "Histo_ISVZ"] / volumes.loc[0, "Histo_VZ"]
col = volumes.loc[:, ["Ratio_Histo_OSVZ", "Ratio_MRI_OSVZ"]]
volumes['Ratio_OSVZ'] = col.mean(axis=1) * volumes.loc[0, "Histo_OSVZ"] / volumes.loc[0, "Histo_VZ"]

volumes.loc[:, ["AverageDensityVZ", "AverageDensityISVZ", "AverageDensityOSVZ"]] /= \
    volumes.loc[0, "AverageDensityVZ"]
volumes["total_cell_number"] = volumes["AverageDensityVZ"] * volumes["Ratio_VZ"] + \
                               volumes["AverageDensityISVZ"] * volumes["Ratio_ISVZ"] + \
                               volumes["AverageDensityOSVZ"] * volumes["Ratio_OSVZ"]



# complicated stuff to sort the list
sorted_number_cells = tuple(zip(*tuple(sorted(list(zip(volumes["Age"],
                                                       volumes["total_cell_number"] /
                                                       volumes.loc[0, "total_cell_number"])),
                                              key=lambda x: x[0]))))
spl = splrep(*sorted_number_cells)
x2 = np.linspace(49, 94, 200)
y2 = splev(x2, spl)
    
    
# From Betizeau et al.

ratio_eomes = pd.DataFrame({"VZ": [16.6, 13.3, 20, 43.3, 58.3],
                            "ISVZ": [88.3, 81.6, 81.6, 83.3, 75.0],
                            "OSVZ": [63.3, 43.3, 56.6, 63.3, 10.0],
                            "ratio_VZ": [43.2, 25, 12.5, 11.3, 10.2],
                            "ratio_ISVZ": [13.6, 12.5, 12.5, 4.5, 6.8],
                            "ratio_OSVZ": [42.0, 61.4, 75.0, 83.0, 81.8],
                           },
                           index=[58, 63, 70, 79, 94]
                          )
ratio_eomes["val"] = (ratio_eomes.VZ * ratio_eomes.ratio_VZ
                   + ratio_eomes.ISVZ * ratio_eomes.ratio_ISVZ
                   + ratio_eomes.OSVZ * ratio_eomes.ratio_OSVZ) \
                  / (ratio_eomes.ratio_VZ + ratio_eomes.ratio_ISVZ + ratio_eomes.ratio_OSVZ)

def plot_number_cells():
    plt.plot(*sorted_number_cells, 'o', x2, y2, label="Ref")
    
def plot_ratio_eomes():
    plt.plot(ratio_eomes.index, ratio_eomes.val / 100, label="Ref")
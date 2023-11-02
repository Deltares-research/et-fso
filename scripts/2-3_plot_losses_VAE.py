import pathlib

import pandas as pd
import matplotlib.pyplot as plt

path = pathlib.Path(__file__).parent.parent / "VAE/training"

data = pd.read_csv(path.joinpath("FSO_VAE-ep200-bs1000-v1.csv"))
print(data)

fig, axes = plt.subplots(nrows=3, figsize=(12, 15))
plt.subplots_adjust(hspace=0.4)

axes[0].plot(data["epoch"], data["loss"], label="training loss")
axes[0].plot(data["epoch"], data["val_loss"], label="validation loss")
axes[0].set_xlim(0, 200)
axes[0].set_ylim(3000, 5000)
axes[0].grid()
axes[0].legend()

axes[1].plot(data["epoch"], data["TF_output_loss"], label="TF_output_loss")
axes[1].plot(data["epoch"], data["val_TF_output_loss"], label="val_TF_output_loss")
axes[1].set_xlim(0, 200)
axes[1].set_ylim(3000, 5000)
axes[1].grid()
axes[1].legend()

axes[2].plot(data["epoch"], data["dist_output_loss"], label="dist_output_loss")
axes[2].plot(data["epoch"], data["val_dist_output_loss"], label="val_dist_output_loss")
axes[2].set_xlim(0, 200)
axes[2].set_ylim(9, 13)
axes[2].grid()
axes[2].legend()

plt.savefig(path / "VAE_v1_losses.png")
plt.show()

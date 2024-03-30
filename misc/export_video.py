from Dataloader.eagle import EagleDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.tri import Triangulation

import matplotlib.animation as animation

d = EagleDataset(data_path="/Volumes/Samsung_T5/Eagle_dataset/", window_length=400, mode='test', with_cells=True)
dataloader = DataLoader(d, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

for i in tqdm(range(10)):
    x = d[i]

    m = x['mesh_pos']
    v = (x['velocity'] ** 2).sum(-1)

    fig, ax = plt.subplots(figsize=(10, 5))
    tri = Triangulation(m[0, :, 0], m[0, :, 1], x['cells'].numpy()[0])
    vmin = v.min()
    vmax = v.max()
    r = ax.tripcolor(tri, v[0], cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    ax.set_aspect('equal')
    fig.suptitle("Timestep 0")

    def animate(i):
        maskedTris = tri.get_masked_triangles()
        r.set_array(v[i][maskedTris].mean(axis=1))
        fig.suptitle(f"Timestep {i}")
        return [r]

    plt.tight_layout()
    fig.subplots_adjust(right=1, left=0)
    anim = animation.FuncAnimation(fig, animate, frames=len(v) - 1, interval=1)
    writer = animation.writers["ffmpeg"](fps=30)
    anim.save(f'{i}.mp4', writer=writer)
    # plt.show()

import matplotlib.pyplot as plt
from publishutil import FigureLayout

fig_layout = FigureLayout("nature")
fig = plt.figure(figsize=fig_layout.get_figsize(n_columns=2, height=1.0), constrained_layout=True)
gs = fig.add_gridspec(3, 3)
ax1 = fig.add_subplot(gs[0, :])
ax1.set_title("gs[0, :]")
ax2 = fig.add_subplot(gs[1, :-1])
ax2.set_title("gs[1, :-1]")
ax3 = fig.add_subplot(gs[1:, -1])
ax3.set_title("gs[1:, -1]")
ax4 = fig.add_subplot(gs[-1, 0])
ax4.set_title("gs[-1, 0]")
ax5 = fig.add_subplot(gs[-1, -2])
ax5.set_title("gs[-1, -2]")

# Add the panel_label attributes
ax1.panel_label = "A"
ax2.panel_label = "B"
ax3.panel_label = "E"
ax4.panel_label = "C"
ax5.panel_label = "D"

# Draw the panel labels
fig_layout.draw_panel_labels(fig)

fig.savefig("test.pdf", bbox_inches="tight")
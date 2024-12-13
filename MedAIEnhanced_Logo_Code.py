'''
MedAI_Cross_Enhanced_Logo_Code.py

This file is a utility file that create the moving logo picture for the streamlit application and the and final reports as well
as the presentation later.
With the help of the matplotlib capabilities allows for a logo that gives some creatvity to the MedAI system.
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.animation import FuncAnimation, PillowWriter

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
ax.set_facecolor('#000000')  # Black background for fill effect
#ax.axis('off')

# DNA Helix base parameters
x = np.linspace(0, 1, 500)

# Heartbeat
heartbeat_base = np.array([[0.1, 0.5], [0.15, 0.6], [0.2, 0.4], [0.3, 0.7], [0.4, 0.5]])
ax.plot(heartbeat_base[:, 0], heartbeat_base[:, 1], color='#1E90FF', lw=2)

# Nodes and connections for neural network
node_positions = [(0.3, 0.6), (0.4, 0.8), (0.6, 0.7), (0.7, 0.5), (0.5, 0.3)]
for pos in node_positions:
    ax.plot(pos[0], pos[1], 'o', color='#FFD700', markersize=8)  # Golden nodes
    ax.plot([0.5, pos[0]], [0.5, pos[1]], color='#FFD700', lw=1)

# Add circular glowing blue background
outer_glow = Circle((0.5, 0.5), 0.42, color='#1E90FF', alpha=0.3)
ax.add_artist(outer_glow)
inner_circle = Circle((0.5, 0.5), 0.4, color='#1E90FF', alpha=0.8)
ax.add_artist(inner_circle)

# Add red cross with a thick black outline
cross_width = 0.08
cross_outline_width = 0.1
# Vertical part of the cross
ax.add_patch(Rectangle((0.46, 0.35), cross_width, 0.3, color="red", zorder=2))
ax.add_patch(Rectangle((0.46, 0.35), cross_width, 0.3, edgecolor="black", linewidth=cross_outline_width, facecolor="none", zorder=3))
# Horizontal part of the cross
ax.add_patch(Rectangle((0.35, 0.46), 0.3, cross_width, color="red", zorder=2))
ax.add_patch(Rectangle((0.35, 0.46), 0.3, cross_width, edgecolor="black", linewidth=cross_outline_width, facecolor="none", zorder=3))

# Add text labels
plt.text(0.5, 0.875, "The MedAI Diagnosis System: The Neural Way to Transform Healthcare", fontsize=16, ha='center', color='#000000', weight='bold')

# Brain outline and patterns
theta = np.linspace(0, 2 * np.pi, 200)
brain_x = 0.5 + 0.2 * np.cos(theta)
brain_y = 0.5 + 0.15 * np.sin(theta)
ax.plot(brain_x, brain_y, color='#FF4500', lw=1.5, alpha=0.8)  # Adjusted to complement red cross

brain_nodes = [(0.5, 0.55), (0.45, 0.6), (0.55, 0.65), (0.6, 0.6), (0.4, 0.5), (0.5, 0.45)]
for b_node in brain_nodes:
    ax.plot(b_node[0], b_node[1], 'o', color='#4CAF50', markersize=6)
    for target in brain_nodes:
        ax.plot([b_node[0], target[0]], [b_node[1], target[1]], color='#4CAF50', lw=0.5, alpha=0.6)

circuit_lines = [((0.5, 0.55), (0.7, 0.55)), ((0.5, 0.45), (0.3, 0.45))]
for start, end in circuit_lines:
    ax.plot([start[0], end[0]], [start[1], end[1]], color='#1E90FF', lw=1, linestyle='--')

# DNA Helix animated lines
line1, = ax.plot(x, 0.25 * np.sin(6 * np.pi * x) + 0.5, color='#1E90FF', lw=2)
line2, = ax.plot(x, 0.25 * np.cos(6 * np.pi * x) + 0.5, color='#87CEFA', lw=2)

# Animation function
def update(frame):
    # Animate DNA helix by shifting sine and cosine waves
    y1 = 0.25 * np.sin(6 * np.pi * x + frame * 0.1) + 0.5
    y2 = 0.25 * np.cos(6 * np.pi * x + frame * 0.1) + 0.5
    line1.set_ydata(y1)
    line2.set_ydata(y2)
    return line1, line2

# Animate
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Save animation as GIF
animated_output_path = "MedAI_Logo.gif"
ani.save(animated_output_path, writer=PillowWriter(fps=20))
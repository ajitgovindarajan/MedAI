'''
Logo.py

This file is a utility file that create the static logo picture for the streamlit application and the and final reports as well
as the presentation later.
With the help of the matplotlib capabilities allows for a logo that gives some creatvity to the MedAI system.
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
ax.set_facecolor('#f0f0f0')

# DNA Helix parameters
x = np.linspace(0, 1, 500)
y1_base = 0.25 * np.sin(6 * np.pi * x) + 0.5
y2_base = 0.25 * np.cos(6 * np.pi * x) + 0.5

# Plot DNA helix
line1, = ax.plot(x, y1_base, color='#1976D2', lw=2)
line2, = ax.plot(x, y2_base, color='#64B5F6', lw=2)

# Heartbeat
heartbeat_base = np.array([[0.1, 0.5], [0.15, 0.6], [0.2, 0.4], [0.3, 0.7], [0.4, 0.5]])
heartbeat_line, = ax.plot(heartbeat_base[:, 0], heartbeat_base[:, 1], color='#1976D2', lw=2)

# Nodes and connections for neural network
node_positions = [(0.3, 0.6), (0.4, 0.8), (0.6, 0.7), (0.7, 0.5), (0.5, 0.3)]
nodes = []
connections = []
for pos in node_positions:
    node, = ax.plot(pos[0], pos[1], 'o', color='#FFC300', markersize=8)
    connection, = ax.plot([0.5, pos[0]], [0.5, pos[1]], color='#FFC300', lw=1)
    nodes.append(node)
    connections.append(connection)

# "+" symbol in the center
plt.text(0.5, 0.5, "+", fontsize=120, ha='center', va='center', color='white', weight='bold')

# Background circle
circle = plt.Circle((0.5, 0.5), 0.4, color='#2196F3', fill=True, alpha=0.6)
ax.add_artist(circle)

# Brain outline and inner network
theta = np.linspace(0, 2 * np.pi, 200)
brain_x = 0.5 + 0.2 * np.cos(theta)
brain_y = 0.5 + 0.15 * np.sin(theta)
ax.plot(brain_x, brain_y, color='#FF5733', lw=1.5, alpha=0.8)

brain_nodes = [(0.5, 0.55), (0.45, 0.6), (0.55, 0.65), (0.6, 0.6), (0.4, 0.5), (0.5, 0.45)]
for b_node in brain_nodes:
    ax.plot(b_node[0], b_node[1], 'o', color='#4CAF50', markersize=6)
    for target in brain_nodes:
        ax.plot([b_node[0], target[0]], [b_node[1], target[1]], color='#4CAF50', lw=0.5, alpha=0.6)

# Circuit-like patterns
circuit_lines = [((0.5, 0.55), (0.7, 0.55)), ((0.5, 0.45), (0.3, 0.45))]
for start, end in circuit_lines:
    ax.plot([start[0], end[0]], [start[1], end[1]], color='#2196F3', lw=1, linestyle='--')

# Add text labels
plt.text(0.5, 0.88, "MedAI", fontsize=24, ha='center', color='#333333', weight='bold')
plt.text(0.5, 0.12, "Diagnosis System", fontsize=16, ha='center', color='#555555', alpha=0.8)

# Remove axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Save static logo
static_output_path = "MedAI_Enhanced_Logo_Static.png"
plt.savefig(static_output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Static logo saved as {static_output_path}")

# Animation function
def update(frame):
    # Update DNA helix animation
    y1 = 0.25 * np.sin(6 * np.pi * x + frame * 0.1) + 0.5
    y2 = 0.25 * np.cos(6 * np.pi * x + frame * 0.1) + 0.5
    line1.set_ydata(y1)
    line2.set_ydata(y2)

    # Update heartbeat animation (pulse effect)
    scale = 1 + 0.1 * np.sin(frame * 0.1)
    heartbeat_line.set_linewidth(scale * 2)

    # Update nodes animation (glow effect)
    for node in nodes:
        node.set_markersize(8 + 2 * np.sin(frame * 0.1))

    return line1, line2, heartbeat_line, *nodes

# Animate
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Save animation as a GIF
animated_output_path = "MedAI_Creative_AI_Neural_Network_Logo.gif"
ani.save(animated_output_path, writer=PillowWriter(fps=20))

print(f"Animation saved as {animated_output_path}")
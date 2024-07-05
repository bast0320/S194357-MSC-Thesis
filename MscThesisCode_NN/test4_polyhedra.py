import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_constraints(ax, A, b):
    x = np.linspace(-5, max(b) * 1.5, 100)
    for I in range(len(b)):
        if A[I][0] != 0:
            y = (b[I] - A[I][0] * x) / A[I][1]
            ax.plot(x, y, label=f'Constraint {I+1}')
    ax.set_xlim(-5, max(b) * 1.5)
    ax.set_ylim(-5, max(b) * 1.5)
    ax.legend()

def simplex_step(A, b, c, x, basic_vars):
    m, n = A.shape
    c_B = c[basic_vars]
    B = A[:, basic_vars]
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        print("Error: B is singular and cannot be inverted.")
        return None, None, None
    
    c_red = c - c_B.dot(B_inv.dot(A))
    entering = np.argmax(c_red)
    if c_red[entering] <= 1e-10:
        return None, None, None
    
    d = np.zeros(n)
    d[entering] = 1
    d[basic_vars] = -B_inv.dot(A[:, entering])
    
    ratios = [(x[bv] / -d[bv], i) for i, bv in enumerate(basic_vars) if d[bv] < 0]
    if not ratios:
        return None, None, None
    
    step, leaving_index = min(ratios)
    leaving = basic_vars[leaving_index]
    x_new = x + step * d
    new_basic_vars = basic_vars.copy()
    new_basic_vars[leaving_index] = entering
    
    return x_new, new_basic_vars, (entering, leaving)

# Define a more challenging 2D problem
A = np.array([
    [2, 1],
    [1, 3],
    [1, -1],
    [-1, 2]
])
b = np.array([8, 12, 3, 6])
c = np.array([3, 4])  # Objective function coefficients to maximize

# Add slack variables to make the initial basic feasible solution obvious
A_slack = np.hstack([A, np.eye(A.shape[0])])
c_slack = np.hstack([c, np.zeros(A.shape[0])])

# Custom starting point (modify these values as needed)
x1_start = 0
x2_start = -2

# Calculate slack variables for the starting point
slack_vars = b - A.dot(np.array([x1_start, x2_start]))
x_initial = np.hstack([[x1_start, x2_start], slack_vars])

# Determine initial basic variables
basic_vars = []
for i in range(A.shape[1], len(x_initial)):
    if x_initial[i] > 1e-10:
        basic_vars.append(i)
while len(basic_vars) < A.shape[0]:
    for i in range(A.shape[1]):
        if i not in basic_vars:
            basic_vars.append(i)
            break

# Perform simplex algorithm steps
steps = [x_initial[:2]]  # Only store first two variables for 2D visualization
x = x_initial
while True:
    x_new, new_basic_vars, pivot = simplex_step(A_slack, b, c_slack, x, basic_vars)
    if x_new is None:
        break
    x = x_new
    basic_vars = new_basic_vars
    steps.append(x[:2])  # Only store first two variables for 2D visualization

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Create a contour plot of the objective function
x1 = np.linspace(-5,  max(b) * 1.5, 400)
x2 = np.linspace(-5,  max(b) * 1.5, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = c[0] * X1 + c[1] * X2
contour = ax.contourf(X1, X2, Z, levels=50, cmap='gray', alpha=0.5)
# Add a colorbar
cbar = fig.colorbar(contour, ax=ax)

plot_constraints(ax, A, b)
ax.set_title('2D Simplex Algorithm Animation')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# Initialize the point, text, and arrows
point, = ax.plot([], [], 'ro', markersize=10)
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')
arrows = []

# Animation function
def animate(i):
    x = steps[i]
    point.set_data([x[0]], [x[1]])
    text.set_text(f'Step {i}\nx1 = {x[0]:.2f}, x2 = {x[1]:.2f}\nObjective = {c.dot(x):.2f}')
    
    if i > 0:
        prev_x = steps[i-1]
        arrow = ax.arrow(prev_x[0], prev_x[1], x[0]-prev_x[0], x[1]-prev_x[1],
                         color='black', head_width=0.2, head_length=0.3, linewidth = 2.5, length_includes_head=True)
        arrows.append(arrow)
    
    return [point, text] + arrows

# Create the animation
anim = FuncAnimation(fig, animate, frames=len(steps), interval=1000, blit=True, repeat=False)
# save as gif
anim.save('simplex_animation.gif', writer='imagemagick')
plt.show()
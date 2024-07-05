import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

def plot_polyhedron_with_steps(A, b, steps):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate points within the feasible region
    num_points = 100000
    points = np.random.rand(num_points, A.shape[1])
    
    # Scale points to satisfy constraints
    scale_factors = b / np.max(A.dot(points.T), axis=1)
    min_scale = np.min(scale_factors)
    points *= min_scale
    
    # Keep only the points that satisfy all constraints
    mask = np.all(A.dot(points.T) <= b[:, np.newaxis], axis=0)
    feasible_points = points[mask]
    
    if len(feasible_points) < 4:
        raise ValueError("Not enough feasible points to create a 3D polyhedron. Check your constraints.")
    
    # Project 4D points to 3D for visualization (using first 3 dimensions)
    feasible_points_3d = feasible_points[:, :3]
    
    hull = ConvexHull(feasible_points_3d)
    
    # Plot the polyhedron
    for simplex in hull.simplices:
        faces = Poly3DCollection([feasible_points_3d[simplex]])
        faces.set_alpha(0.2)
        faces.set_color('g')
        ax.add_collection3d(faces)
    
    # Set axis limits based on feasible points
    ax.set_xlim(0, np.max(feasible_points_3d[:, 0]))
    ax.set_ylim(0, np.max(feasible_points_3d[:, 1]))
    ax.set_zlim(0, np.max(feasible_points_3d[:, 2]))
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('Constraint Polyhedron with Simplex Steps')
    
    # Plot each step
    for i, x in enumerate(steps):
        x_3d = x[:3]  # Project to 3D
        ax.scatter(x_3d[0], x_3d[1], x_3d[2], color='red', s=100)
        ax.text(x_3d[0], x_3d[1], x_3d[2], f'Step {i}', fontsize=8)
    
    plt.show()

def simplex_step(A, b, c, x, basic_vars):
    m, n = A.shape
    c_B = c[basic_vars]
    B = A[:, basic_vars]
    
    # Use pseudoinverse instead of inverse to handle non-square matrices
    B_inv = np.linalg.pinv(B)
    
    # Calculate reduced costs
    c_red = c - c_B.dot(B_inv.dot(A))
    
    # Find entering variable
    entering = np.argmax(c_red)
    if c_red[entering] <= 1e-10:  # Consider values close to zero as optimal
        return None, None, None  # Optimal solution reached
    
    # Calculate direction
    d = np.zeros(n)
    d[entering] = 1
    d[basic_vars] = -B_inv.dot(A[:, entering])
    
    # Find leaving variable
    ratios = []
    for i, bv in enumerate(basic_vars):
        if d[bv] < 0:
            ratio = -x[bv] / d[bv]
            ratios.append((ratio, i))
    
    if not ratios:
        return None, None, None  # Unbounded problem
    
    # Determine step size and leaving variable
    step, leaving_index = min(ratios)
    leaving = basic_vars[leaving_index]
    
    # Update solution
    x_new = x + step * d
    
    # Update basic variables
    new_basic_vars = basic_vars.copy()
    new_basic_vars[leaving_index] = entering
    
    return x_new, new_basic_vars, (entering, leaving)

# Define the problem
A = np.array([
    [3, 1, 1, 4],
    [1, -3, 2, 3],
    [2, 1, 3, -1]
])
b = np.array([12, 6, 10])
c = np.array([2, 4, 3, 1])  # Objective function coefficients to maximize

# Find an initial basic feasible solution
res = linprog(-c, A_ub=A, b_ub=b, method='highs')
x = res.x
basic_vars = [i for i in range(len(x)) if x[i] > 1e-10]

print("Initial basic feasible solution:", x)
print("Initial basic variables:", [i+1 for i in basic_vars])

# Perform simplex algorithm steps
steps = [x]  # Start with the initial solution
for step in range(10):  # Allow for more steps if needed
    print(f"\nStep {step + 1}:")
    print(f"Current solution: {x}")
    print(f"Current objective value: {c.dot(x)}")
    
    x_new, new_basic_vars, pivot = simplex_step(A, b, c, x, basic_vars)
    
    if x_new is None:
        print("Optimal solution reached or problem is unbounded.")
        break
    
    x = x_new
    basic_vars = new_basic_vars
    steps.append(x)  # Add the new solution to our steps
    print(f"Entering variable: x{pivot[0] + 1}")
    print(f"Leaving variable: x{pivot[1] + 1}")

print("\nFinal solution:", x)
print("Final objective value:", c.dot(x))

# Plot the polyhedron with all steps
plot_polyhedron_with_steps(A, b, steps)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def shape_functions(xi, eta):
    N0 = 0.25 * (1 - xi) * (1 - eta)
    N1 = 0.25 * (1 + xi) * (1 - eta)
    N2 = 0.25 * (1 + xi) * (1 + eta)
    N3 = 0.25 * (1 - xi) * (1 + eta)
    return [N0, N1, N2, N3]

def plot_shape_functions_contours():
    xi = np.linspace(-1, 1, 50)
    eta = np.linspace(-1, 1, 50)
    XI, ETA = np.meshgrid(xi, eta)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    shape_titles = ['N0', 'N1', 'N2', 'N3']

    for i in range(4):
        Ni = np.vectorize(lambda xi, eta: shape_functions(xi, eta)[i])(XI, ETA)
        ax = axes[i // 2, i % 2]
        cp = ax.contourf(XI, ETA, Ni, levels=20)
        fig.colorbar(cp, ax=ax)
        ax.set_title(f'Shape Function {shape_titles[i]}')
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')
        ax.set_aspect('equal')

    st.pyplot(fig)

def plot_shape_functions_3d():
    fig = plt.figure(figsize=(16, 10))
    titles = ['N0', 'N1', 'N2', 'N3']
    xi = np.linspace(-1, 1, 20)
    eta = np.linspace(-1, 1, 20)
    XI, ETA = np.meshgrid(xi, eta)
    shape_values = shape_functions(XI, ETA)

    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.plot_surface(XI, ETA, shape_values[i], cmap='viridis')
        ax.set_title(f'Shape Function {titles[i]}')
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')
        ax.set_zlabel(f'$N_{i}(\\xi,\\eta)$')

    st.pyplot(fig)

# === Streamlit layout ===
st.title("Shape Functions Visualization (QUAD4)")
st.write("This app visualizes the bilinear shape functions used in 2D quadrilateral elements.")

st.header("Contour plots")
plot_shape_functions_contours()

st.header("3D surface plots")
plot_shape_functions_3d()

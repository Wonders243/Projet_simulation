from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def tracer_territoires(systeme):
    points = [pos['position'] for pos in systeme.noeuds.values()]
    labels = list(systeme.noeuds.keys())
    vor = Voronoi(points)

    # Affichage
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange')

    # Affichage des noms d’espèces
    for i, (x, y) in enumerate(points):
        ax.text(x, y, labels[i], color='black', ha='center', va='center', fontsize=8)

    plt.axis('equal')
    plt.title("Territoires par espèces (Voronoï)")
    plt.show()
import numpy as np
from scipy.spatial import KDTree
import pyvista as pv
from bandu.translate import TranslatePoints

# read positions from LAMMPS file
#positions_path = rf'\\research.drive.wisc.edu\dcfredrickso\Patrick\SSMC\Positions.dump'
positions_path = '../examples/color_atoms_kick_plane.txt'
#positions_path = '../examples/First_positions.txt'
print('Reading positions file')
with open(positions_path) as f:
    lines = f.readlines()
print("Positions read")

# get info from header of positions file
num_atoms = int(lines[3].strip())
cell_len = float(lines[5].strip().split(' ')[-1]) - float(lines[5].strip().split(' ')[0])
shift = float(lines[5].strip().split(' ')[0])
head_len = 9

# get positions and convert from string to float and make point cloud
print('Constructing point cloud from positions data')
first_pos = lines[head_len:head_len + num_atoms]
positions =  np.zeros((num_atoms,3))
rgb_vals = np.zeros((num_atoms,3))
for i, pos in enumerate(first_pos):
    pos = pos.strip().split(' ')
    pos = [ch for ch in pos if ch != '']
    positions[i:] = [float(p) for p in pos[2:5]]
    rgb_vals[i:] = [float(c) for c in pos[5:]]
opacities1 = [1 if np.sum(num) > 2.5 else 0 for num in rgb_vals]
opacities2 = [1 if np.sum(num) < 2.5 else 0 for num in rgb_vals]
scale = False
if scale:
    positions *= cell_len
positions -= shift
positions = pv.PolyData(positions)
print('Point cloud made')

# grid to sample rdf on
sampling = 15
grid = pv.ImageData(
    dimensions=(sampling,sampling,sampling),
    origin=(0.5*cell_len/sampling,0.5*cell_len/sampling,0.5*cell_len/sampling),
    spacing=(cell_len/sampling,cell_len/sampling,cell_len/sampling)
)

# nearest neighbor search
print('Neighbor Search')
scalars = np.zeros((sampling**3,1))
tree = KDTree(
    data = positions.points,
    leafsize = 10
)
for i, search_pt in enumerate(grid.points):
    nbr_inds = tree.query_ball_point(
        x = search_pt,
        r = 10
    )
    nearest_nbrs = np.take(positions.points, nbr_inds, axis=0)
    nbr_dists = np.linalg.norm(nearest_nbrs-search_pt, axis=1)
    hist, bin_edges = np.histogram(
        nbr_dists,
        bins = 200,
        range = (0,10)
    )
    scalars[i] = np.max(hist)

# interpolate grid
trans_points, trans_scalars = TranslatePoints(
    points = grid.points,
    values = scalars,
    lattice_vecs = cell_len*np.identity(3)
)
trans_grid = pv.PolyData(trans_points)
trans_grid['values'] = trans_scalars
magnitude = 0.01
grid = grid.interpolate(
    trans_grid,
    radius = magnitude*cell_len/sampling
)
contours = grid.contour(
    isosurfaces = 10,
    method = 'contour',
    rng = [7,13]
)

# cubic partition
sub_cubes = pv.MultiBlock()
for pt in grid.points:
    sub_cubes.append(pv.Cube(
        center=pt,
        x_length=cell_len/sampling,
        y_length=cell_len/sampling,
        z_length=cell_len/sampling
    ))

# plot
print('Plotting')
p = pv.Plotter()

cell = pv.Cube(
    center=(cell_len/2,cell_len/2,cell_len/2),
    x_length=cell_len,
    y_length=cell_len,
    z_length=cell_len
)
p.add_mesh(
    cell,
    style='wireframe',
    color='black'
)
p.add_mesh(
    contours,
    opacity=0.5
)
p.add_mesh(
    positions,
    style='points',
    point_size=10,
    scalars=rgb_vals,
    opacity=opacities1,
    render_points_as_spheres=True
)
'''
p.add_volume(
    volume=grid, #type: ignore
)
p.add_mesh(
    sub_cubes,
    style='wireframe',
    color='blue'
)
p.add_mesh(
    pv.PolyData(grid.points),
    color='green'
)
'''
p.show()
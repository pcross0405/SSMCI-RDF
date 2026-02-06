import numpy as np
import pyvista as pv

# read positions from LAMMPS file
#positions_path = rf'\\research.drive.wisc.edu\dcfredrickso\Patrick\SSMC\Positions.dump'
positions_path = 'color_atoms_kick_plane.txt'
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
opacities = [1 if np.sum(num) > 2.5 else 0 for num in rgb_vals]
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

# compute rdf at each grid point
radius = cell_len/sampling
dr = 0.5
grid_scalars = []
scalars = 0
if scalars:
    print('Computing RDF')
    for pt in grid.points:
        max_count = 0
        start = 0.01
        while dr <= radius:
            s1 = pv.Sphere(
                center=pt,
                radius=start
            )
            s2 = pv.Sphere(
                center=pt,
                radius=dr
            )
            p1 = positions.select_enclosed_points(s1)
            p1_mask = p1['SelectedPoints'].view(bool)
            p1 = positions.extract_points(p1_mask, adjacent_cells=False)
            p2 = positions.select_enclosed_points(s2)
            p2_mask = p2['SelectedPoints'].view(bool)
            p2 = positions.extract_points(p2_mask, adjacent_cells=False)
            count = len([p for p in p2.points if p not in p1.points])
            if count > max_count:
                max_count = count
            dr += 0.5
            start += 0.5
        grid_scalars.append(max_count)
    print('RDF complete')
    print(grid_scalars)

# liquid interface of constant density
vals = np.zeros((len(grid.points),1))
eta = 0.95
print('Compute Gaussians')
for i, pt in enumerate(grid.points):
    dist = np.linalg.norm(pt-positions.points, axis=1)
    gaussian = (2*np.pi*eta**2)**(-3/2) * np.exp(-dist**2/(2*eta**2))
    vals[i] = np.sum(gaussian)

# interpolation
resolution = sampling
print('Interpolating')
hi_res_grid = pv.ImageData(
    dimensions=3*[resolution],
    origin=3*[0.5*cell_len/resolution],
    spacing=3*[cell_len/resolution]
)
grid['values'] = vals
interp_grid = hi_res_grid.interpolate(
    grid,
    radius=eta
)
contours = interp_grid.contour(
    isosurfaces = 5,
    method = 'contour'
)
contours = contours.smooth_taubin(n_iter=100, pass_band=0.05)

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
    positions,
    style='points',
    point_size=10,
    scalars=rgb_vals,
    opacity=1,
    render_points_as_spheres=True
)
'''
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
p.add_mesh(
    contours,
    color='red',
    opacity=1,
    smooth_shading=True
)

p.show()
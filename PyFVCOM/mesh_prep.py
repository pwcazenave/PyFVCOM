import numpy as np
import scipy.interpolate as si
import shapely.geometry as sg

def read_smesh_polygons(boundary_file='boundary_poly.txt', islands_file='island_polys.txt'):
    boundary_polygon = []
    with open(boundary_file, 'r') as f:
        for this_line in f:
            this_line_clean = this_line.rstrip('\n')
            parsed_line = this_line_clean.split(' ')
            if len(parsed_line) > 1:
                boundary_polygon.append(parsed_line)
    boundary_polygon = np.asarray(boundary_polygon, dtype=float)

    islands_dict = {}
    islands_counter = -1
    this_islands_poly = []
    with open(islands_file, 'r') as f:
        for this_line in f:
            this_line_clean = this_line.rstrip('\n')
            parsed_line = this_line_clean.split(' ')
    
            if len(parsed_line) == 1:
                islands_dict[islands_counter] = np.asarray(this_islands_poly, dtype=float)
                islands_counter += 1
                this_islands_poly = []
            else:
                this_islands_poly.append(parsed_line)

    islands_dict[islands_counter] = np.asarray(this_islands_poly, dtype=float)
    islands_dict.pop(-1)

    return boundary_polygon, islands_dict

def write_smesh_polygons(boundary_polygon, islands_dict, boundary_file='boundary_poly.txt', islands_file='island_polys.txt'):
    with open(boundary_file, 'w') as f:
        f.write('{:d}\n'.format(len(boundary_polygon)))
        for this_node in boundary_polygon:
            f.write('{:.6f} {:.6f}\n'.format(*this_node))

    with open(islands_file, 'w') as f:
        for this_island in islands_dict.values():
            f.write('{:d}\n'.format(len(this_island)))
            for this_node in this_island:
                f.write('{:.6f} {:.6f}\n'.format(*this_node))

def read_cst_file(infile='coast.cst'):
    # Assumes coastline is first polygon 
    islands_dict = {}
    boundary_counter = 0
    island_counter = -1
    this_nodestr = []

    with open(infile, 'r') as f:
        line_list = f.readlines()

    for file_line in line_list[3:]:
        file_line_fmt = np.asarray(file_line.split(' '), dtype=float)
        
        if file_line_fmt[1] in [0,1]:
            this_nodestr = np.asarray(this_nodestr)

            if boundary_counter == 0:
                boundary_coast_points = this_nodestr
            else:
                islands_dict[island_counter] = this_nodestr

            boundary_counter += 1
            island_counter += 1
            this_nodestr = []

        else:
            this_nodestr.append(file_line_fmt) 

    return boundary_coast_points, islands_dict


def write_cst_file(boundary_coast_points, islands_dict, outfile='coast.cst'):
    with open(outfile, 'w') as f:
        f.write('COAST\n')
        f.write('{:d}\n'.format(len(islands_dict.values())+1))
        f.write('{:d} 0\n'.format(len(boundary_coast_points)))
        for this_node in boundary_coast_points:
            f.write('{:.6f} {:.6f}\n'.format(*this_node))
        for this_island in islands_dict.values():
            f.write('{:d} 1\n'.format(len(this_island)))
            for this_node in this_island:
                f.write('{:.6f} {:.6f}\n'.format(*this_node))

def _hi_res_line(pt_1, pt_2, res):
    line_pts = int(np.floor(np.sqrt((pt_2[0] - pt_1[0])**2 + (pt_2[1] - pt_1[1])**2)/res) + 2)
    return np.asarray([np.linspace(pt_1[0], pt_2[0], line_pts), np.linspace(pt_1[1], pt_2[1], line_pts)]).T

def hi_res_polygon(poly_pts, res):
    new_poly_points = _hi_res_line(poly_pts[0], poly_pts[1],res)
    for i in np.arange(1,len(poly_pts)-1):
         new_poly_points = np.vstack([new_poly_points, _hi_res_line(poly_pts[i], poly_pts[i+1],res)])

    return new_poly_points

def poly_normals(poly_pts):
    """


    """
    clockwise = check_poly_clockwise(poly_pts)
    if not clockwise:
        poly_pts = np.flipud(poly_pts)

    unit_normal_vecs = []

    this_normal_vec = [-(poly_pts[1,1] - poly_pts[-1,1]), (poly_pts[1,0] - poly_pts[-1,0])]
    unit_normal_vecs.append(this_normal_vec/np.linalg.norm(this_normal_vec))

    for i in np.arange(1, len(poly_pts) -1):
        this_normal_vec = [-(poly_pts[i+1,1] - poly_pts[i-1,1]), (poly_pts[i+1,0] - poly_pts[i-1,0])]
        unit_normal_vecs.append(this_normal_vec/np.linalg.norm(this_normal_vec))

    this_normal_vec = [-(poly_pts[0,1] - poly_pts[-2,1]), (poly_pts[0,0] - poly_pts[-2,0])]
    unit_normal_vecs.append(this_normal_vec/np.linalg.norm(this_normal_vec))

    if not clockwise:
        unit_normal_vecs = np.flipud(unit_normal_vecs)

    return unit_nromal_vecs

def check_poly_clockwise(poly_pts):
    shoelace_sum = (poly_pts[0,0] - poly_pts[-1,0])*(poly_pts[0,1] + poly_pts[-1,1])
    for i in np.arange(0, len(poly_pts)-1):
        shoelace_sum += (poly_pts[i+1,0] - poly_pts[i,0])*(poly_pts[i+1,1] + poly_pts[i,1])   

    if shoelace_sum > 0:
        return True
    else:
        return False




def close_channels(modified_boundary_poly, islands_dict, resolution, remove_small_islands_first=True, remove_small_islands_last=False):
    """


    """

    ## Simplify the islands first so there are (hopefully!) less things to intersect with the boundary
    
    # Convert into polygons, remove any with area < resolution^2 as they are too tiny to consider, and buffer    
    poly_islands = {}
    islands_to_process = []
    counter = 0
    for this_island in islands_dict.values():
        this_poly = sg.Polygon(this_island)        
        if not remove_small_islands_first:
            poly_islands[counter] = this_poly.buffer(resolution)
            islands_to_process.append(counter)
            counter += 1
        elif this_poly.area > resolution**2:
            poly_islands[counter] = this_poly.buffer(resolution)
            islands_to_process.append(counter)
            counter += 1

    bdry_poly = sg.Polygon(modified_boundary_poly)    
    poly_islands[counter] = bdry_poly.buffer(resolution)
    islands_to_process.append(counter)
    counter += 1

    # Loop over polygons, if they intersect then merge (and remove holes) and put at the bottom of the list

    while len(islands_to_process) > 0:
        this_process = islands_to_process.pop(0)
        this_island = poly_islands[this_process]

        for this_comp_island in islands_to_process:
            if this_island.intersects(poly_islands[this_comp_island]):
                new_island = this_island.union(poly_islands[this_comp_island])
                
                # Check for holes

                               
                islands_to_process.remove(this_comp_island)
                poly_islands.pop(this_process)
                poly_islands.pop(this_comp_island)              

                islands_to_process.append(counter)
                poly_islands[counter] = new_island
                counter += 1                
                break
    
        print('Island {} of {} processed'.format(this_process, counter))

    new_islands = {}
    counter = 0
    for this_island in poly_islands.values():
        reduce_island = this_island.buffer(-resolution)
        try:
            for this_red_island in reduce_island:
                new_islands[counter] = this_red_island
                counter += 1 
        except TypeError:
            new_islands[counter] = reduce_island
            counter +=1

    # we assume the land boundary is the largest area polygon
    bdry_poly_key = list(new_islands)[0]
    for this_key, this_poly in new_islands.items():
        if this_poly.area > new_islands[bdry_poly_key].area:
            bdry_poly_key = this_key

    new_bdry = new_islands.pop(bdry_poly_key)
    
    new_islands_xy = {}
    for this_key, this_poly in new_islands.items():
        new_islands_xy[this_key] = np.asarray(this_poly.exterior.xy).T

    new_bdry_xy = np.asarray(new_bdry.exterior.xy).T

    return new_bdry_xy, new_islands_xy

def smooth_by_interpolate(fvcom_filereader, cell_or_node='node'):
    if cell_or_node == 'node':
        interp_from = [fvcom_filereader.grid.x, fvcom_filereader.grid.y, fvcom_filereader.grid.h]
        interp_intermediate = [fvcom_filereader.grid.xc, fvcom_filereader.grid.yc,
                                            fvcom_filereader.grid.h_center]

    elif cell_or_node == 'cell':
        interp_intermediate = [fvcom_filereader.grid.x, fvcom_filereader.grid.y, fvcom_filereader.grid.h]
        interp_from = [fvcom_filereader.grid.xc, fvcom_filereader.grid.yc,
                                            fvcom_filereader.grid.h_center]

    first_interpolater = si.LinearNDInterpolator((interp_from[0], interp_from[1]),interp_from[2])
    first_smooth = first_interpolater((interp_intermediate[0], interp_intermediate[1]))

    second_interpolater = si.LinearNDInterpolator((interp_intermediate[0], interp_intermediate[1]),
                                    first_smooth)
    new_output = second_interpolater((interp_from[0], interp_from[1]))
    new_nan = np.isnan(new_output)
    new_output[new_nan] = interp_from[2][new_nan]

    return new_output

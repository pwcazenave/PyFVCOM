import numpy as np
import copy



"""

There are two commonly used methods for calculating the gradient on an unstructured grid, the Green-Gauss and Least Squares
These implementations are taken from 'The Finite Volume Method in Computational Fluid Dynamics' Moukalled, Mangani, and Darwish. Springer 2016


See also Sryakos et al 2017. for a (slightly alarming) discussion of the order of the associated errors

"""

def grad_ls_method(field, stencils, r_cfs, w_ks):
    """

    The stencils, r_cfs, w_ks are dictionaries calculated by grad_ls_stencil_weight_calc()

    """    
    grad_x = np.zeros(field.shape)
    grad_y = np.zeros(field.shape)
    
    for this_step_ind, this_step in enumerate(field):
        for this_node_ind, this_node_theta in enumerate(this_step):
            this_stencil = stencils[this_node_ind]
            this_stencil_delta_theta = this_step[this_stencil] - this_node_theta
            this_stencil_delta_xk = np.asarray(r_cfs[this_node_ind])[:,0]
            this_stencil_delta_yk = np.asarray(r_cfs[this_node_ind])[:,1]
            this_stencil_wghts = np.asarray(w_ks[this_node_ind])

            rhs_vector = np.sum(np.vstack([this_stencil_wghts*this_stencil_delta_xk*this_stencil_delta_theta,
                                                this_stencil_wghts*this_stencil_delta_yk*this_stencil_delta_theta]), axis=1)

            t_xx = np.sum(this_stencil_wghts*this_stencil_delta_xk*this_stencil_delta_xk)
            t_xy = np.sum(this_stencil_wghts*this_stencil_delta_xk*this_stencil_delta_yk)
            t_yy = np.sum(this_stencil_wghts*this_stencil_delta_yk*this_stencil_delta_yk)

            lhs_matrix = np.matrix([[t_xx, t_xy], [t_xy, t_yy]]) 
            this_grad_out = lhs_matrix.I * rhs_vector[:, np.newaxis] 
            
            grad_x[this_step_ind, this_node_ind] = this_grad_out[0]
            grad_y[this_step_ind, this_node_ind] = this_grad_out[1]

    return grad_x, grad_y


def grad_ls_stencil_weight_calc(filereader, stencil_type=3, weight_type='inverse'):
    """

    Stencil types:
        1 - nbe, i.e. elements sharing a long edge
        2 -      i.e. elements sharing any node
        3 - nbe + nbe i.e. elements sharing a long edge and elements sharing a long edge with them


    """   

    stencils = {}

    if stencil_type == 1:
        for i, this_row in enumerate(filereader.grid.nbe):
            stencils[i] = this_row[this_row != -1]

    elif stencil_type == 3:
        for i, this_row in enumerate(filereader.grid.nbe):
            start_list = list(this_row[:])
            for this_node in this_row:
               for this_neighbour in filereader.grid.nbe[this_node,:]:
                    start_list.append(this_neighbour) 
            start_list = list(set(start_list))
            stencils[i] = np.asarray(start_list)[~np.isin(np.asarray(start_list),[-1,i])]
            

    r_cfs = {}
    for this_element, this_stencil in stencils.items():
        this_element_xy = [filereader.grid.xc[this_element], filereader.grid.yc[this_element]]
        stencil_vecs = []    
        for this_stencil_node in this_stencil:
            stencil_vecs.append([filereader.grid.xc[this_stencil_node] - this_element_xy[0], filereader.grid.yc[this_stencil_node] - this_element_xy[1]])
        r_cfs[this_element] = stencil_vecs

    w_ks = {}
    if weight_type == 'inverse':
        for this_element, this_r_cfs in r_cfs.items():
            wghts = []
            for this_rc in this_r_cfs:
                 wghts.append(1/np.sqrt(this_rc[0]**2 + this_rc[1]**2))
            w_ks[this_element] = wghts
 
    return stencils, r_cfs, w_ks


def green_gauss_gradient_method(field, filereader, skewness_correct=False, iterations=3, extended_stencil=False):

    # Calculate face centroids and surface vectors
    face_centroids_x = np.zeros(filereader.grid.triangles.shape)
    face_centroids_y = np.zeros(filereader.grid.triangles.shape)

    surface_vector_i = np.zeros(filereader.grid.triangles.shape)
    surface_vector_j = np.zeros(filereader.grid.triangles.shape)

    node_indexs = [[0,1], [1,2], [2,0]]

    for i, this_tri_nodes in enumerate(filereader.grid.triangles):
        for j, this_indexs in enumerate(node_indexs):
            this_tri_node_0 = this_tri_nodes[this_indexs[0]]
            this_tri_node_1 = this_tri_nodes[this_indexs[1]]

            face_centroids_x[i,j] = 0.5*(filereader.grid.x[this_tri_node_0] + filereader.grid.x[this_tri_node_1])
            face_centroids_y[i,j] = 0.5*(filereader.grid.y[this_tri_node_0] + filereader.grid.y[this_tri_node_1])

            surface_vector_i[i,j] = filereader.grid.y[this_tri_node_1] - filereader.grid.y[this_tri_node_0]
            surface_vector_j[i,j] = filereader.grid.x[this_tri_node_0] - filereader.grid.x[this_tri_node_1]

    # Interpolation factor
    interp_factor = np.zeros(filereader.grid.triangles.shape)
    for i, this_nbe in enumerate(filereader.grid.nbe):
        for j, this_indexs in enumerate(node_indexs):
            Fjfj = np.sqrt((filereader.grid.xc[this_nbe[j]] - face_centroids_x[i,j])**2 + (filereader.grid.yc[this_nbe[j]] - face_centroids_y[i,j])**2)
            Cfj = np.sqrt((filereader.grid.xc[i] - face_centroids_x[i,j])**2 + (filereader.grid.yc[i] - face_centroids_y[i,j])**2)
            interp_factor[i,j] = Fjfj/(Fjfj + Cfj)
    
    # Cell volumes
    filereader.calculate_areas()
    Vc = filereader.grid.areas

    # Green gauss with iterative correction
    if skewness_correct:
        phi_fdash = np.zeros(filereader.grid.triangles.shape)
        for i, this_nbe in enumerate(filereader.grid.nbe):
            for j, this_neighbour in enumerate(this_nbe):
                if this_neighbour != -1:
                    phi_fdash[i,j] = 0.5*(field[i]  + field[this_neighbour])
        
        grad_current = _phi_grad_calc(phi_fdash, [surface_vector_i, surface_vector_j], Vc)       

        for this_it in np.arange(0, iterations):
            new_phi = np.zeros(filereader.grid.triangles.shape)

            for i, this_nbe in enumerate(filereader.grid.nbe):
                for j, this_neighbour in enumerate(this_nbe):
                    if this_neighbour != -1:
                        r_f = np.asarray([face_centroids_x[i,j], face_centroids_y[i,j]])
                        r_C = np.asarray([filereader.grid.xc[i], filereader.grid.yc[i]])
                        r_F = np.asarray([filereader.grid.xc[this_neighbour], filereader.grid.yc[this_neighbour]])
                        r_adj_vec = r_f - 0.5*(r_C + r_F)
                        grad_add_vec = np.asarray(grad_current[i,:] + grad_current[this_neighbour,:])
                        grad_r_dot = r_adj_vec[0]*grad_add_vec[0] + r_adj_vec[1]*grad_add_vec[1]
                        new_phi[i,j] = phi_fdash[i,j] + 0.5*grad_r_dot

            grad_current = _phi_grad_calc(new_phi, [surface_vector_i, surface_vector_j], Vc)
            phi_fdash = copy.deepcopy(new_phi)

        grad_out = grad_current

    # Simple Green gauss
    else:
        phi_fs = np.zeros(filereader.grid.triangles.shape)

        for i, this_nbe in enumerate(filereader.grid.nbe):
            for j, this_neighbour in enumerate(this_nbe):
                if this_neighbour != -1:
                    this_interp_factor = interp_factor[i,j]
                    phi_fs[i,j] = this_interp_factor*field[i]  + (1- this_interp_factor)*field[this_neighbour] 
 
        grad_out = _phi_grad_calc(phi_fs, [surface_vector_i, surface_vector_j], Vc)

    return grad_out

def _phi_grad_calc(phis, surface_vector, Vc):
    grad_field = np.zeros([len(surface_vector[0]),2])
    for i, this_phi in enumerate(phis):
        grad_field[i,0] = (1/Vc[i])*np.sum(this_phi*surface_vector[0][i,:])
        grad_field[i,1] = (1/Vc[i])*np.sum(this_phi*surface_vector[1][i,:])

    return grad_field

def vertical_integration(field, filereader):
    pass

from cellcalc import get_primitive_hkl, MID, get_pri_vec_inplane
from interface_generator import core, print_near_axis, convert_vector_index
from csl_generator import print_list, getsigmas, get_theta_m_n_list
from numpy import array
from numpy import array, dot, round
from numpy.linalg import inv, norm
import numpy as np
import glob
import shutil
import os
from cellcalc import get_primitive_hkl, get_pri_vec_inplane, get_normal_index
from interface_generator import core, print_near_axis, convert_vector_index, write_trans_file
from numpy import array, dot, round, cross, ceil
from numpy.linalg import inv, det, norm
from numpy import cross, dot, ceil
from numpy.linalg import norm, inv

def getLatP():
        """
        read the Lattice Parameter
        """
        with open('atomsout','r') as f:
            lines=f.readlines()
            LatP = lines[-5:][0].split()[4].replace(';', '')
        return float(LatP)
		
def get_a_b(CSL, axis):
    hkl_perp_axis = MID(CSL, axis)
    a, b = get_pri_vec_inplane(hkl_perp_axis, CSL).T

    if(norm(cross(axis,[1,0,0])) < 1e-8):
        b = a + b
    elif (norm(cross(axis,[1,1,1])) < 1e-8):
        if dot(a,b) < 0:
            b = a + b
        b = a + b
    if (abs(norm(a) - norm(b)) < 1e-8):
        raise RuntimeError ('the tow vectors are identical!')

    return a.T, b.T
	
	
def get_STGB_MLs(CSL, n_1, n_2):
    hkl_1 = MID(CSL, n_1)
    hkl_2 = MID(CSL, n_2)

    return hkl_1, hkl_2
	
	
def get_expansion_xyz(cell):
    exp_x = ceil(100/norm(cell[:,0]))
    exp_y = ceil(20/norm(cell[:,1]))
    exp_z = ceil(20/norm(cell[:,2]))
    return exp_x, exp_y, exp_z
	
def get_gb_files(interface, hkl, axis, sigma, axis_name, hkl_name, ab, file, axis_num, ab_num, bond_length):
    interface.compute_bicrystal(hkl, normal_ortho = True, plane_ortho = True, lim = 50, tol = 1e-10)
#    print(f'U1 is {interface.bicrystal_U1}')
    half_cell = dot(interface.lattice_1, interface.bicrystal_U1)
    print(f'half_cell is {half_cell}')
#    print(f'lattice1_2 is {interface.lattice_1}')
    x,y,z = get_expansion_xyz(half_cell)
    axis_x, axis_y, axis_z = axis_name
    axis_name_num = 100*axis_x + 10*axis_y +axis_z
    
#    dirname = '{0} {1} {2} {3}'.format(np.array(axis_name, dtype = int),int(sigma), np.array(hkl_name, dtype = int), ab)
    dirname = f'{axis_name_num}_{int(sigma)}_{ab}.stgb'
    os.mkdir(dirname)
    os.chdir(dirname)

    if (axis_name == [1,1,1]):    
        interface.get_bicrystal(xyz_1 = [x,y,z], xyz_2 =[x,y,z],filename = 'atominfile', filetype='LAMMPS',mirror = True)
    else:
        interface.get_bicrystal(xyz_1 = [x,y,z], xyz_2 =[x,y,z],filename = 'atominfile', filetype='LAMMPS',mirror = False)

    eps = 1e-5
    define_bicrystal_regions(interface.xhi)
    CNID = dot(interface.orient, interface.CNID)
    length_1 = norm(CNID[:,0])
    length_2 = norm(CNID[:,1])
    area = norm(cross(CNID[:,0],CNID[:,1]))
    GB_area = norm(interface.lattice_bi[1])*norm(interface.lattice_bi[2])
    supercell_atoms = dot(interface.lattice_bi, interface.atoms_bi.T).T
    atoms_aroundgb = supercell_atoms[(supercell_atoms[:,0] >= (interface.xhi/2-bond_length*2)) & (supercell_atoms[:,0]<=(bond_length*2 + interface.xhi/2))]
    file.write(f'{CNID[:,0][1]} {CNID[:,0][2]} {CNID[:,1][1]} {CNID[:,1][2]} {length_1} {length_2} {area} {sigma} {axis_num} {ab_num} {GB_area} {len(atoms_aroundgb)} \n'.format(, , \
                                                                   , , \
                                                                   , , , , , , ))
    v1 = np.array([0,1.,0])*CNID[:,0][1] + np.array([0,0,1.])*CNID[:,0][2]
    v2 = np.array([0,1.,0])*CNID[:,1][1] + np.array([0,0,1.])*CNID[:,1][2]

    n1 = int(ceil(norm(v1)/0.2))
    n2 = int(ceil(norm(v2)/0.2))
    write_trans_file(v1,v2,n1,n2)
    get_potential_proto()

    os.chdir(os.pardir)

def define_bicrystal_regions(xhi):
    tol = 1e-5
    """
    generate a file defining some regions in the LAMMPS and define the atoms
    inside these regions into some groups.
    argument:
    region_names --- list of name of regions
    region_los --- list of the low bounds
    region_his --- list of the hi bounds
    """
    end_fixbulk1 = xhi/2-30
    start_fixbulk2 = xhi/2+30
    start_middle = xhi/2-20
    end_middle = xhi/2+30
    start_right = xhi/2 + tol
    start_bulk = 160
    end_bulk = 165


    with open('blockfile', 'w') as fb:
        fb.write('region fixbulk1 block EDGE {0:.16f} EDGE EDGE EDGE EDGE units box \n'.\
        format(end_fixbulk1))
        fb.write('region fixbulk2 block {0:.16f} EDGE EDGE EDGE EDGE EDGE units box \n'.\
        format(start_fixbulk2))
        fb.write('region middle block {0:.16f} {1:.16f} EDGE EDGE EDGE EDGE units box \n'.\
        format(start_middle,end_middle))
        fb.write('region right block {0:.16f} EDGE EDGE EDGE EDGE EDGE units box \n'.\
        format(start_right))
        fb.write('region bulk block {0:.16f} {1:.16f} EDGE EDGE EDGE EDGE units box \n'.\
        format(start_bulk,end_bulk))
        fb.write('group fixbulk1 region fixbulk1 \n')
        fb.write('group fixbulk2 region fixbulk2 \n')
        fb.write('group middle region middle \n')
        fb.write('group right region right \n')
        fb.write('group bulk region bulk \n')
		
def get_all_STGBs(axis_list, theta_list, sigma_list):
    eps = 1e-5
    file = open('CNIDs','w')
    my_interface = core('Al_mp-134_conventional_standard.cif', 'Al_mp-134_conventional_standard.cif')
    my_interface.parse_limit(du = 1e-4, S  =  1e-4, sgm1=100, sgm2=100, dd =  1e-4)
    factor = getLatP()/(2*norm(my_interface.lattice_1[0,0]))
    my_interface.lattice_1 =  my_interface.lattice_1*factor
    my_interface.lattice_2 =  my_interface.lattice_2*factor
    my_interface.conv_lattice_1 =  my_interface.conv_lattice_1*factor
    my_interface.conv_lattice_2 =  my_interface.conv_lattice_2*factor  
    count = 1
    bond_length = getLatP()/np.sqrt(2)
    for i in range(len(axis_list)):
        axis = axis_list[i]
        axis_name = axis_list[i]
        axis = convert_vector_index(my_interface.conv_lattice_1, my_interface.lattice_1, axis)
        for j in range(len(sigma_list[i])):
            print(axis)
            print(theta_list[i][j])
            sigma = sigma_list[i][j]
            my_interface.search_one_position(axis,theta_list[i][j]-0.001,1,0.001)
            CSL = my_interface.CSL
            n_1, n_2 = get_a_b(CSL, dot(my_interface.lattice_1,axis))
            hkl_1, hkl_2 = get_STGB_MLs(my_interface.lattice_1, n_1, n_2)
            hkl_name_1 = get_primitive_hkl(hkl_1, my_interface.lattice_1, my_interface.conv_lattice_1)
            hkl_name_2 = get_primitive_hkl(hkl_2, my_interface.lattice_2, my_interface.conv_lattice_2)

            get_gb_files(my_interface, hkl_1, axis, sigma, axis_name, hkl_name_1, 'a_{}'.format(count), file, i+1, 1, bond_length)

            get_gb_files(my_interface, hkl_2, axis, sigma, axis_name, hkl_name_2, 'b_{}'.format(count+1), file, i+1, 2, bond_length)

            
            count += 2
    file.close()
	
	
def get_theta_sigma_list(axis_list):
    theta_list = []
    sigma_list = []
    for i in axis_list:
        lists, thetas = getsigmas(i, 100)
        theta_list.append(thetas)
        sigma_list.append(lists)
    theta_list[0][0:0], theta_list[1][0:0] = [0],[0]
    sigma_list[0][0:0], sigma_list[1][0:0] = [1],[1]
    return theta_list,sigma_list
	
def get_potential_proto():
    # get potential files and input files from the file named potential_proto
    # potential_proto file should be subdirectory of directory that includes program
    input_file_path = os.path.join(os.pardir,'potential_proto\\*')
    for file in glob.glob(input_file_path):

        shutil.copy(file,os.getcwd())
		
def replace_func(fname, replace_set):
    target, replace = replace_set
    
    with open(fname, 'r') as f1:
        tmp_list =[]
        for row in f1:
            if row.find(target) != -1:
                tmp_list.append(replace)
            else:
                tmp_list.append(row)
    
    with open(fname, 'w') as f2:
        for i in range(len(tmp_list)):
            f2.write(tmp_list[i])
			
def getLatE():
        """
        read the Energy of single crystal 
        """
        with open('atomsout','r') as f:
            lines=f.readlines()
            LatE = lines[-4:][0].split()[4].replace(';', '')
        return float(LatE)
		
# this is the function for optmizing energy
def opt_E():
    proto2_file_path = os.path.join(os.getcwd(),'potential_proto\\proto.in')
    replace_setA = ('variable minimumenergy equal', f'variable minimumenergy equal {getLatE()}\n') 
    replace_func(proto2_file_path,replace_setA)
	

def main():
#    os.system('mpirun -np 12 lmp_mpi < proto.in > atomsout')
    
    axis_list = [[1,0,0], [1,1,0], [1,1,1]]
    theta_list, sigma_list = get_theta_sigma_list(axis_list)
    opt_E()
    
    get_all_STGBs(axis_list, theta_list, sigma_list)
if __name__ == '__main__':
    main()
	
	

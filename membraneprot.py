'''When the PDB of a transmembrane protein is given, returns a vmd molecule containing the transmembrane protein
plus two slabs of dummy atoms representing the cellular membrane.'''

# Mariona Torrens and Inés Sentís


from htmd import *
import argparse
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import time
from htmd.molecule.util import maxDistance

parser=argparse.ArgumentParser( description = "Given the PDB file of a transmembrane protein, it creates an approximation of how it is inserted in the cellular membrane")
parser.add_argument("-i", "--input",
                    dest = "PDBprot",
                    action = "store",
                    default = None,
                    required = True,
                    help = "Input must be a string written in command line specifying the PDB ID of the protein.")

op = parser.parse_args() 

def dummy_leaflet(prot,name, sp =3):
    '''Returns a leaflet of dummy atoms, which represents a layer of the membrane. 
    Each dummy atom is placed at the xy plane separated from the other by a distance of sp Angstroms.'''
    pcenter = np.mean(prot.get('coords','protein'),axis=0)
    N = int(maxDistance(prot, 'all',pcenter))
    dl=Molecule()
    dl.empty(N*N) 
    dl.set("name",name)
    dl.set("record","HETATM")   
    resids=np.array(range(1,N*N+1))
    dl.set("resid",resids)
    dl.set("resname","DUM")    
    coords=[]
    for i in range(0,sp*N,sp):
        for j in range(0,sp*N,sp):
            coords.append([i,j,0])
    dl.set("coords",coords)        
    return (dl)

def prot_vector(mol_int):
    '''Calculates the protein vector as the average vector perpendicular to the aromatic atoms of the protein.'''
    CGs = mol_int.get("serial", "protein and aromatic and name CG")
    x=[]
    y=[]
    z=[]
    for serial in CGs:
        c1=mol_int.get("coords", "protein and serial " +str(serial))
        c2=mol_int.get("coords", "protein and serial " +str(serial+1))
        c3=mol_int.get("coords", "protein and serial " +str(serial+2))
        if len(c1)!=0:                                          
            v1=[c1[0]-c2[0],c1[1]-c2[1], c1[2]-c2[2]]
            v2=[c1[0]-c3[0],c1[1]-c3[1], c1[2]-c3[2]]
            # ortogonal vector:
            ort_v= [v1[1]*v2[2] - v1[2]*v2[1] , v1[0]*v2[2] - v1[2]*v2[0], v1[0]*v2[1] - v1[1]*v2[0]]
            x.append(ort_v[0])
            y.append(ort_v[1])
            z.append(ort_v[2])
    total=len(x)    
    prot_vect=[sum(x)/total , sum(y)/total, sum(z)/total]
    return (prot_vect)

def rotate(mol_int,prot_vect):
    '''From a protein vector, it computes the rotation matrices needed to rotate the protein so that the its
    vector becomes perpendicular to the membrane. Then, applies the matrices to rotate the protein.'''
    xy_pr=prot_vect[0:2]
    x_axis=[1,0]
    cos_a=(xy_pr[0]*x_axis[0] + xy_pr[1]*x_axis[1])/(((xy_pr[0]**2 + xy_pr[1]**2)**0.5)*((x_axis[0]**2 + x_axis[1]**2)**0.5))
    a_angle=np.arccos(cos_a)
    
    xz_pr=prot_vect[0:3:2]
    z_axis=[0,1]
    cos_b=(xz_pr[0]*z_axis[0] + xz_pr[1]*z_axis[1])/(((xz_pr[0]**2 + xz_pr[1]**2)**0.5)*((z_axis[0]**2 + z_axis[1]**2)**0.5))
    b_angle=np.arccos(cos_b)
    
    
    z_rotation = rotationmatrix.rotationMatrix([0, 0, 1],a_angle)
    y_rotation = rotationmatrix.rotationMatrix([0, 1, 0],b_angle)
    multiplication = np.dot(y_rotation, z_rotation)
    
    pcenter = np.mean(mol_int.get('coords','protein'),axis=0)
    mol_int.rotateBy(multiplication,center=pcenter, sel="protein")

def hydroph_calculation(mol_int,prot_chains,coords_z_max,coords_z_min):
    '''Calculate the hdrophobicity of the protein along the z axis. For each iteration, the hydrophobicity is calculated
    considering the aminoacids found within a window of 10 Angstroms of the z axis. Returns a list of the hidrophobicity
    score of each window (cake), and a list of all the intervals of z values considered in string format (x_axis) and in 
    list format (interval).'''
    hydro={"ILE":4.5 , "VAL":4.2 , "LEU":3.8 , "PHE":2.8 ,"CYS":2.5 , "MET":1.9 , "ALA":1.8 ,"GLY":-0.4 , "THR":-0.7, "SER":-0.8,"TRP":-0.9,"TYR":-1.3,"PRO":-1.6,"HIS":-3.2,"GLU":-3.5,"GLN":-3.5,"ASP":-3.5,"ASN":-3.5,"LYS":-3.9,"ARG":-4.5}
    score_sum=int()
    cake=list()
    x_axis=[]
    interval=[]
    
    print("Calculating hydrophobicity:")    
    rg = range(coords_z_min,coords_z_max,2)
    for z in rg:
        score_sum=0
        percent = '{0:.1f}'.format((rg.index(z)/len(rg))*100)
        print("--- z=", z, " (",percent,"% ) ----")
        x_axis.append("(" + str(z) + " - " + str(z+10) + ")")
        interval.append([z,z+10])
        for chain in prot_chains:
            coords =mol_int.get("coords","name CA and protein and chain " + str(chain))
            residues =mol_int.get("resid","name CA and protein and chain " + str(chain))
            res_coo = dict(zip(residues, coords))
            for res, c in res_coo.items():
                if  z<c[2]<=z+10:
                    resname=mol_int.get("resname", "name CA and protein and resid "+str(res))
                    resname = str(resname[0])
                    if resname in hydro.keys():
                        score_sum += hydro[resname]
        cake.append(score_sum)
    print("Done!")
    return (cake,x_axis,interval)

def extremes_list(whole_list, max_index, cutoff=3): 
    '''Takes a list of hydrophobicity scores or a list of z values for which each score has been calculated, 
    and divides it into two sublist. The region within 6 Angstroms (3*3) of the z value with maximum hydrophobicity is 
    excluded. Returns two sublist with the remaining elements at the left and right extremes of the protein.'''
    right = np.array(whole_list[max_index+cutoff:])
    left = np.array(whole_list[:max_index-cutoff])
    return(left, right)

def find_z_position (side, sublist_scores, sublist_interval,coords_z_min,coords_z_max):
    '''Takes a sublist of hydrophobicity scores (the ones at the right or left extreme of the protein) 
    and the corresponding sublist of z positions. Finds the local minimums of hydrophobicity, 
    which are candidate points to place the leaflet. The local minimum nearest to the center of the plot 
    gives the position where the leaflet will be placed.'''
    min_sub_index=argrelextrema(sublist_scores, np.less)
    if min_sub_index == True:
        if side == "l":
            position_index= np.amax(min_sub_index)
        elif side == "r":
            position_index=np.amin(min_sub_index)
        position=sublist_interval[position_index]
        pos_mean = (position[0]+position[1])/2
        return (pos_mean)
    else:
        if side == "l":
            position = coords_z_min
            return(position)
        elif side == "r":
            position = coords_z_max
            return(position)

def set_new_membrane_coords(mol_int,dl_indicator, pos_mean):
    '''Given a z value (pos_mean), sets the new coordinates to each dummy atom of a leaflet.'''
    dl_c= mol_int.get("coords","resname DUM and name " +dl_indicator)
    for xyz in dl_c:
        xyz[2]=pos_mean
    mol_int.set(field="coords",value=dl_c, sel="resname DUM and name " +dl_indicator)

def view_vmd(mol_int):    
    '''visualization of the molecule object containing a protein and leaflets of dummy atom in VMD.'''
    htmd.config(viewer='vmd')
    mol_int.reps.remove()   
    mol_int.reps.add(sel='protein', style='NewCartoon', color='2')
    mol_int.reps.add(sel='resname DUM', style='Licorice', color='name')
    print(mol_int.reps)    
    mol_int.view()
    time.sleep(6000)


def insert_prot_to_membrane(protein):
    '''Main funcion of the module'''
    # Create the protein object
    prot = Molecule(protein)  
    prot.filter("protein")
    #Creating the membrane
    dl1 = dummy_leaflet(prot,"O")
    dl2 = dummy_leaflet(prot,"N")
    membr = dl1.copy()
    membr.append(dl2)
    # Integration of the protein to the membrane
    prot.center()
    membr.center()
    mol_int=prot.copy()
    mol_int.append(membr)
    # Obtaining the protein vector
    prot_vect=prot_vector(mol_int)
    # Rotation of the protein
    rotate(mol_int,prot_vect)
    # Position of the membrane at the z axis
    prot_chains=np.unique(prot.get("chain", "protein"))
    coords_all = mol_int.get("coords","protein and name CA")
    coords_z=[c[2] for c in coords_all]
    coords_z_max = int(max(coords_z))+1 
    coords_z_min = int(min(coords_z))-1
    (cake,x_axis,interval)=hydroph_calculation(mol_int,prot_chains,coords_z_max,coords_z_min)
    #Plot hydrophobicity scores
    x=list(range(1,len(cake)+1))
    plt.xticks(x,x_axis, rotation="vertical")
    plt.plot(x, cake)
    plt.show()
    # New positin of the membrane
    cake=list(cake)
    max_index = cake.index(max(cake))	
    (subcake_l,subcake_r) = extremes_list(cake,max_index)
    (sub_int_l,sub_int_r) = extremes_list(interval,max_index)
    pos_l_mean = find_z_position('l',subcake_l, sub_int_l,coords_z_min,coords_z_max)
    pos_r_mean = find_z_position('r',subcake_r, sub_int_r,coords_z_min,coords_z_max)
    set_new_membrane_coords(mol_int,"O", pos_l_mean)
    set_new_membrane_coords(mol_int,"N", pos_r_mean)
    # VMD visualisation of the result
    view_vmd(mol_int)



if __name__ == "__main__":
    insert_prot_to_membrane(op.PDBprot)
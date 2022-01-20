# %% [markdown]
'''
# PyMAPDL Example
In this 'VS Code 'Python Code file' a built-in example is selected, its model description is printed and results are selected and plotted.  
Selection is done by assigning variables
'''

# %%
from turtle import window_height
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ansys.mapdl.core import launch_mapdl


# create remote points and joints
def make_sbc(mapdl, x, y, z, tid='', cid='', tdof='111111', _type='rigid', pinb=None):
    """create remote point at lspecified location, mapdl object must have nodes selected prior to passing to this function.  The pinb parameter may be used to filter nodes to within a radius of the pilot node.

    Args:
        mapdl: active pyMAPDL object with distributed nodes pre-selected
        x (float): X location of pilot node in global coordinate system (cycs=0)
        y (float): Y location of pilot node
        z (float): Z location of pilot node
        tid (str, optional): user-specified target element type number. Defaults to '' (next available number).
        cid (str, optional): user-specified contact element type number. Defaults to '' (next available number).
        tdof (str, optional): active DOFs on the pilot node. Defaults to '111111'.
        _type (str, optional): Surface based constraint type. Defaults to 'rigid'.

    Returns:
        n_pilot: pilot node id
        rid: real constant id for surface based constraint pair
    """
    # target element type
    tid = mapdl.et(itype=tid, ename=170, kop2=1, kop4=tdof)
    mapdl.keyopt(itype=tid, knum=7, value=0)
    kop4 = {'rigid': 0, 'deformable': 1}[_type]
    cid = mapdl.et(itype=cid, ename=175, kop2=2, kop4=kop4)
    mapdl.keyopt(itype=cid, knum=12, value=5)
    rmx = mapdl.get(entity='RCON', item1='NUM',it1num='MAX')
    rid = rmx+1
    mapdl.r(nset=rid)
    
    if pinb:
        # create local sperical cs, reselect nodes within pinball radius
        mapdl.local(kcn=999, kcs=2, xc=x, yc=y, zc=z)
        cs_id = mapdl.parameters.csys
        mapdl.nsel(type_='r', item='LOC', comp='X', vmin=0., vmax=pinb)
        mapdl.csys(0)
        mapdl.csdele(cs_id)
    # create contact elements
    mapdl.type(cid)
    mapdl.real(rid)
    mapdl.mat(rid)
    n_nodes = mapdl.get(entity='NODE', item1='COUNT')
    nid = mapdl.get(entity='NODE', item1='NUM', it1num='MIN')
    for i in range(int(n_nodes)):
        mapdl.e(nid)
        nid = q.ndnext(nid)
    # create target pilot element
    mapdl.allsel()
    n_pilot = mapdl.n(x=x, y=y, z=z)
    mapdl.tshap('pilo')
    mapdl.type(tid)
    mapdl.real(rid)
    mapdl.mat(rid)
    e_pilot = mapdl.e(n_pilot)
    mapdl.tshap()
    return n_pilot, rid


def plot_nodal_disp(grid, result, idx_set=0, dof='uZ', scale=2.):
    """plots nodal displacement for a mapdl result set.

    Args:
        grid (vtk grid): grid object from mapdl.mesh.grid (prior to exiting mapdl)
        result (mapdl result): result object from solved mapdl, mapdl.result
        idx_set (int, optional): 0-based set index for result sets. Defaults to 0.
        dof (str, optional): direction of nodal displacement.  Must be one of 'uX', 'uY', 'uZ'. Defaults to 'uZ'.
    """
    gridn = grid.point_data["ansys_node_num"]
    nnum, disp = result.nodal_displacement(idx_set)
    idx_map = {'uX':0, 'uY':1, 'uZ':2}
    u = pd.DataFrame(disp, index=nnum).reindex(gridn, fill_value=0.)
    grid.point_data[dof] = u.loc[:,idx_map[dof]].values
    grid.set_active_scalars(dof)
    grid.point_data['disp'] = u.values[:, :3]
    grid.warp_by_vector('disp', factor=scale).plot(cmap='plasma', show_edges=True)
    return None


mapdl = launch_mapdl(override=True, license_type="ansys", cleanup_on_exit=True)
mapdl.clear()
mapdl.prep7()
mapdl.units("BIN")
mapdl.csys(kcn=0)
q = mapdl.queries

# PV NSM
nsm = .002

# Faceseet material
tfs = 0.03
mid = 1
Ex = 10.0e6
nu_xy = 0.3
dens = 0.1 / 386.089
mapdl.mp('EX', mid, Ex)
mapdl.mp('PRXY', mid, nu_xy)
mapdl.mp('DENS', mid, dens)

# Core material
tc = 0.25
mid = 2
Ez = 75e3
Gxz = 45e3
Gyz = 22e3
dens = (3.1 / 12**3) / 386.089

mapdl.mp('EX', mid, 10)
mapdl.mp('EY', mid, 10)
mapdl.mp('EZ', mid, Ez)
mapdl.mp('GXY', mid, 10.)
mapdl.mp('GXZ', mid, Gxz)
mapdl.mp('GYZ', mid, Gyz)
mapdl.mp('PRXY', mid, 0.)
mapdl.mp('PRXZ', mid, 0.)
mapdl.mp('PRYZ', mid, 0.)
mapdl.mp('DENS', mid, dens)

# create shell property
mapdl.et(1, 181, kop1=0, kop3=2)
mapdl.sectype(secid=1, type_="SHELL")
mapdl.secdata(tfs, 1, 0, 3)
mapdl.secdata(tc, 2, 0, 3)
mapdl.secdata(tfs, 1, 0, 3)
mapdl.secoffset('MID')
mapdl.seccontrol(val4=nsm)

# create joint property
mid = 10
K_1 = 1e6
K_2 = 1e6
K_3 = 1e6
K_4 = 1e3
K_5 = 1e5
K_6 = 1e5
mapdl.et(mid,250,0)
mapdl.r(mid)
mapdl.rmore(K_1, K_2, K_3, K_4, K_5, K_6)

# create geometry (areas)
w = 20.
h = 20.
xc = 0.
yc = 0.
dx = 2.0

a1 = mapdl.blc4(xcorner=xc, ycorner=yc, width=w, height=h)
mapdl.agen(itime=2, na1='ALL', dx=w+dx)

# mesh
mapdl.allsel()
mid=1
mapdl.aatt(mat=mid, type_=mid, secn=mid)
mapdl.aesize('all', dx/2.)
mapdl.mshape(0, '2D')
mapdl.mopt('SPLIT', 'OFF')
mapdl.smrtsize(sizlvl=4)
mapdl.csys(kcn=0)
mapdl.amesh("all")


# %%
# base remote points
mapdl.lsel(type_='s', item='LOC', comp='X', vmin=0.)
mapdl.nsll(type_='s', nkey=1)
n_base_1, r_base = make_sbc(mapdl, 0., h/4., 0., pinb=dx)
mapdl.lsel(type_='s', item='LOC', comp='X', vmin=0.)
mapdl.nsll(type_='s', nkey=1)
n_base_2, r_base = make_sbc(mapdl, 0., 3*h/4., 0., pinb=dx)

# ref remote points
mapdl.lsel(type_='s', item='LOC', comp='X', vmin=w)
mapdl.nsll(type_='s', nkey=1)
nr1, rr1 = make_sbc(mapdl, w+dx/2., h/4., 0., pinb=dx)
mapdl.lsel(type_='s', item='LOC', comp='X', vmin=w)
mapdl.nsll(type_='s', nkey=1)
nr2, rr2 = make_sbc(mapdl, w+dx/2., 3*h/4., 0., pinb=dx)

# mob remote points
mapdl.lsel(type_='s', item='LOC', comp='X', vmin=w+dx)
mapdl.nsll(type_='s', nkey=1)
nm1, rm1 = make_sbc(mapdl, w+dx/2., h/4., 0., pinb=dx)
mapdl.lsel(type_='s', item='LOC', comp='X', vmin=w+dx)
mapdl.nsll(type_='s', nkey=1)
nm2, rm2 = make_sbc(mapdl, w+dx/2., 3*h/4., 0., pinb=dx)

# Add joints
mapdl.allsel()
mid = 10
mapdl.mat(mid)
mapdl.type(mid)
mapdl.real(mid)
mapdl.e(nr1,nm1)
mapdl.e(nr2,nm2)

# %%
# solving
mapdl.allsel()
mapdl.slashsolu()
mapdl.outres(item='ALL', freq='NONE')
mapdl.outres(item='NSOL', freq='ALL')
mapdl.outres(item='RSOL', freq='ALL')
mapdl.outres(item='ESOL', freq='ALL')
mapdl.outres(item='VENG', freq='ALL')
mapdl.outres(item='MISC', freq='ALL')
mapdl.d(n_base_1, "ALL")
mapdl.d(n_base_2, "ALL")
mapdl.antype('MODAL')
mapdl.modopt('LANB', 10)
mapdl.mxpand(elcalc="YES")
o = mapdl.solve()
print(o)
mapdl.aux2()
mapdl.combine(filetype='RST')
mapdl.post1()


# %%
grid = mapdl.mesh.grid
result = mapdl.result
plot_nodal_disp(grid, result, idx_set=0, scale=10.)
mapdl.exit()


# %%

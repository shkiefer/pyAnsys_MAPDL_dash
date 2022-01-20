# %% [markdown]
'''
# PyMAPDL Example
In this 'VS Code 'Python Code file' a built-in example is selected, its model description is printed and results are selected and plotted.  
Selection is done by assigning variables
'''

# %%
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
    q = mapdl.queries
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


def get_mass(mapdl):
    mapdl.post1()
    mapdl.allsel()
    mapdl.set(1, 1)
    mtot = np.zeros(6)
    for j, d in enumerate(['X', 'Y', 'Z']):
        mtot[j] = mapdl.get(entity='ELEM', item1='MTOT', it1num=d)
        mtot[j+3] = mapdl.get(entity='ELEM', item1='IOR', it1num=d)    
    return pd.Series(mtot, index=['TX', 'TY', 'TZ', 'RX', 'RY', 'RZ'])


def get_mem(mapdl):
    mapdl.post1()
    mapdl.allsel()
    mapdl.set(1, 1)
    df_mass = get_mass(mapdl)

    n_modes=mapdl.get(entity='ACTIVE', item1='SET', it1num='NSET')
    effms = []
    for i in range(int(n_modes)):
        mapdl.set(1, i+1)
        freq = mapdl.get(entity='ACTIVE', item1='SET', it1num='FREQ')
        effmi = np.zeros(13)
        effmi[0] = freq
        for j, d in enumerate(['X', 'Y', 'Z', 'ROTX', 'ROTY', 'ROTZ']):
            effmi[j+1] = mapdl.get(entity='MODE', entnum=i+1, item1='EFFM', item2='DIREC', it2num=d)
        effmi[7:] = effmi[1:7] / df_mass.values
        effms.append(effmi)
    effm = np.vstack(effms)
    df = pd.DataFrame(effm, columns=['freq', 'TX_abs', 'TY_abs', 'TZ_abs', 'RX_abs', 'RY_abs', 'RZ_abs', 'TX', 'TY', 'TZ', 'RX', 'RY', 'RZ'])
    return df


def get_sene(mapdl, comp_name=None):
    """Gets strain energy for each result set of mapdl object

    Args:
        mapdl: pyAnsy mapdl object
        comp_name (String, optional): Name of component to get strain energy of or None for full model. Defaults to None.

    Returns:
        pd.DataFrame: Pandas Dataframe with fequencies and strain energies 
    """
    mapdl.post1()
    mapdl.allsel()
    n_modes=mapdl.get(entity='ACTIVE', item1='SET', it1num='NSET')
    mapdl.etable('ERAS')
    if comp_name is not None:
        mapdl.cmsel(type_='s', name=comp_name, entity='ELEM')
    
    freqs = []
    senes = []
    for i in range(int(n_modes)):
        mapdl.set(1, i+1)
        freq = mapdl.get(entity='ACTIVE', item1='SET', it1num='FREQ')
        freqs.append(freq)
        mapdl.etable(lab='mysene', item='SENE')
        mapdl.sabs(1)
        mapdl.ssum()
        sene = mapdl.get(entity='SSUM', entnum=0, item1='ITEM', it1num='mysene')
        senes.append(sene)
    return pd.DataFrame({'freq': freqs, 'sene': senes})


def get_sene_comps(mapdl, comp_names):
    """Gets relative strain energy for each component & mode

    Args:
        mapdl: pyAnsy mapdl object
        comp_names (List): List of strings of component names

    Returns:
        pd.DataFrame: pandas dataFrame with frequences and component relative strain energies
    """
    sene_total = get_sene(mapdl).loc[:, 'sene'].values
    dfs=[]
    for c in comp_names:
        df1 = get_sene(mapdl, comp_name=c)
        df = pd.DataFrame({'freq': df1['freq'].values, c: df1['sene'].values / sene_total})
        dfs.append(df)
    df_sene = pd.concat([d.set_index('freq') for d in dfs], axis=1, ignore_index=False, sort=True) 
    return df_sene


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


def make_pv_array(panel_w, panel_h, n_panels=2, nsm_pv = 0.002, tc=0.25, tfs=0.03, k_hinge=10e3, dx_panel=2.):
    mapdl = launch_mapdl(override=True, license_type="ansys", cleanup_on_exit=True)
    mapdl.clear()
    mapdl.prep7()
    mapdl.units("BIN")
    mapdl.csys(kcn=0)

    # Faceseet material
    mid = 1
    Ex = 10.0e6
    nu_xy = 0.3
    dens = 0.1 / 386.089
    mapdl.mp('EX', mid, Ex)
    mapdl.mp('PRXY', mid, nu_xy)
    mapdl.mp('DENS', mid, dens)

    # Core material
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
    mapdl.seccontrol(val4=nsm_pv)

    # create joint property
    mid = 10
    K_1 = 1e6
    K_2 = 1e6
    K_3 = 1e6
    K_4 = 1e5
    K_5 = k_hinge
    K_6 = 1e5
    mapdl.et(mid,250,0)
    mapdl.r(mid)
    mapdl.rmore(K_1, K_2, K_3, K_4, K_5, K_6)

    # create geometry (areas)
    a1 = mapdl.blc4(xcorner=0., ycorner=-panel_w/2, width=panel_h, height=panel_w)
    mapdl.agen(itime=n_panels, na1='ALL', dx=panel_h + dx_panel)

    # mesh
    mapdl.allsel()
    mid=1
    mapdl.aatt(mat=mid, type_=mid, secn=mid)
    mapdl.aesize('all', dx_panel / 2.)
    mapdl.mshape(0, '2D')
    mapdl.mopt('SPLIT', 'OFF')
    mapdl.smrtsize(sizlvl=4)
    mapdl.csys(kcn=0)
    mapdl.amesh("all")

    # base remote points
    mapdl.lsel(type_='s', item='LOC', comp='X', vmin=0.)
    mapdl.nsll(type_='s', nkey=1)
    n_base_1, r_base = make_sbc(mapdl, 0., -panel_w/4., 0., pinb=dx_panel)
    mapdl.lsel(type_='s', item='LOC', comp='X', vmin=0.)
    mapdl.nsll(type_='s', nkey=1)
    n_base_2, r_base = make_sbc(mapdl, 0., panel_w/4., 0., pinb=dx_panel)

    if n_panels > 1:
        for i in range(n_panels-1):
            x_ref = panel_h * (i + 1) + dx_panel * i
            x_mob = x_ref + dx_panel
            # ref remote points
            mapdl.lsel(type_='s', item='LOC', comp='X', vmin=x_ref)
            mapdl.nsll(type_='s', nkey=1)
            nr1, rr1 = make_sbc(mapdl, x_ref + dx_panel / 2., -panel_w / 4., 0., pinb=dx_panel)
            mapdl.lsel(type_='s', item='LOC', comp='X', vmin=x_ref)
            mapdl.nsll(type_='s', nkey=1)
            nr2, rr2 = make_sbc(mapdl, x_ref + dx_panel / 2., panel_w / 4., 0., pinb=dx_panel)

            # mob remote points
            mapdl.lsel(type_='s', item='LOC', comp='X', vmin=x_mob)
            mapdl.nsll(type_='s', nkey=1)
            nm1, rm1 = make_sbc(mapdl, x_ref + dx_panel / 2., -panel_w / 4., 0., pinb=dx_panel)
            mapdl.lsel(type_='s', item='LOC', comp='X', vmin=x_mob)
            mapdl.nsll(type_='s', nkey=1)
            nm2, rm2 = make_sbc(mapdl, x_ref + dx_panel / 2., panel_w / 4., 0., pinb=dx_panel)

            # Add joints
            mapdl.allsel()
            mid = 10
            mapdl.mat(mid)
            mapdl.type(mid)
            mapdl.real(mid)
            mapdl.e(nr1, nm1)
            mapdl.e(nr2, nm2)

    # make components
    mapdl.esel(type_='s', item='TYPE', vmin=1)
    mapdl.cm(cname='shell', entity='ELEM')
    mapdl.esel(type_='s', item='TYPE', vmin=10)
    mapdl.cm(cname='joint', entity='ELEM')
    mapdl.allsel()

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
    mapdl.modopt('LANB', 3)
    mapdl.mxpand(elcalc="YES")
    o = mapdl.solve()

    grid = mapdl.mesh.grid
    result = mapdl.result
    
    df_sene = get_sene_comps(mapdl, ['shell', 'joint'])
    mapdl.exit()
    return result, grid, df_sene




# %%
panel_w = 20.
panel_h = 20.
tc = 0.25
r, g, df_sene = make_pv_array(panel_w=panel_w, panel_h=panel_h, tc=tc)


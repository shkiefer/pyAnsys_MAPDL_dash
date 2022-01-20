import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash.exceptions import PreventUpdate
from dash import dcc, Output, Input, State, html, dash_table
from dash.dash_table import FormatTemplate 
from dash.dash_table.Format import Format, Scheme
from dash.long_callback import DiskcacheLongCallbackManager
import dash_bootstrap_components as dbc
import dash_vtk
import diskcache
from dash_vtk.utils import to_mesh_state

from ansys.mapdl.core import launch_mapdl, Mapdl


APP_ID = 'pyAnsys'

def make_colorbar(title, rng, bgnd='rgb(51, 76, 102)'):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       colorscale='plasma',
                       showscale=True,
                       cmin=rng[0],
                       cmax=rng[1],
                       colorbar=dict(
                           title_text=title, 
                           title_font_color='white', 
                           title_side='top',
                           thicknessmode="pixels", thickness=50,
                           #  lenmode="pixels", len=200,
                           yanchor="middle", y=0.5, ypad=10,
                           xanchor="left", x=0., xpad=10,
                           ticks="outside",
                           tickcolor='white',
                           tickfont={'color':'white'}
                           #  dtick=5
                       )
        ),
            hoverinfo='none'
        )
    )
    fig.update_layout(width=150, margin={'b': 0, 'l': 0, 'r': 0, 't': 0}, autosize=False, plot_bgcolor=bgnd)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


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
    df['mode'] = np.arange(len(df)) + 1
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
    df_sene['mode'] = np.arange(len(df)) + 1
    df_sene = df_sene.reset_index(drop=False)
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
    grid = grid.warp_by_vector('disp', factor=scale)
    return grid


def make_pv_array(panel_w, panel_h, n_panels=2, nsm_pv=0.002, tc=0.25, tfs=0.03, k_hinge=10e3, dx_panel=2., prog=None):
    mapdl = launch_mapdl(override=True, license_type="ansys", cleanup_on_exit=True, additional_switches="-smp")
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

    if prog:
        prog((str(2), str(10)))
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
    if prog:
        prog((str(3), str(10)))

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

    if prog:
        prog((str(4), str(10)))
    grid = mapdl.mesh.grid
    result = mapdl.result
    
    df_sene = get_sene_comps(mapdl, ['shell', 'joint']).set_index('mode')
    df_mem = get_mem(mapdl).loc[:, ['mode', 'TX', 'TY', 'TZ', 'RX', 'RY', 'RZ']].set_index('mode')
    df_sene_mem = pd.concat([df_mem, df_sene], axis=1, ignore_index=False, sort=True).reset_index(drop=False)

    mapdl.exit()
    if prog:
        prog((str(6), str(10)))
    return result, grid, df_sene_mem


def data_bars(df, column, bar_max=1., bar_min=0., clr='#0074D9'):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [
        ((bar_max - bar_min) * i) + bar_min
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    {clr} 0%,
                    {clr} {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage, clr=clr)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles




cache = diskcache.Cache('./cache')
lcm = DiskcacheLongCallbackManager(cache)
app = dash.Dash(
    __name__, 
    long_callback_manager=lcm,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.layout = dbc.Container([
    html.H1('PyAnsys MAPDL in a Dash App', className="mt-3"),
    html.P('Design your solar array! (not really, dont use these assumed properties)'),
    dbc.Row([
        dbc.Col([
            dbc.Form([
                dbc.Label('Number of Panels', html_for=f'{APP_ID}_n_panels_input'),
                dbc.Input(id=f'{APP_ID}_n_panels_input', type="number", min=0, max=10, step=1, value=2)
            ])
        ]),
        dbc.Col([
            dbc.Form([
                dbc.Label('Panel Width (in)', html_for=f'{APP_ID}_w_panel_input'),
                dbc.Input(id=f'{APP_ID}_w_panel_input', type="number", min=1., value=20, step='any'),
                dbc.FormText("Y direction"),
            ])

        ]),
        dbc.Col([
            dbc.Form([
                dbc.Label('Panel Height (in)', html_for=f'{APP_ID}_h_panel_input'),
                dbc.Input(id=f'{APP_ID}_h_panel_input', type="number", min=1., value=20, step='any'),
                dbc.FormText("X / deployment direction")
            ])
        ]),       
        dbc.Col([
            dbc.Form([
                dbc.Label('Panel Core Thickness (in)', html_for=f'{APP_ID}_tc_input'),
                dbc.Input(id=f'{APP_ID}_tc_input', type="number", min=.125, value=.25, step='any'),
            ])

        ]), 
    ]),
    dbc.ButtonGroup([
        dbc.Button('Hide / Show Additional Options', id=f'{APP_ID}_options_collapse_button', color='secondary'),
        dbc.Button('Solve', id=f'{APP_ID}_solve_button', color='primary'),        
    ]),
    dbc.Collapse(
        id=f'{APP_ID}_options_collapse',
        is_open=False,
        children=[
            dbc.Row([
                dbc.Col([
                    dbc.Form([
                        html.Div([
                            dbc.Label('Non Structural Mass (lbs/in^2)', html_for=f'{APP_ID}_nsm_input'),
                            dbc.Input(id=f'{APP_ID}_nsm_input', type="number", min=0., value=0.002, step='any'),
                            dbc.FormText("Photo-voltaics, harness, etc"),
                                                    ],
                            className="mb-3",
                            )
                    ]),
                    dbc.Form([
                        html.Div([
                            dbc.Label('Facesheet Thickness (in)', html_for=f'{APP_ID}_tfs_input'),
                            dbc.Input(id=f'{APP_ID}_tfs_input', type="number", min=0.01, value=0.030, step='any'),
                            dbc.FormText("per skin, 2 skins per panel")
                                                    ],
                            className="mb-3",
                            )
                    ])
                ]),
                dbc.Col([
                    dbc.Form([
                        html.Div([
                            dbc.Label('Hinge Effective Rotational Stiffness (in-lbs/rad)', html_for=f'{APP_ID}_k_hinge_input'),
                            dbc.Input(id=f'{APP_ID}_k_hinge_input', type="number", min=1000, value=10e3, step=1000),
                            dbc.FormText("Stiffness about pin axis"),
                            ],
                            className="mb-3",
                            )
                    ]),
                    dbc.Form([
                        html.Div([
                            dbc.Label('Panel Separation (in)', html_for=f'{APP_ID}_dx_panel_input'),
                            dbc.Input(id=f'{APP_ID}_dx_panel_input', type="number", min=0., value=2., step='any'),   
                        ],
                        className="mb-3",
                        )
                    ])

               
                ]),
                dbc.Col([
                    dbc.Form([
                        html.Div([
                            dbc.Label('Displacement Scale Factor', html_for=f'{APP_ID}_scale_input'),
                            dbc.Input(id=f'{APP_ID}_scale_input', type="number", value=5.),  
                        ],
                        className="mb-3",
                        )
                    ]),
                    dbc.Form([
                        html.Div([
                            dbc.Label('Show Edges', html_for=f'{APP_ID}_show_edges_switch'),
                            dbc.Switch(
                                id=f'{APP_ID}_show_edges_switch',
                                label='Show Edges',
                                value=True,
                            ), 
                        ],
                        className="mb-3",
                        )
                    ])
                ]),
            ],
            className="mt-3",
            )
        ]
    ),
    dbc.Row([
        dbc.Col([
            html.Progress(id=f'{APP_ID}_progress_bar', value='0', max='10', style={'visibility':"hidden"}),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(
                style={"width": "100%", "height": "60vh"},
                children=[
                    dash_vtk.View(
                        id=f'{APP_ID}_vtk_view',
                        children=dash_vtk.GeometryRepresentation(
                            id=f'{APP_ID}_geom_rep_mesh',
                            children=[],
                            property={"edgeVisibility": True, "opacity": 1, "pointSize": 20, "lineWidth": 2},
                            colorMapPreset="Plasma (matplotlib)",
                        ),
                    ),
                ]
            )
        ],
            width=10
        ),
        dbc.Col([
            dcc.Graph(
                id=f'{APP_ID}_results_colorbar_graph',
                style={"width": "100%", "height": "60vh"},
            ),
 
        ],
            width=2
        )
    ],
    className="g-0"
    ),
    dbc.Row([
        dbc.Col(
            id=f'{APP_ID}_mem_sene_dt_div',
        ),
    ],
        className="mt-3"
    )
])





@app.callback(
    Output(f'{APP_ID}_options_collapse', "is_open"),
    [Input(f'{APP_ID}_options_collapse_button', "n_clicks")],
    [State(f'{APP_ID}_options_collapse', "is_open")],
)
def toggle_options_collapse(n, is_open):
    if n:
        return not is_open
    return is_open



@app.long_callback(
    Output(f'{APP_ID}_geom_rep_mesh', 'children'),
    Output(f'{APP_ID}_geom_rep_mesh', 'colorDataRange'),
    Output(f'{APP_ID}_results_colorbar_graph', 'figure'),
    Output(f'{APP_ID}_mem_sene_dt_div', 'children'),
    Input(f'{APP_ID}_solve_button', 'n_clicks'),
    State(f'{APP_ID}_n_panels_input', 'value'),
    State(f'{APP_ID}_w_panel_input','value'),
    State(f'{APP_ID}_h_panel_input', 'value'),
    State(f'{APP_ID}_nsm_input', 'value'),
    State(f'{APP_ID}_tfs_input', 'value'),
    State(f'{APP_ID}_k_hinge_input', 'value'),
    State(f'{APP_ID}_dx_panel_input', 'value'),
    State(f'{APP_ID}_tc_input', 'value'),
    State(f'{APP_ID}_scale_input', 'value'),
    prevent_initial_call=True,
    running=[
        (Output(f'{APP_ID}_solve_button', "disabled"), True, False),
        (
            Output(f'{APP_ID}_progress_bar', "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        ),
    ],
    progress=[Output(f'{APP_ID}_progress_bar', "value"), Output(f'{APP_ID}_progress_bar', "max")],
)
def dash_vtk_update_grid(set_progress, n_clicks, n_panels, panel_w, panel_h, nsm_pv, tfs, k_hinge, dx_panel, tc, scale):

    if any([v is None for v in [n_clicks, n_panels, panel_w, panel_h, nsm_pv, tfs, k_hinge, dx_panel, tc, scale]]):
        raise PreventUpdate


    set_progress((str(1), str(10)))
    res, ugrid, df_sene_mem = make_pv_array(panel_w, panel_h, n_panels=n_panels, nsm_pv=nsm_pv, tc=tc, tfs=tfs, k_hinge=k_hinge, dx_panel=dx_panel, prog=set_progress)
    set_progress((str(8), str(10)))
    f0 = df_sene_mem['freq'].iloc[0]
    dof ='uZ'
    ugrid = plot_nodal_disp(ugrid, res, scale=scale, dof=dof)

    view_max = ugrid[dof].max()
    view_min = ugrid[dof].min()
    rng = [view_min, view_max]

    fig_cb = make_colorbar(f'Z Displacement (in)<Br>Freq: {f0:0.3f}', rng)

    mesh_state = to_mesh_state(ugrid, field_to_keep=dof)
    set_progress((str(9), str(10)))

    dt = dash_table.DataTable(
        data=df_sene_mem.to_dict('records'),
        sort_action='native',
        columns=[
            {'name': 'Mode', 'id': 'mode', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.decimal_integer)},
            {'name': 'Frequency (Hz)', 'id': 'freq', 'type': 'numeric', 'format': Format(precision=3, scheme=Scheme.decimal)},
            {'name': 'TX', 'id': 'TX', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.percentage)},
            {'name': 'TY', 'id': 'TY', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.percentage)},
            {'name': 'TZ', 'id': 'TZ', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.percentage)},
            {'name': 'RX', 'id': 'RX', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.percentage)},
            {'name': 'RY', 'id': 'RY', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.percentage)},
            {'name': 'RZ', 'id': 'RZ', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.percentage)},
            {'name': 'SENE Hinges', 'id': 'joint', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.percentage)},
            {'name': 'SENE Panels', 'id': 'shell', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.percentage)},
        ],
        style_data_conditional=(
            data_bars(df_sene_mem, 'TX', clr='rgb(11, 94, 215)') +
            data_bars(df_sene_mem, 'TY', clr='rgb(11, 94, 215)') +
            data_bars(df_sene_mem, 'TZ', clr='rgb(11, 94, 215)') +
            data_bars(df_sene_mem, 'RX', clr='rgb(11, 94, 215)') +
            data_bars(df_sene_mem, 'RY', clr='rgb(11, 94, 215)') +
            data_bars(df_sene_mem, 'RZ', clr='rgb(11, 94, 215)') +
            data_bars(df_sene_mem, 'joint', clr='rgb(11, 94, 215)') +
            data_bars(df_sene_mem, 'shell', clr='rgb(11, 94, 215)')
        ),
    )

    set_progress((str(10), str(10)))
    return dash_vtk.Mesh(state=mesh_state), rng, fig_cb, dt  




@app.callback(
    Output(f'{APP_ID}_geom_rep_mesh', 'property'),
    Input(f'{APP_ID}_show_edges_switch', 'value')
)
def pyansys_mapdl_show_edges(show_edges):
    if show_edges is None:
        raise PreventUpdate
    return {"edgeVisibility": show_edges, "opacity": 1, "pointSize": 20, "lineWidth": 2}


if __name__ == '__main__':
    app.run_server(debug=True)
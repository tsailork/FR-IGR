import json
import urllib.request
import os

dot_code = """
digraph SolverFlow {
    fontname="Inter, sans-serif";
    node [fontname="Inter, sans-serif", shape=box, style="filled,rounded", penwidth=2, margin="0.2,0.1"];
    edge [fontname="Inter, sans-serif", color="#94a3b8", fontcolor="#f8fafc", penwidth=2];
    bgcolor="transparent";
    rankdir=TB;
    nodesep=0.4;
    ranksep=0.6;

    subgraph cluster_pre {
        label="1. Pre-Iterations";
        labeljust="l";
        fontcolor="#0f766e";
        color="#134e4a";
        style="rounded,dashed";
        
        node [fillcolor="#0f766e", color="#2dd4bf", fontcolor="#f0fdfa"];
        
        load_params [label="Load Parameters\\n(Parameters)"];
        setup_solver [label="Setup Solver\\n(Solver::Solver)"];
        init_cells [label="Cell Decomposition\\n(initialize_cells)"];
        bfs_wall [label="BFS Wall Tracking\\n(flag_refinement_coarsening)"];
        connectivity [label="Tree Connectivity\\n(setup_cell_connectivity)"];
        init_state [label="Initialize State\\n(IC::apply/Restart)"];
        init_igr [label="Initial IGR Setup\\n(compute_sensor)"];
        
        { rank=same; load_params; setup_solver; init_cells; bfs_wall; connectivity; init_state; init_igr; }
        load_params -> setup_solver -> init_cells -> bfs_wall -> connectivity -> init_state -> init_igr;
    }

    subgraph cluster_iter {
        label="2. During Iterations (Time-Stepping Loop)";
        labeljust="l";
        fontcolor="#4338ca";
        color="#312e81";
        style="rounded,dashed";
        
        node [fillcolor="#4338ca", color="#818cf8", fontcolor="#eef2ff"];
        stop_check [label="Check STOP File\\n(Exists? -> break)"];
        dt [label="Compute Time Step\\n(Solver::compute_dt)"];
        
        subgraph cluster_rk3 {
            label="SSP-RK3 Stage (Solver::step_rk3)";
            labeljust="l";
            fontcolor="#0284c7";
            color="#0c4a6e";
            style="rounded,solid";
            penwidth=2;
            
            node [fillcolor="#0369a1", color="#38bdf8", fontcolor="#f0f9ff"];
            ib [label="Update IB\\n(update_mask)"];
            igr [label="Calculate IGR\\n(entropic_pressure)"];
            fluxes [label="Calculate Fluxes\\n(euler/viscous)"];
            update_state [label="Update State\\n(State + dt*dU)"];
            limiters [label="Apply Limiters\\n(positivity)"];
        }
        
        diag [label="Update Diagnostics\\n(Diagnostics::update)"];
        output [label="Checkpoints & Plots\\n(VTKWriter::write)"];
        
        { rank=same; stop_check; dt; ib; igr; fluxes; update_state; limiters; diag; output; }
        stop_check -> dt -> ib -> igr -> fluxes -> update_state -> limiters -> diag -> output;
        
        output -> stop_check [label=" t < T_FINAL", style=dashed, constraint=false];
    }

    subgraph cluster_post {
        label="3. Post-Iteration";
        labeljust="l";
        fontcolor="#be123c";
        color="#881337";
        style="rounded,dashed";
        
        node [fillcolor="#be123c", color="#fb7185", fontcolor="#fff1f2"];
        finish [label="Simulation Complete\\n(Clean exit)"];
        
        { rank=same; finish; }
    }

    init_igr -> stop_check;
    stop_check -> finish [label=" STOP exists", color="#ef4444", fontcolor="#ef4444"];
    output -> finish [label=" t >= T_FINAL"];
    
    /* Invisible edges to force left-alignment of clusters */
    load_params -> stop_check -> finish [style=invis, weight=100];
}
"""

url = 'https://quickchart.io/graphviz'
data = json.dumps({'graph': dot_code}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as response:
        svg_data = response.read().decode('utf-8')
        
    os.makedirs(os.path.join(os.path.dirname(__file__), '../assets'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), '../assets/solver_flow_diagram.svg')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(svg_data)
    print(f"Successfully generated {out_path}")
except Exception as e:
    print(f"Error generating diagram: {e}")

// User Interface, forms synchronization, and modal controllers

import { state } from './state.js';
import { saveWebcontour } from './api.js';
import { draw } from './canvas_renderer.js';

// Populate UI form inputs from inputs state
export function populateFormFields() {
    const setVal = (id, section, key, isCheckbox = false) => {
        const el = document.getElementById(id);
        if (!el) return;
        const val = state.inputs[section] ? state.inputs[section][key] : "";
        if (isCheckbox) {
            el.checked = val === "true" || val === "1";
        } else {
            el.value = val;
        }
    };

    // Physics
    setVal("phy-gamma", "Physics", "GAMMA");
    setVal("phy-ic-type", "Physics", "IC_TYPE");
    setVal("phy-rho-inf", "Physics", "RHO_INF");
    setVal("phy-u-inf", "Physics", "U_INF");
    setVal("phy-v-inf", "Physics", "V_INF");
    setVal("phy-p-inf", "Physics", "P_INF");

    // Solver
    setVal("sol-p-deg", "Solver", "P_DEG");
    setVal("sol-cfl", "Solver", "CFL");
    setVal("sol-t-final", "Solver", "T_FINAL");
    setVal("sol-num-threads", "Solver", "NUM_THREADS");

    // Navier-Stokes
    setVal("ns-enable", "NavierStokes", "ENABLE_NS", true);
    setVal("ns-re", "NavierStokes", "RE");
    setVal("ns-pr", "NavierStokes", "PR");
    setVal("ns-mach-ref", "NavierStokes", "MACH_REF");
    setVal("ns-br2-eta", "NavierStokes", "NS_BR2_ETA");

    // IGR
    setVal("igr-enable", "Regularization", "ENABLE_IGR", true);
    setVal("igr-alpha", "Regularization", "ALPHA_SCALE");
    setVal("igr-grad-type", "Regularization", "IGR_GRADIENT_TYPE");
    setVal("igr-type", "Regularization", "IGR_TYPE");
    setVal("igr-tau-r", "Regularization", "IGR_TAU_R");
    setVal("igr-sub-iters", "Regularization", "IGR_SUB_ITERS");

    // Limiters
    setVal("lim-pos-enable", "Stabilization", "ENABLE_POS_LIMITER", true);
    setVal("lim-pos-eps", "Stabilization", "POS_LIMITER_EPS");
    setVal("lim-entropy-enable", "Stabilization", "ENABLE_ENTROPY_LIMITER", true);

    // Immersed Boundary
    setVal("ib-enable", "ImmersedBoundary", "ENABLE_IB", true);
    setVal("ib-method", "ImmersedBoundary", "IB_METHOD");
    setVal("ib-shape", "ImmersedBoundary", "IB_SHAPE");
    setVal("ib-sharp", "ImmersedBoundary", "IB_SHARP", true);
    setVal("ib-smooth-width", "ImmersedBoundary", "IB_SMOOTH_WIDTH");
    setVal("ib-center-x", "ImmersedBoundary", "IB_CENTER_X");
    setVal("ib-center-y", "ImmersedBoundary", "IB_CENTER_Y");
    setVal("ib-radius", "ImmersedBoundary", "IB_RADIUS");
    setVal("ib-naca-code", "ImmersedBoundary", "IB_NACA_CODE");
    setVal("ib-aoa", "ImmersedBoundary", "IB_AOA");
    setVal("ib-chord", "ImmersedBoundary", "IB_RADIUS"); // chord is stored as IB_RADIUS in parameters
    setVal("ib-thermal", "ImmersedBoundary", "IB_THERMAL_TYPE");
    setVal("ib-temperature", "ImmersedBoundary", "IB_TEMPERATURE");

    // SBM Specific Options
    setVal("ib-sbm-diag", "ImmersedBoundary", "ENABLE_SBM_DIAGNOSTICS", true);
    setVal("ib-enable-3c", "ImmersedBoundary", "ENABLE_IB_3C", true);
    setVal("ib-dl-scale", "ImmersedBoundary", "IB_DL_SCALE");
    setVal("ib-l-scale", "ImmersedBoundary", "IB_L_SCALE");

    // VPM Specific Options
    setVal("ib-pen-eta", "ImmersedBoundary", "IB_PENALIZATION_ETA");
    setVal("ib-vel-x", "ImmersedBoundary", "IB_VELOCITY_X");
    setVal("ib-vel-y", "ImmersedBoundary", "IB_VELOCITY_Y");

    // MULTI Options
    setVal("ib-q-time-map", "ImmersedBoundary", "IB_Q_TIME_MAP");

    // Toggle fields based on selections
    toggleIBShapeFields();
    toggleIBMethodFields();

    // Parse quads
    state.quads = [];
    if (state.inputs["ImmersedBoundary"]) {
        const numQuads = parseInt(state.inputs["ImmersedBoundary"]["IB_NUM_QUADS"] || 0);
        for (let i = 0; i < numQuads; ++i) {
            const qVal = state.inputs["ImmersedBoundary"][`IB_QUAD_${i}`];
            if (qVal) state.quads.push(qVal);
        }
    }
    renderQuadsTable();

    // Parse curves/polys
    state.polys = [];
    if (state.inputs["ImmersedBoundary"]) {
        const numPolys = parseInt(state.inputs["ImmersedBoundary"]["IB_NUM_POLYS"] || 0);
        for (let i = 0; i < numPolys; ++i) {
            const pVal = state.inputs["ImmersedBoundary"][`IB_POLY_${i}`];
            if (pVal) {
                const parts = pVal.trim().split(/\s+/);
                if (parts.length >= 8) {
                    state.polys.push({
                        dir: parts[0],
                        a0: parseFloat(parts[1]) || 0,
                        b0: parseFloat(parts[2]) || 0,
                        c0: parseFloat(parts[3]) || 0,
                        a1: parseFloat(parts[4]) || 0,
                        b1: parseFloat(parts[5]) || 0,
                        c1: parseFloat(parts[6]) || 0,
                        side: parts[7]
                    });
                }
            }
        }
    }
    renderPolysTable();

    // I/O
    setVal("io-output-interval", "IO", "OUTPUT_INTERVAL");
    setVal("io-restart-interval", "IO", "RESTART_INTERVAL");
    setVal("io-restart-file", "IO", "RESTART_FILE");
    setVal("io-restart-time", "IO", "RESTART_TIME");

    // Probes
    parseProbesFromConfig();
    renderProbesTable();
}

export function toggleIBShapeFields() {
    const shape = document.getElementById("ib-shape").value;
    const circleOpts = document.getElementById("field-circle-opts");
    const nacaOpts = document.getElementById("field-naca-opts");
    const multiOpts = document.getElementById("field-multi-opts");
    if (shape === "CIRCLE") {
        circleOpts.style.display = "block";
        nacaOpts.style.display = "none";
        multiOpts.style.display = "none";
    } else if (shape === "NACA") {
        circleOpts.style.display = "block"; // Need center coords as leading edge
        nacaOpts.style.display = "block";
        multiOpts.style.display = "none";
    } else if (shape === "MULTI") {
        circleOpts.style.display = "none";
        nacaOpts.style.display = "none";
        multiOpts.style.display = "block";
    }
}

export function toggleIBMethodFields() {
    const method = document.getElementById("ib-method").value;
    const sbmOpts = document.getElementById("field-sbm-opts");
    const vpmOpts = document.getElementById("field-vpm-opts");
    if (method === "SBM") {
        sbmOpts.style.display = "block";
        vpmOpts.style.display = "none";
    } else {
        sbmOpts.style.display = "none";
        vpmOpts.style.display = "block";
    }
}

export function renderQuadsTable() {
    const container = document.getElementById("multi-quads-list");
    if (!container) return;
    container.innerHTML = "";
    
    if (state.quads.length === 0) {
        container.innerHTML = `<p style="font-size: 0.8rem; color: #888; margin: 5px 0;">No quadrilaterals defined.</p>`;
        return;
    }
    
    state.quads.forEach((q, idx) => {
        const div = document.createElement("div");
        div.className = "quad-item-row";
        div.style.display = "flex";
        div.style.alignItems = "center";
        div.style.gap = "5px";
        div.style.marginBottom = "5px";
        
        const label = document.createElement("span");
        label.textContent = `Q${idx}:`;
        label.style.fontSize = "0.75rem";
        label.style.fontWeight = "bold";
        label.style.minWidth = "30px";
        
        const input = document.createElement("input");
        input.type = "text";
        input.value = q;
        input.className = "input-sm quad-coords-input";
        input.style.flex = "1";
        input.style.fontSize = "0.75rem";
        input.placeholder = "x1 y1 x2 y2 x3 y3 x4 y4";
        input.addEventListener("input", (e) => {
            state.quads[idx] = e.target.value;
        });
        
        const btnDel = document.createElement("button");
        btnDel.type = "button";
        btnDel.className = "action-btn-sm delete-btn";
        btnDel.innerHTML = "&times;";
        btnDel.style.backgroundColor = "#e04e4e";
        btnDel.style.color = "white";
        btnDel.style.border = "none";
        btnDel.style.borderRadius = "3px";
        btnDel.style.cursor = "pointer";
        btnDel.style.padding = "2px 6px";
        btnDel.addEventListener("click", () => {
            state.quads.splice(idx, 1);
            renderQuadsTable();
        });
        
        div.appendChild(label);
        div.appendChild(input);
        div.appendChild(btnDel);
        container.appendChild(div);
    });
}

export function renderPolysTable() {
    const tbody = document.getElementById("multi-polys-body");
    if (!tbody) return;
    tbody.innerHTML = "";
    
    if (state.polys.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #888; font-size: 0.8rem;">No curves defined.</td></tr>`;
        return;
    }
    
    state.polys.forEach((p, idx) => {
        const tr = document.createElement("tr");
        
        const tdDir = document.createElement("td");
        const selectDir = document.createElement("select");
        selectDir.className = "select-sm";
        selectDir.style.fontSize = "0.75rem";
        selectDir.innerHTML = `<option value="Y">Y</option><option value="X">X</option>`;
        selectDir.value = p.dir;
        
        const tdSide = document.createElement("td");
        const selectSide = document.createElement("select");
        selectSide.className = "select-sm";
        selectSide.style.fontSize = "0.75rem";
        
        const updateSideOptions = (sel, dir) => {
            if (dir === "Y") {
                sel.innerHTML = `<option value="ABOVE">ABOVE</option><option value="BELOW">BELOW</option>`;
            } else {
                sel.innerHTML = `<option value="LEFT">LEFT</option><option value="RIGHT">RIGHT</option>`;
            }
        };
        
        selectDir.addEventListener("change", (e) => {
            p.dir = e.target.value;
            updateSideOptions(selectSide, p.dir);
            p.side = selectSide.value;
        });
        tdDir.appendChild(selectDir);
        
        const tdCoeffs0 = document.createElement("td");
        tdCoeffs0.style.display = "flex";
        tdCoeffs0.style.gap = "2px";
        const createCoeffInput = (val, key) => {
            const input = document.createElement("input");
            input.type = "number";
            input.step = "any";
            input.value = val;
            input.className = "input-sm";
            input.style.width = "40px";
            input.style.fontSize = "0.7rem";
            input.addEventListener("input", (e) => {
                p[key] = parseFloat(e.target.value) || 0;
            });
            return input;
        };
        tdCoeffs0.appendChild(createCoeffInput(p.a0, "a0"));
        tdCoeffs0.appendChild(createCoeffInput(p.b0, "b0"));
        tdCoeffs0.appendChild(createCoeffInput(p.c0, "c0"));
        
        const tdCoeffs1 = document.createElement("td");
        tdCoeffs1.style.display = "flex";
        tdCoeffs1.style.gap = "2px";
        tdCoeffs1.appendChild(createCoeffInput(p.a1, "a1"));
        tdCoeffs1.appendChild(createCoeffInput(p.b1, "b1"));
        tdCoeffs1.appendChild(createCoeffInput(p.c1, "c1"));
        
        updateSideOptions(selectSide, p.dir);
        selectSide.value = p.side;
        selectSide.addEventListener("change", (e) => {
            p.side = e.target.value;
        });
        tdSide.appendChild(selectSide);
        
        const tdAct = document.createElement("td");
        const btnDel = document.createElement("button");
        btnDel.type = "button";
        btnDel.className = "action-btn-sm";
        btnDel.innerHTML = "&times;";
        btnDel.style.backgroundColor = "#e04e4e";
        btnDel.style.color = "white";
        btnDel.addEventListener("click", () => {
            state.polys.splice(idx, 1);
            renderPolysTable();
        });
        tdAct.appendChild(btnDel);
        
        tr.appendChild(tdDir);
        tr.appendChild(tdCoeffs0);
        tr.appendChild(tdCoeffs1);
        tr.appendChild(tdSide);
        tr.appendChild(tdAct);
        
        tbody.appendChild(tr);
    });
}

// Parse probes from inputs configuration
export function parseProbesFromConfig() {
    state.probes = [];
    if (!state.inputs["Probes"]) return;
    for (const [key, val] of Object.entries(state.inputs["Probes"])) {
        if (key.startsWith("PROBE_")) {
            const parts = val.split(",");
            if (parts.length >= 3) {
                state.probes.push({
                    name: key,
                    x: parseFloat(parts[0].trim()),
                    y: parseFloat(parts[1].trim()),
                    variable: parts[2].trim()
                });
            }
        }
    }
}

// Redraw point probes table
export function renderProbesTable() {
    const tbody = document.getElementById("probes-list-body");
    tbody.innerHTML = "";
    state.probes.forEach((probe, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${probe.name}</td>
            <td>${probe.x.toFixed(4)}</td>
            <td>${probe.y.toFixed(4)}</td>
            <td>${probe.variable}</td>
            <td><button class="delete-btn" onclick="deleteProbe(${idx})">Delete</button></td>
        `;
        tbody.appendChild(tr);
    });
}

export function deleteProbe(idx) {
    state.probes.splice(idx, 1);
    renderProbesTable();
    updateProbesInConfig();
    draw();
}
// Expose to window globally to satisfy HTML inline onclick handlers
window.deleteProbe = deleteProbe;

export function updateProbesInConfig() {
    if (!state.inputs["Probes"]) state.inputs["Probes"] = {};
    
    // Clear old PROBE_ values
    for (const key of Object.keys(state.inputs["Probes"])) {
        if (key.startsWith("PROBE_")) {
            delete state.inputs["Probes"][key];
        }
    }
    
    // Write new values
    state.probes.forEach((probe, idx) => {
        state.inputs["Probes"][`PROBE_${idx+1}`] = `${probe.x}, ${probe.y}, ${probe.variable}`;
    });
}

// Compile forms back into inputs state
export function collectFormFields() {
    const getVal = (id, section, key, isNumber = false, isCheckbox = false) => {
        const el = document.getElementById(id);
        if (!el) return;
        if (!state.inputs[section]) state.inputs[section] = {};
        if (isCheckbox) {
            state.inputs[section][key] = el.checked ? "true" : "false";
        } else {
            state.inputs[section][key] = el.value;
        }
    };

    // Physics
    getVal("phy-gamma", "Physics", "GAMMA");
    getVal("phy-ic-type", "Physics", "IC_TYPE");
    getVal("phy-rho-inf", "Physics", "RHO_INF");
    getVal("phy-u-inf", "Physics", "U_INF");
    getVal("phy-v-inf", "Physics", "V_INF");
    getVal("phy-p-inf", "Physics", "P_INF");

    // Solver
    getVal("sol-p-deg", "Solver", "P_DEG");
    getVal("sol-cfl", "Solver", "CFL");
    getVal("sol-t-final", "Solver", "T_FINAL");
    getVal("sol-num-threads", "Solver", "NUM_THREADS");

    // Navier-Stokes
    getVal("ns-enable", "NavierStokes", "ENABLE_NS", false, true);
    getVal("ns-re", "NavierStokes", "RE");
    getVal("ns-pr", "NavierStokes", "PR");
    getVal("ns-mach-ref", "NavierStokes", "MACH_REF");
    getVal("ns-br2-eta", "NavierStokes", "NS_BR2_ETA");

    // IGR
    getVal("igr-enable", "Regularization", "ENABLE_IGR", false, true);
    getVal("igr-alpha", "Regularization", "ALPHA_SCALE");
    getVal("igr-grad-type", "Regularization", "IGR_GRADIENT_TYPE");
    getVal("igr-type", "Regularization", "IGR_TYPE");
    getVal("igr-tau-r", "Regularization", "IGR_TAU_R");
    getVal("igr-sub-iters", "Regularization", "IGR_SUB_ITERS");

    // Limiters
    getVal("lim-pos-enable", "Stabilization", "ENABLE_POS_LIMITER", false, true);
    getVal("lim-pos-eps", "Stabilization", "POS_LIMITER_EPS");
    getVal("lim-entropy-enable", "Stabilization", "ENABLE_ENTROPY_LIMITER", false, true);

    // Immersed Boundary
    getVal("ib-enable", "ImmersedBoundary", "ENABLE_IB", false, true);
    getVal("ib-method", "ImmersedBoundary", "IB_METHOD");
    const shape = document.getElementById("ib-shape").value;
    state.inputs["ImmersedBoundary"]["IB_SHAPE"] = shape;
    getVal("ib-sharp", "ImmersedBoundary", "IB_SHARP", false, true);
    getVal("ib-smooth-width", "ImmersedBoundary", "IB_SMOOTH_WIDTH");
    getVal("ib-center-x", "ImmersedBoundary", "IB_CENTER_X");
    getVal("ib-center-y", "ImmersedBoundary", "IB_CENTER_Y");
    
    if (shape === "CIRCLE") {
        getVal("ib-radius", "ImmersedBoundary", "IB_RADIUS");
    } else if (shape === "NACA") {
        getVal("ib-naca-code", "ImmersedBoundary", "IB_NACA_CODE");
        getVal("ib-aoa", "ImmersedBoundary", "IB_AOA");
        getVal("ib-chord", "ImmersedBoundary", "IB_RADIUS"); // chord is stored as IB_RADIUS in parameters
    } else if (shape === "MULTI") {
        getVal("ib-q-time-map", "ImmersedBoundary", "IB_Q_TIME_MAP");
        
        // Clear all previous IB_QUAD_i and IB_POLY_i keys
        if (state.inputs["ImmersedBoundary"]) {
            for (const key of Object.keys(state.inputs["ImmersedBoundary"])) {
                if (key.startsWith("IB_QUAD_") || key.startsWith("IB_POLY_")) {
                    delete state.inputs["ImmersedBoundary"][key];
                }
            }
        } else {
            state.inputs["ImmersedBoundary"] = {};
        }
        
        // Write quads
        state.inputs["ImmersedBoundary"]["IB_NUM_QUADS"] = state.quads.length.toString();
        state.quads.forEach((q, idx) => {
            state.inputs["ImmersedBoundary"][`IB_QUAD_${idx}`] = q;
        });
        
        // Write polys
        state.inputs["ImmersedBoundary"]["IB_NUM_POLYS"] = state.polys.length.toString();
        state.polys.forEach((p, idx) => {
            state.inputs["ImmersedBoundary"][`IB_POLY_${idx}`] = `${p.dir} ${p.a0} ${p.b0} ${p.c0} ${p.a1} ${p.b1} ${p.c1} ${p.side}`;
        });
    }
    
    getVal("ib-thermal", "ImmersedBoundary", "IB_THERMAL_TYPE");
    getVal("ib-temperature", "ImmersedBoundary", "IB_TEMPERATURE");

    // SBM Specific Options
    getVal("ib-sbm-diag", "ImmersedBoundary", "ENABLE_SBM_DIAGNOSTICS", false, true);
    getVal("ib-enable-3c", "ImmersedBoundary", "ENABLE_IB_3C", false, true);
    getVal("ib-dl-scale", "ImmersedBoundary", "IB_DL_SCALE");
    getVal("ib-l-scale", "ImmersedBoundary", "IB_L_SCALE");

    // VPM Specific Options
    getVal("ib-pen-eta", "ImmersedBoundary", "IB_PENALIZATION_ETA");
    getVal("ib-vel-x", "ImmersedBoundary", "IB_VELOCITY_X");
    getVal("ib-vel-y", "ImmersedBoundary", "IB_VELOCITY_Y");

    // I/O
    getVal("io-output-interval", "IO", "OUTPUT_INTERVAL");
    getVal("io-restart-interval", "IO", "RESTART_INTERVAL");
    getVal("io-restart-file", "IO", "RESTART_FILE");
    getVal("io-restart-time", "IO", "RESTART_TIME");

    updateProbesInConfig();
}

export function updateStatusUI(running) {
    const badge = document.getElementById("status-badge");
    const stopBtn = document.getElementById("btn-stop");
    const actionBtns = ["btn-run", "btn-restart", "btn-clean", "btn-clean-only"];

    if (running) {
        badge.textContent = "Simulation Running";
        badge.className = "status-indicator running";
        stopBtn.disabled = false;
        stopBtn.classList.remove("disabled");
        actionBtns.forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.disabled = true;
        });
    } else {
        badge.textContent = "Solver Idle";
        badge.className = "status-indicator idle";
        stopBtn.disabled = true;
        stopBtn.classList.add("disabled");
        actionBtns.forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.disabled = false;
        });
    }
}

export function appendLogLine(line) {
    const term = document.getElementById("terminal-log");
    term.textContent += "\n" + line;
    term.scrollTop = term.scrollHeight;
}

export function openBCEditorModal() {
    const modal = document.getElementById("modal-edit-bc");
    const faceName = state.editingBC.face === "L" ? "Left" : state.editingBC.face === "R" ? "Right" : state.editingBC.face === "B" ? "Bottom" : "Top";
    document.getElementById("bc-title-desc").textContent = `Block ${state.editingBC.blockId} Face ${faceName} (Current: ${state.editingBC.currentBC})`;
    
    const select = document.getElementById("bc-type-select");
    const stateGroup = document.getElementById("bc-opts-state");
    const connectGroup = document.getElementById("bc-opts-connect");
    
    const bcVal = state.editingBC.currentBC;
    if (bcVal.includes(":") && !isNaN(parseInt(bcVal.split(":")[0]))) {
        select.value = "BLOCK_CONNECT";
        connectGroup.style.display = "block";
        stateGroup.style.display = "none";
        
        const parts = bcVal.split(":");
        document.getElementById("bc-connect-block").value = parts[0];
        document.getElementById("bc-connect-face").value = parts[1];
    } else {
        if (bcVal.startsWith("CHARACTERISTIC")) {
            select.value = "CHARACTERISTIC";
            stateGroup.style.display = "block";
            connectGroup.style.display = "none";
            document.getElementById("bc-state-input").value = bcVal.substring(15);
        } else if (bcVal.startsWith("INFLOW_SUPERSONIC")) {
            select.value = "INFLOW_SUPERSONIC";
            stateGroup.style.display = "block";
            connectGroup.style.display = "none";
            document.getElementById("bc-state-input").value = bcVal.substring(18);
        } else if (bcVal.startsWith("STATIC_PRESSURE")) {
            select.value = "STATIC_PRESSURE";
            stateGroup.style.display = "block";
            connectGroup.style.display = "none";
            document.getElementById("bc-state-input").value = bcVal.substring(16);
        } else {
            select.value = bcVal; // WALL, WALL_NOSLIP, TRANSMISSIVE
            stateGroup.style.display = "none";
            connectGroup.style.display = "none";
        }
    }
    
    modal.style.display = "flex";
}

export function openBlockEditorModal() {
    const modal = document.getElementById("modal-edit-block");
    document.getElementById("eb-nx").value = state.editingBlock ? state.editingBlock.nx : 100;
    document.getElementById("eb-ny").value = state.editingBlock ? state.editingBlock.ny : 100;
    document.getElementById("eb-xmin").value = state.editingBlock ? state.editingBlock.xMin : 0.0;
    document.getElementById("eb-xmax").value = state.editingBlock ? state.editingBlock.xMax : 1.0;
    document.getElementById("eb-ymin").value = state.editingBlock ? state.editingBlock.yMin : 0.0;
    document.getElementById("eb-ymax").value = state.editingBlock ? state.editingBlock.yMax : 1.0;
    
    const titleEl = document.getElementById("eb-modal-title");
    if (titleEl) {
        titleEl.textContent = state.editingBlock ? "Edit Block Settings" : "Add New Block";
    }
    const deleteBtn = document.getElementById("btn-eb-delete");
    if (deleteBtn) {
        deleteBtn.style.display = state.editingBlock ? "inline-block" : "none";
    }
    modal.style.display = "flex";
}

export function openContourSettingsModal(varName) {
    const modal = document.getElementById("modal-contour-settings");
    if (!modal) return;
    
    const settings = state.contourSettings[varName] || {
        cmap: "viridis",
        levels: 50,
        range_mode: "auto",
        vmin: 0.0,
        vmax: 1.0,
        show_grid: false
    };
    
    document.getElementById("contour-cmap").value = settings.cmap;
    document.getElementById("contour-levels").value = settings.levels;
    
    const isAuto = settings.range_mode === "auto";
    document.getElementById("contour-range-auto").checked = isAuto;
    document.getElementById("contour-manual-range-group").style.display = isAuto ? "none" : "flex";
    
    document.getElementById("contour-vmin").value = settings.vmin;
    document.getElementById("contour-vmax").value = settings.vmax;
    document.getElementById("contour-show-grid").checked = settings.show_grid;
    
    modal.style.display = "flex";
}

export async function saveContourSettings() {
    const varName = state.visualizerMode.replace("contour_", "");
    const modal = document.getElementById("modal-contour-settings");
    
    const settings = {
        cmap: document.getElementById("contour-cmap").value,
        levels: parseInt(document.getElementById("contour-levels").value) || 50,
        range_mode: document.getElementById("contour-range-auto").checked ? "auto" : "manual",
        vmin: parseFloat(document.getElementById("contour-vmin").value) || 0.0,
        vmax: parseFloat(document.getElementById("contour-vmax").value) || 1.0,
        show_grid: document.getElementById("contour-show-grid").checked
    };
    
    state.contourSettings[varName] = settings;
    modal.style.display = "none";
    
    try {
        const data = await saveWebcontour(varName, settings);
        if (data.status === "success") {
            draw();
        } else {
            console.error("Failed to save .webcontour backend side:", data.message);
        }
    } catch (err) {
        console.error("Failed to POST webcontour settings:", err);
    }
}

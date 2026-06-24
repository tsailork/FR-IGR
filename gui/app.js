// FR-IGR GUI Application Frontend Logic

let inputs = {};
let domain = {};
let blocks = [];
let probes = [];
let quads = [];
let polys = [];
let statusInterval = null;
let activeTab = "physics";
let visualizerMode = "grid"; // 'grid', 'contour_rho', 'contour_press', 'contour_mach', 'contour_sigma', 'contour_phi'
let isRunning = false;
let isProbeModeActive = false;

// Playback state
let playbackActiveVtm = null;
let playbackTimesteps = [];
let playbackIndex = -1;
let playbackInterval = null;
let contourSettings = {}; // cached settings by varName

// Canvas state
const canvas = document.getElementById("grid-canvas");
const ctx = canvas.getContext("2d");
let scale = 1.0;
let offsetX = 0.0;
let offsetY = 0.0;
let bounds = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
let lastMousePos = { x: 0, y: 0 };
let hoveredElement = null; // { type: 'boundary'/'block', blockId, face/details }

// Chart histories
let residualHistory = [];
let probeHistory = [];

// Initialize Page
window.addEventListener("DOMContentLoaded", () => {
    fetchConfig();
    startStatusPolling();
    setupEventListeners();
    resizeCanvas();
});

window.addEventListener("resize", () => {
    resizeCanvas();
    draw();
});

// Resize visualizer canvas to fit container
function resizeCanvas() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth - 40;
    canvas.height = container.clientHeight - 80;
}

// Fetch domain and inputs configs from backend
async function fetchConfig() {
    try {
        const res = await fetch("/api/config");
        const data = await res.json();
        inputs = data.inputs || {};
        domain = data.domain || {};
        
        // Parse blocks from domain
        blocks = [];
        for (const [key, val] of Object.entries(domain)) {
            if (key.startsWith("Block")) {
                const id = parseInt(key.replace("Block", ""));
                blocks.push({
                    id: id,
                    nx: parseInt(val.N_ELEM_X || 20),
                    ny: parseInt(val.N_ELEM_Y || 20),
                    xMin: parseFloat(val.X_MIN || 0.0),
                    xMax: parseFloat(val.X_MAX || 1.0),
                    yMin: parseFloat(val.Y_MIN || 0.0),
                    yMax: parseFloat(val.Y_MAX || 1.0),
                    bcL: val.BC_L || "TRANSMISSIVE",
                    bcR: val.BC_R || "TRANSMISSIVE",
                    bcB: val.BC_B || "TRANSMISSIVE",
                    bcT: val.BC_T || "TRANSMISSIVE"
                });
            }
        }
        blocks.sort((a, b) => a.id - b.id);
        
        // Update document case path
        document.getElementById("active-case-dir").textContent = window.location.pathname === "/" ? "cases/default_case" : window.location.pathname;

        populateFormFields();
        calculateOverallBounds();
        draw();
    } catch (err) {
        console.error("Error fetching config:", err);
    }
}

// Populate UI form inputs from configs
function populateFormFields() {
    const setVal = (id, section, key, isCheckbox = false) => {
        const el = document.getElementById(id);
        if (!el) return;
        const val = inputs[section] ? inputs[section][key] : "";
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

    // Toggle fields based on active selections
    toggleIBShapeFields();
    toggleIBMethodFields();

    // Parse quads
    quads = [];
    if (inputs["ImmersedBoundary"]) {
        const numQuads = parseInt(inputs["ImmersedBoundary"]["IB_NUM_QUADS"] || 0);
        for (let i = 0; i < numQuads; ++i) {
            const qVal = inputs["ImmersedBoundary"][`IB_QUAD_${i}`];
            if (qVal) quads.push(qVal);
        }
    }
    renderQuadsTable();

    // Parse curves/polys
    polys = [];
    if (inputs["ImmersedBoundary"]) {
        const numPolys = parseInt(inputs["ImmersedBoundary"]["IB_NUM_POLYS"] || 0);
        for (let i = 0; i < numPolys; ++i) {
            const pVal = inputs["ImmersedBoundary"][`IB_POLY_${i}`];
            if (pVal) {
                const parts = pVal.trim().split(/\s+/);
                if (parts.length >= 8) {
                    polys.push({
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

function toggleIBShapeFields() {
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

function toggleIBMethodFields() {
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

function renderQuadsTable() {
    const container = document.getElementById("multi-quads-list");
    if (!container) return;
    container.innerHTML = "";
    
    if (quads.length === 0) {
        container.innerHTML = `<p style="font-size: 0.8rem; color: #888; margin: 5px 0;">No quadrilaterals defined.</p>`;
        return;
    }
    
    quads.forEach((q, idx) => {
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
            quads[idx] = e.target.value;
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
            quads.splice(idx, 1);
            renderQuadsTable();
        });
        
        div.appendChild(label);
        div.appendChild(input);
        div.appendChild(btnDel);
        container.appendChild(div);
    });
}

function renderPolysTable() {
    const tbody = document.getElementById("multi-polys-body");
    if (!tbody) return;
    tbody.innerHTML = "";
    
    if (polys.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #888; font-size: 0.8rem;">No curves defined.</td></tr>`;
        return;
    }
    
    polys.forEach((p, idx) => {
        const tr = document.createElement("tr");
        
        // 1. Dir Dropdown
        const tdDir = document.createElement("td");
        const selectDir = document.createElement("select");
        selectDir.className = "select-sm";
        selectDir.style.fontSize = "0.75rem";
        selectDir.innerHTML = `<option value="Y">Y</option><option value="X">X</option>`;
        selectDir.value = p.dir;
        
        // 4. Side Dropdown
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
        
        // 2. Coeffs q=0
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
        
        // 3. Coeffs q=1
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
        
        // 5. Actions (Delete button)
        const tdAct = document.createElement("td");
        const btnDel = document.createElement("button");
        btnDel.type = "button";
        btnDel.className = "action-btn-sm";
        btnDel.innerHTML = "&times;";
        btnDel.style.backgroundColor = "#e04e4e";
        btnDel.style.color = "white";
        btnDel.addEventListener("click", () => {
            polys.splice(idx, 1);
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

function evaluateIBQ(t) {
    const timeMapStr = (document.getElementById("ib-q-time-map") ? document.getElementById("ib-q-time-map").value : "0.0:0.0") || "0.0:0.0";
    const pairs = timeMapStr.trim().split(/\s+/).map(s => {
        const parts = s.split(":");
        return { t: parseFloat(parts[0]) || 0, q: parseFloat(parts[1]) || 0 };
    });
    if (pairs.length === 0) return 0.0;
    pairs.sort((a, b) => a.t - b.t);
    if (t <= pairs[0].t) return pairs[0].q;
    if (t >= pairs[pairs.length - 1].t) return pairs[pairs.length - 1].q;
    for (let i = 0; i < pairs.length - 1; ++i) {
        const t0 = pairs[i].t;
        const t1 = pairs[i+1].t;
        if (t >= t0 && t <= t1) {
            const q0 = pairs[i].q;
            const q1 = pairs[i+1].q;
            if (Math.abs(t1 - t0) < 1e-12) return q0;
            return q0 + (t - t0) / (t1 - t0) * (q1 - q0);
        }
    }
    return 0.0;
}

// Parse probes from inputs configuration
function parseProbesFromConfig() {
    probes = [];
    if (!inputs["Probes"]) return;
    for (const [key, val] of Object.entries(inputs["Probes"])) {
        if (key.startsWith("PROBE_")) {
            const parts = val.split(",");
            if (parts.length >= 3) {
                probes.push({
                    name: key,
                    x: parseFloat(parts[0].strip ? parts[0].strip() : parts[0].trim()),
                    y: parseFloat(parts[1].strip ? parts[1].strip() : parts[1].trim()),
                    variable: (parts[2].strip ? parts[2].strip() : parts[2].trim())
                });
            }
        }
    }
}

// Redraw point probes table
function renderProbesTable() {
    const tbody = document.getElementById("probes-list-body");
    tbody.innerHTML = "";
    probes.forEach((probe, idx) => {
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

function deleteProbe(idx) {
    probes.splice(idx, 1);
    renderProbesTable();
    updateProbesInConfig();
}

function updateProbesInConfig() {
    if (!inputs["Probes"]) inputs["Probes"] = {};
    
    // Clear old PROBE_ values
    for (const key of Object.keys(inputs["Probes"])) {
        if (key.startsWith("PROBE_")) {
            delete inputs["Probes"][key];
        }
    }
    
    // Write new values
    probes.forEach((probe, idx) => {
        inputs["Probes"][`PROBE_${idx+1}`] = `${probe.x}, ${probe.y}, ${probe.variable}`;
    });
}

// Compile forms back into JSON
function collectFormFields() {
    const getVal = (id, section, key, isNumber = false, isCheckbox = false) => {
        const el = document.getElementById(id);
        if (!el) return;
        if (!inputs[section]) inputs[section] = {};
        if (isCheckbox) {
            inputs[section][key] = el.checked ? "true" : "false";
        } else {
            inputs[section][key] = el.value;
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
    inputs["ImmersedBoundary"]["IB_SHAPE"] = shape;
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
        if (inputs["ImmersedBoundary"]) {
            for (const key of Object.keys(inputs["ImmersedBoundary"])) {
                if (key.startsWith("IB_QUAD_") || key.startsWith("IB_POLY_")) {
                    delete inputs["ImmersedBoundary"][key];
                }
            }
        } else {
            inputs["ImmersedBoundary"] = {};
        }
        
        // Write quads
        inputs["ImmersedBoundary"]["IB_NUM_QUADS"] = quads.length.toString();
        quads.forEach((q, idx) => {
            inputs["ImmersedBoundary"][`IB_QUAD_${idx}`] = q;
        });
        
        // Write polys
        inputs["ImmersedBoundary"]["IB_NUM_POLYS"] = polys.length.toString();
        polys.forEach((p, idx) => {
            inputs["ImmersedBoundary"][`IB_POLY_${idx}`] = `${p.dir} ${p.a0} ${p.b0} ${p.c0} ${p.a1} ${p.b1} ${p.c1} ${p.side}`;
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

// Calculate bounding box of the multi-block grid
function calculateOverallBounds() {
    if (blocks.length === 0) {
        bounds = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
        return;
    }
    
    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;
    
    blocks.forEach(b => {
        xMin = Math.min(xMin, b.xMin);
        xMax = Math.max(xMax, b.xMax);
        yMin = Math.min(yMin, b.yMin);
        yMax = Math.max(yMax, b.yMax);
    });
    
    bounds = { xMin, xMax, yMin, yMax };
    
    // Fit to Canvas: Calculate scale and offsets
    const margin = 45;
    const drawW = canvas.width - margin * 2;
    const drawH = canvas.height - margin * 2;
    
    const domainW = xMax - xMin;
    const domainH = yMax - yMin;
    
    // Scale to fit maintaining aspect ratio
    scale = Math.min(drawW / domainW, drawH / domainH);
    
    // Center it on canvas
    offsetX = margin + (drawW - domainW * scale) / 2 - xMin * scale;
    // Remember canvas coordinates: Y increases down, physical coordinates Y increases up
    offsetY = canvas.height - margin - (drawH - domainH * scale) / 2 + yMin * scale;
}

// Map physical coordinate (x, y) to screen pixels (cx, cy)
function mapCoordToCanvas(x, y) {
    const cx = offsetX + x * scale;
    const cy = offsetY - y * scale;
    return { x: cx, y: cy };
}

// Map screen pixels (cx, cy) back to physical coordinates (x, y)
function mapCanvasToCoord(cx, cy) {
    const x = (cx - offsetX) / scale;
    const y = (offsetY - cy) / scale;
    return { x, y };
}

// Validation Engine
function runValidation() {
    // 1. Validate element counts
    for (const b of blocks) {
        if (b.nx <= 0 || b.ny <= 0) {
            showValidationToast(`Block ${b.id} has invalid dimensions (${b.nx} x ${b.ny}). Must be positive integers.`);
            return false;
        }
        if (b.xMax <= b.xMin || b.yMax <= b.yMin) {
            showValidationToast(`Block ${b.id} coordinates are invalid: [${b.xMin}, ${b.xMax}] x [${b.yMin}, ${b.yMax}]`);
            return false;
        }
    }

    // 2. Validate block neighbor symmetry
    for (const b of blocks) {
        const checkFaceBC = (bcStr, faceName) => {
            // Check if connectivity (e.g. "1:L")
            if (bcStr.includes(":") && !isNaN(parseInt(bcStr.split(":")[0]))) {
                const parts = bcStr.split(":");
                const targetId = parseInt(parts[0]);
                const targetFace = parts[1];
                
                const targetBlock = blocks.find(x => x.id === targetId);
                if (!targetBlock) {
                    showValidationToast(`Grid Connection Error: Block ${b.id} face ${faceName} connects to missing Block ${targetId}.`);
                    return false;
                }
                
                // Expected connectivity from target side
                const expectedBC = `${b.id}:${faceName}`;
                let actualBC = "";
                if (targetFace === "L") actualBC = targetBlock.bcL;
                else if (targetFace === "R") actualBC = targetBlock.bcR;
                else if (targetFace === "B") actualBC = targetBlock.bcB;
                else if (targetFace === "T") actualBC = targetBlock.bcT;
                
                if (actualBC !== expectedBC) {
                    showValidationToast(`Symmetry Error: Block ${b.id} face ${faceName} points to Block ${targetId}:${targetFace}, but that face has boundary condition '${actualBC}' (expected '${expectedBC}').`);
                    return false;
                }

                // Check resolution matching
                const myElems = (faceName === "L" || faceName === "R") ? b.ny : b.nx;
                const nElems = (targetFace === "L" || targetFace === "R") ? targetBlock.ny : targetBlock.nx;
                if (myElems !== nElems) {
                    showValidationToast(`Mesh Mismatch: Block ${b.id} has ${myElems} elements on face ${faceName}, but connected Block ${targetId} has ${nElems} elements on face ${targetFace}.`);
                    return false;
                }
            }
            return true;
        };

        if (!checkFaceBC(b.bcL, "L")) return false;
        if (!checkFaceBC(b.bcR, "R")) return false;
        if (!checkFaceBC(b.bcB, "B")) return false;
        if (!checkFaceBC(b.bcT, "T")) return false;
    }

    // 3. Check point probes are inside the domain
    for (const p of probes) {
        let inside = false;
        for (const b of blocks) {
            if (p.x >= b.xMin && p.x <= b.xMax && p.y >= b.yMin && p.y <= b.yMax) {
                inside = true;
                break;
            }
        }
        if (!inside) {
            showValidationToast(`Warning: Point probe '${p.name}' is located at (${p.x.toFixed(3)}, ${p.y.toFixed(3)}), which is outside the computational domain.`);
            // Warning is non-fatal, return true but display toast
        }
    }

    return true;
}

function showValidationToast(msg, isError = true) {
    const toast = document.getElementById("validation-toast");
    const msgEl = document.getElementById("validation-message");
    const titleEl = document.getElementById("toast-title") || toast.querySelector("h4");
    
    if (titleEl) {
        titleEl.textContent = isError ? "Configuration Error" : "Success";
    }
    msgEl.textContent = msg;
    
    if (isError) {
        toast.classList.remove("success-toast");
        toast.classList.add("error-toast");
    } else {
        toast.classList.remove("error-toast");
        toast.classList.add("success-toast");
    }
    
    toast.classList.add("show");
    
    if (window.toastTimeout) clearTimeout(window.toastTimeout);
    window.toastTimeout = setTimeout(() => {
        toast.classList.remove("show");
    }, isError ? 8000 : 5000);
}

// Setup static DOM listeners
function setupEventListeners() {
    // Config Save Action
    document.getElementById("btn-save-config").addEventListener("click", async () => {
        collectFormFields();
        if (!runValidation()) return;

        // Clear all previous Block entries in domain so deleted ones are removed
        for (const key of Object.keys(domain)) {
            if (key.startsWith("Block")) {
                delete domain[key];
            }
        }

        // Re-compile domain object
        blocks.forEach(b => {
            const key = `Block${b.id}`;
            domain[key] = {
                N_ELEM_X: b.nx.toString(),
                N_ELEM_Y: b.ny.toString(),
                X_MIN: b.xMin.toString(),
                X_MAX: b.xMax.toString(),
                Y_MIN: b.yMin.toString(),
                Y_MAX: b.yMax.toString(),
                BC_L: b.bcL,
                BC_R: b.bcR,
                BC_B: b.bcB,
                BC_T: b.bcT
            };
        });

        try {
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: jsonStringify({ inputs, domain })
            });
            const data = await res.json();
            if (res.status === 200 && data.status === "success") {
                showValidationToast("Configuration saved successfully!", false);
                fetchConfig();
            } else {
                showValidationToast("Error: " + data.message, true);
            }
        } catch (err) {
            showValidationToast("Failed to save configuration: " + err, true);
        }
    });

    // Form Change: NACA toggle
    document.getElementById("ib-shape").addEventListener("change", toggleIBShapeFields);
    document.getElementById("ib-method").addEventListener("change", toggleIBMethodFields);

    // MULTI shape Add buttons
    const btnAddQuad = document.getElementById("btn-add-quad");
    if (btnAddQuad) {
        btnAddQuad.addEventListener("click", () => {
            quads.push("0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0");
            renderQuadsTable();
        });
    }
    const btnAddPoly = document.getElementById("btn-add-poly");
    if (btnAddPoly) {
        btnAddPoly.addEventListener("click", () => {
            polys.push({
                dir: "Y",
                a0: 0, b0: 0, c0: 0.0,
                a1: 0, b1: 0, c1: 0.0,
                side: "BELOW"
            });
            renderPolysTable();
        });
    }

    // Add Block Button
    const btnAddBlock = document.getElementById("btn-add-block");
    if (btnAddBlock) {
        btnAddBlock.addEventListener("click", () => {
            editingBlock = null;
            openBlockEditorModal();
        });
    }

    // Show Labels Checkbox
    const chkShowLabels = document.getElementById("chk-show-labels");
    if (chkShowLabels) {
        chkShowLabels.addEventListener("change", () => {
            draw();
        });
    }

    // Contour Settings Gear Button
    const btnContourGear = document.getElementById("btn-contour-gear");
    if (btnContourGear) {
        btnContourGear.addEventListener("click", () => {
            const varName = visualizerMode.replace("contour_", "");
            // Fetch/open contour modal
            openContourSettingsModal(varName);
        });
    }

    // Contour Modal Save & Cancel
    const btnContourSave = document.getElementById("btn-contour-save");
    if (btnContourSave) {
        btnContourSave.addEventListener("click", saveContourSettings);
    }
    const btnContourCancel = document.getElementById("btn-contour-cancel");
    if (btnContourCancel) {
        btnContourCancel.addEventListener("click", () => {
            document.getElementById("modal-contour-settings").style.display = "none";
        });
    }
    const contourRangeAuto = document.getElementById("contour-range-auto");
    if (contourRangeAuto) {
        contourRangeAuto.addEventListener("change", (e) => {
            document.getElementById("contour-manual-range-group").style.display = e.target.checked ? "none" : "flex";
        });
    }

    // Playback Timeline Controls
    const btnPlayPause = document.getElementById("btn-play-pause");
    if (btnPlayPause) {
        btnPlayPause.addEventListener("click", togglePlayback);
    }
    const btnStepPrev = document.getElementById("btn-step-prev");
    if (btnStepPrev) {
        btnStepPrev.addEventListener("click", () => stepPlayback(-1));
    }
    const btnStepNext = document.getElementById("btn-step-next");
    if (btnStepNext) {
        btnStepNext.addEventListener("click", () => stepPlayback(1));
    }
    const playbackSlider = document.getElementById("playback-slider");
    if (playbackSlider) {
        playbackSlider.addEventListener("input", (e) => {
            if (playbackInterval) {
                // Pause if playing
                togglePlayback();
            }
            playbackIndex = parseInt(e.target.value);
            updatePlaybackView();
        });
    }

    // Sidebar Config Tabs switcher
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", e => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            activeTab = btn.getAttribute("data-tab");
            document.getElementById(`tab-${activeTab}`).classList.add("active");
        });
    });

    // Bottom Console Tabs switcher
    document.querySelectorAll(".bottom-tab-btn").forEach(btn => {
        btn.addEventListener("click", e => {
            document.querySelectorAll(".bottom-tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".bottom-tab-pane").forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            const btab = btn.getAttribute("data-btab");
            document.getElementById(`btab-${btab}`).classList.add("active");
        });
    });

    // Toggle Visualizer mode
    document.getElementById("select-visualizer-mode").addEventListener("change", async e => {
        visualizerMode = e.target.value;
        
        const gearBtn = document.getElementById("btn-contour-gear");
        const reloadBtn = document.getElementById("btn-contour-reload");
        const playbackPanel = document.getElementById("playback-panel");
        
        if (playbackInterval) {
            clearInterval(playbackInterval);
            playbackInterval = null;
            const btn = document.getElementById("btn-play-pause");
            if (btn) btn.textContent = "▶ Play";
        }
        
        if (visualizerMode.startsWith("contour_")) {
            if (gearBtn) gearBtn.style.display = "inline-block";
            if (reloadBtn) reloadBtn.style.display = "inline-block";
            await fetchPlaybackHistory();
        } else {
            if (gearBtn) gearBtn.style.display = "none";
            if (reloadBtn) reloadBtn.style.display = "none";
            if (playbackPanel) playbackPanel.style.display = "none";
            playbackActiveVtm = null;
        }
        
        draw();
    });

    // Probe Mode toggle
    const probeModeBtn = document.getElementById("btn-probe-mode");
    probeModeBtn.addEventListener("click", () => {
        isProbeModeActive = !isProbeModeActive;
        if (isProbeModeActive) {
            probeModeBtn.textContent = "Cancel Probe";
            probeModeBtn.classList.add("active");
            canvas.parentElement.classList.add("probe-mode");
        } else {
            probeModeBtn.textContent = "Add Probe";
            probeModeBtn.classList.remove("active");
            canvas.parentElement.classList.remove("probe-mode");
        }
    });

    // Custom Probe form toggle
    document.getElementById("btn-add-probe-form").addEventListener("click", () => {
        const name = prompt("Enter probe name (e.g. PROBE_MY):");
        if (!name) return;
        const x = parseFloat(prompt("Enter X coordinate:"));
        const y = parseFloat(prompt("Enter Y coordinate:"));
        const variable = prompt("Enter sampling variable (Density, Pressure, Mach, Sigma):", "Pressure");
        if (isNaN(x) || isNaN(y) || !variable) return;
        
        probes.push({ name, x, y, variable });
        renderProbesTable();
        updateProbesInConfig();
    });

    // Simulation Execution Actions
    document.getElementById("btn-run").addEventListener("click", () => triggerSimulationRun(false));
    document.getElementById("btn-restart").addEventListener("click", triggerRestart);
    document.getElementById("btn-clean").addEventListener("click", () => triggerSimulationRun(true));
    document.getElementById("btn-clean-only").addEventListener("click", triggerCleanOnly);
    document.getElementById("btn-stop").addEventListener("click", stopSimulation);
    
    const reloadBtn = document.getElementById("btn-contour-reload");
    if (reloadBtn) {
        reloadBtn.addEventListener("click", triggerContourReload);
    }

    // Canvas Mouse Listeners
    canvas.addEventListener("mousemove", onCanvasMouseMove);
    canvas.addEventListener("click", onCanvasClick);
    canvas.addEventListener("dblclick", onCanvasDblClick);
}

// Trigger solver execute via backend API
async function triggerSimulationRun(clean) {
    if (isRunning) return;
    try {
        const res = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: jsonStringify({ clean: clean })
        });
        const data = await res.json();
        if (data.status === "success") {
            appendLogLine(`[GUI] Solver started successfully (PID: ${data.pid})...`);
            isRunning = true;
            updateStatusUI(true);
        } else {
            appendLogLine(`[GUI] Run failed: ${data.message}`);
        }
    } catch (err) {
        appendLogLine(`[GUI] Connection error launching solver: ${err}`);
    }
}

// Stop active simulation run
async function stopSimulation() {
    try {
        const res = await fetch("/api/stop", { method: "POST" });
        const data = await res.json();
        if (data.status === "success") {
            appendLogLine("[GUI] STOP signal sent to solver. Waiting for clean shutdown...");
        } else {
            appendLogLine("[GUI] Stop request failed: " + data.message);
        }
    } catch (err) {
        appendLogLine("[GUI] Error stopping solver: " + err);
    }
}

// Trigger restart sequence (stops run, updates config, restarts)
async function triggerRestart() {
    appendLogLine("[GUI] Triggering restart sequence (stopping run, updating config, restarting)...");
    try {
        const res = await fetch("/api/restart", { method: "POST" });
        const data = await res.json();
        if (data.status === "success") {
            appendLogLine(`[GUI] Solver restarted successfully (PID: ${data.pid}) from t=${data.restart_time.toFixed(3)} using ${data.restart_file}`);
            isRunning = true;
            updateStatusUI(true);
        } else {
            appendLogLine(`[GUI] Restart failed: ${data.message}`);
            showToast("Restart failed: " + data.message, true);
        }
    } catch (err) {
        appendLogLine(`[GUI] Connection error triggering restart: ${err}`);
    }
}

// Clean case outputs without launching the solver
async function triggerCleanOnly() {
    if (isRunning) return;
    if (!confirm("Are you sure you want to clean all outputs? This will delete all VTK files and logs.")) return;
    try {
        const res = await fetch("/api/clean", { method: "POST" });
        const data = await res.json();
        if (data.status === "success") {
            appendLogLine("[GUI] Case outputs cleaned successfully.");
            // Reset local states
            residualHistory = [];
            probeHistory = [];
            playbackTimesteps = [];
            playbackActiveVtm = null;
            playbackIndex = -1;
            drawPlots();
            
            // Hide playback panel
            const panel = document.getElementById("playback-panel");
            if (panel) panel.style.display = "none";
            
            draw();
            showToast("Outputs cleaned successfully!", false);
        } else {
            showToast("Failed to clean outputs: " + data.message, true);
        }
    } catch (err) {
        appendLogLine(`[GUI] Connection error cleaning outputs: ${err}`);
    }
}

// Clear the backend cache and reload visualizer contours
async function triggerContourReload() {
    try {
        appendLogLine("[GUI] Clearing visualizer cache and reloading contours...");
        const res = await fetch("/api/clear_cache", { method: "POST" });
        const data = await res.json();
        if (data.status === "success") {
            showToast("Cache cleared! Reloading...", false);
            // Fetch playback history (clears client-side preloaders and re-fetches list)
            await fetchPlaybackHistory();
            draw();
        } else {
            showToast("Failed to clear cache: " + data.message, true);
        }
    } catch (err) {
        appendLogLine(`[GUI] Connection error reloading contours: ${err}`);
    }
}

// Poll state from Python server
function startStatusPolling() {
    statusInterval = setInterval(async () => {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            
            // Check running status
            isRunning = data.running;
            updateStatusUI(isRunning);

            // Update terminal logs
            if (data.logs && data.logs.length > 0) {
                const term = document.getElementById("terminal-log");
                term.textContent = data.logs.join("\n");
                // Auto scroll to bottom
                term.scrollTop = term.scrollHeight;
            }

            // Save history for plotting
            residualHistory = data.residuals || [];
            probeHistory = data.probes || [];

            // Draw line plots
            drawPlots();

            // Refresh flow contours if active
            if (visualizerMode.startsWith("contour_")) {
                await fetchPlaybackHistory();
                draw();
            }
        } catch (err) {
            console.error("Error polling status:", err);
        }
    }, 1000);
}

function updateStatusUI(running) {
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

function appendLogLine(line) {
    const term = document.getElementById("terminal-log");
    term.textContent += "\n" + line;
    term.scrollTop = term.scrollHeight;
}

// Mouse Move: hover inspector
function onCanvasMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    lastMousePos = { x: mx, y: my };
    
    if (visualizerMode.startsWith("contour_")) {
        // Contour mode hover logic
        handleContourHover(mx, my);
        return;
    }

    const coord = mapCanvasToCoord(mx, my);
    
    // Check if hovering block elements or boundaries
    let hovered = null;
    const edgeTolerance = 12;

    for (const b of blocks) {
        // Boundaries checks
        const pL = mapCoordToCanvas(b.xMin, (b.yMin+b.yMax)/2);
        const pR = mapCoordToCanvas(b.xMax, (b.yMin+b.yMax)/2);
        const pB = mapCoordToCanvas((b.xMin+b.xMax)/2, b.yMin);
        const pT = mapCoordToCanvas((b.xMin+b.xMax)/2, b.yMax);
        
        const bLx = mapCoordToCanvas(b.xMin, 0).x;
        const bRx = mapCoordToCanvas(b.xMax, 0).x;
        const bBy = mapCoordToCanvas(0, b.yMin).y;
        const bTy = mapCoordToCanvas(0, b.yMax).y;

        // Mouse inside block?
        if (coord.x >= b.xMin && coord.x <= b.xMax && coord.y >= b.yMin && coord.y <= b.yMax) {
            hovered = { type: "block", blockId: b.id, block: b };

            // Check if close to Left boundary
            if (Math.abs(mx - bLx) < edgeTolerance) {
                hovered = { type: "boundary", blockId: b.id, face: "L", bc: b.bcL, block: b };
            }
            // Right boundary
            else if (Math.abs(mx - bRx) < edgeTolerance) {
                hovered = { type: "boundary", blockId: b.id, face: "R", bc: b.bcR, block: b };
            }
            // Bottom boundary
            else if (Math.abs(my - bBy) < edgeTolerance) {
                hovered = { type: "boundary", blockId: b.id, face: "B", bc: b.bcB, block: b };
            }
            // Top boundary
            else if (Math.abs(my - bTy) < edgeTolerance) {
                hovered = { type: "boundary", blockId: b.id, face: "T", bc: b.bcT, block: b };
            }
        }
    }

    hoveredElement = hovered;
    updateTooltip(mx, my);
}

// Tooltip display manager
function updateTooltip(mx, my) {
    const tooltip = document.getElementById("canvas-tooltip");
    if (!hoveredElement) {
        tooltip.style.opacity = 0;
        return;
    }
    
    let content = "";
    if (hoveredElement.type === "block") {
        const b = hoveredElement.block;
        content = `<strong>Block ${b.id}</strong>\nElements: ${b.nx} x ${b.ny}\nBounds X: [${b.xMin.toFixed(2)}, ${b.xMax.toFixed(2)}]\nBounds Y: [${b.yMin.toFixed(2)}, ${b.yMax.toFixed(2)}]`;
    } else if (hoveredElement.type === "boundary") {
        const faceName = hoveredElement.face === "L" ? "Left" : hoveredElement.face === "R" ? "Right" : hoveredElement.face === "B" ? "Bottom" : "Top";
        content = `<strong>Boundary: Block ${hoveredElement.blockId} ${faceName} Face</strong>\nBC Type: ${hoveredElement.bc}`;
    } else if (hoveredElement.type === "contour") {
        content = `X: ${hoveredElement.x.toFixed(4)}\nY: ${hoveredElement.y.toFixed(4)}\nValue: ${hoveredElement.val.toFixed(5)}`;
    }
    
    tooltip.innerHTML = content;
    tooltip.style.left = (mx + 20) + "px";
    tooltip.style.top = (my + 10) + "px";
    tooltip.style.opacity = 0.95;
}

// Canvas Click: add probes or edit BCs
let editingBC = null; // { blockId, face }
function onCanvasClick(e) {
    if (isProbeModeActive) {
        // Place point probe
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const coord = mapCanvasToCoord(mx, my);

        // Check if inside domain
        let inside = false;
        for (const b of blocks) {
            if (coord.x >= b.xMin && coord.x <= b.xMax && coord.y >= b.yMin && coord.y <= b.yMax) {
                inside = true;
                break;
            }
        }

        if (inside) {
            const name = prompt("Point probe placed! Enter unique name:", `PROBE_${probes.length + 1}`);
            if (name) {
                const variable = prompt("Enter variable name (Density, Pressure, Mach, Sigma):", "Pressure");
                if (variable) {
                    probes.push({ name, x: coord.x, y: coord.y, variable });
                    renderProbesTable();
                    updateProbesInConfig();
                    showValidationToast(`Probe added at physical (${coord.x.toFixed(3)}, ${coord.y.toFixed(3)})`);
                }
            }
        } else {
            showValidationToast("Cannot place probe: click must be inside the grid block bounds.");
        }
        
        // Deactivate probe mode
        document.getElementById("btn-probe-mode").click();
        return;
    }

    if (hoveredElement && hoveredElement.type === "boundary") {
        // Edit boundary condition click
        editingBC = { blockId: hoveredElement.blockId, face: hoveredElement.face, currentBC: hoveredElement.bc };
        openBCEditorModal();
    }
}

// Canvas Double Click: Edit block resolution
let editingBlock = null;
function onCanvasDblClick() {
    if (hoveredElement && hoveredElement.type === "block") {
        editingBlock = hoveredElement.block;
        openBlockEditorModal();
    }
}

// BCEditor Modals Management
function openBCEditorModal() {
    const modal = document.getElementById("modal-edit-bc");
    const faceName = editingBC.face === "L" ? "Left" : editingBC.face === "R" ? "Right" : editingBC.face === "B" ? "Bottom" : "Top";
    document.getElementById("bc-title-desc").textContent = `Block ${editingBC.blockId} Face ${faceName} (Current: ${editingBC.currentBC})`;
    
    // Parse current BC setup
    const select = document.getElementById("bc-type-select");
    const stateGroup = document.getElementById("bc-opts-state");
    const connectGroup = document.getElementById("bc-opts-connect");
    
    const bcVal = editingBC.currentBC;
    if (bcVal.includes(":") && !isNaN(parseInt(bcVal.split(":")[0]))) {
        // Connection BC
        select.value = "BLOCK_CONNECT";
        connectGroup.style.display = "block";
        stateGroup.style.display = "none";
        
        const parts = bcVal.split(":");
        document.getElementById("bc-connect-block").value = parts[0];
        document.getElementById("bc-connect-face").value = parts[1];
    } else {
        // Check for specific BC type
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

// Hook options displaying inside BC Modal
document.getElementById("bc-type-select").addEventListener("change", e => {
    const val = e.target.value;
    const stateGroup = document.getElementById("bc-opts-state");
    const connectGroup = document.getElementById("bc-opts-connect");
    if (val === "BLOCK_CONNECT") {
        connectGroup.style.display = "block";
        stateGroup.style.display = "none";
    } else if (["CHARACTERISTIC", "INFLOW_SUPERSONIC", "STATIC_PRESSURE"].includes(val)) {
        connectGroup.style.display = "none";
        stateGroup.style.display = "block";
        if (val === "CHARACTERISTIC" || val === "INFLOW_SUPERSONIC") {
            document.getElementById("bc-state-input").value = "1.0:5.0:0.0:0.75"; // Default reference state
        } else {
            document.getElementById("bc-state-input").value = "0.75"; // Default pressure
        }
    } else {
        connectGroup.style.display = "none";
        stateGroup.style.display = "none";
    }
});

document.getElementById("btn-bc-save").addEventListener("click", () => {
    const selectVal = document.getElementById("bc-type-select").value;
    let finalBC = "";
    
    if (selectVal === "BLOCK_CONNECT") {
        const blockId = document.getElementById("bc-connect-block").value;
        const face = document.getElementById("bc-connect-face").value;
        finalBC = `${blockId}:${face}`;
    } else if (["CHARACTERISTIC", "INFLOW_SUPERSONIC", "STATIC_PRESSURE"].includes(selectVal)) {
        const state = document.getElementById("bc-state-input").value;
        finalBC = `${selectVal}:${state}`;
    } else {
        finalBC = selectVal;
    }
    
    // Save to local block object
    const targetBlock = blocks.find(x => x.id === editingBC.blockId);
    if (targetBlock) {
        if (editingBC.face === "L") targetBlock.bcL = finalBC;
        else if (editingBC.face === "R") targetBlock.bcR = finalBC;
        else if (editingBC.face === "B") targetBlock.bcB = finalBC;
        else if (editingBC.face === "T") targetBlock.bcT = finalBC;
    }
    
    document.getElementById("modal-edit-bc").style.display = "none";
    draw();
});

document.getElementById("btn-bc-cancel").addEventListener("click", () => {
    document.getElementById("modal-edit-bc").style.display = "none";
});

// Block editor modals management
function openBlockEditorModal() {
    const modal = document.getElementById("modal-edit-block");
    document.getElementById("eb-nx").value = editingBlock ? editingBlock.nx : 100;
    document.getElementById("eb-ny").value = editingBlock ? editingBlock.ny : 100;
    document.getElementById("eb-xmin").value = editingBlock ? editingBlock.xMin : 0.0;
    document.getElementById("eb-xmax").value = editingBlock ? editingBlock.xMax : 1.0;
    document.getElementById("eb-ymin").value = editingBlock ? editingBlock.yMin : 0.0;
    document.getElementById("eb-ymax").value = editingBlock ? editingBlock.yMax : 1.0;
    
    const titleEl = document.getElementById("eb-modal-title");
    if (titleEl) {
        titleEl.textContent = editingBlock ? "Edit Block Settings" : "Add New Block";
    }
    const deleteBtn = document.getElementById("btn-eb-delete");
    if (deleteBtn) {
        deleteBtn.style.display = editingBlock ? "inline-block" : "none";
    }
    modal.style.display = "flex";
}

document.getElementById("btn-eb-save").addEventListener("click", () => {
    const nx = parseInt(document.getElementById("eb-nx").value);
    const ny = parseInt(document.getElementById("eb-ny").value);
    const xMin = parseFloat(document.getElementById("eb-xmin").value);
    const xMax = parseFloat(document.getElementById("eb-xmax").value);
    const yMin = parseFloat(document.getElementById("eb-ymin").value);
    const yMax = parseFloat(document.getElementById("eb-ymax").value);
    
    if (editingBlock) {
        editingBlock.nx = nx;
        editingBlock.ny = ny;
        editingBlock.xMin = xMin;
        editingBlock.xMax = xMax;
        editingBlock.yMin = yMin;
        editingBlock.yMax = yMax;
        
        const blockName = `Block${editingBlock.id}`;
        if (domain[blockName]) {
            domain[blockName]["N_ELEM_X"] = nx.toString();
            domain[blockName]["N_ELEM_Y"] = ny.toString();
            domain[blockName]["X_MIN"] = xMin.toString();
            domain[blockName]["X_MAX"] = xMax.toString();
            domain[blockName]["Y_MIN"] = yMin.toString();
            domain[blockName]["Y_MAX"] = yMax.toString();
        }
    } else {
        let newId = 0;
        blocks.forEach(b => {
            if (b.id >= newId) newId = b.id + 1;
        });
        
        const newBlock = {
            id: newId,
            nx: nx,
            ny: ny,
            xMin: xMin,
            xMax: xMax,
            yMin: yMin,
            yMax: yMax,
            bcL: "TRANSMISSIVE",
            bcR: "TRANSMISSIVE",
            bcB: "TRANSMISSIVE",
            bcT: "TRANSMISSIVE"
        };
        blocks.push(newBlock);
        
        const blockName = `Block${newId}`;
        domain[blockName] = {
            N_ELEM_X: nx.toString(),
            N_ELEM_Y: ny.toString(),
            X_MIN: xMin.toString(),
            X_MAX: xMax.toString(),
            Y_MIN: yMin.toString(),
            Y_MAX: yMax.toString(),
            BC_L: "TRANSMISSIVE",
            BC_R: "TRANSMISSIVE",
            BC_B: "TRANSMISSIVE",
            BC_T: "TRANSMISSIVE"
        };
    }
    
    document.getElementById("modal-edit-block").style.display = "none";
    calculateOverallBounds();
    draw();
});

document.getElementById("btn-eb-cancel").addEventListener("click", () => {
    document.getElementById("modal-edit-block").style.display = "none";
});

const btnEbDelete = document.getElementById("btn-eb-delete");
if (btnEbDelete) {
    btnEbDelete.addEventListener("click", () => {
        if (editingBlock) {
            const idx = blocks.indexOf(editingBlock);
            if (idx !== -1) {
                blocks.splice(idx, 1);
            }
            const blockName = `Block${editingBlock.id}`;
            delete domain[blockName];
        }
        document.getElementById("modal-edit-block").style.display = "none";
        calculateOverallBounds();
        draw();
    });
}

// Canvas Drawer
let latestVtsData = [];
let vtsMinMax = { min: 0, max: 1 };
async function fetchVTSData(varName) {
    try {
        const res = await fetch(`/api/vts_data?var=${varName}`);
        latestVtsData = await res.json();
        
        // Find min/max values across all parsed grids
        let min = Infinity, max = -Infinity;
        latestVtsData.forEach(block => {
            block.values.forEach(v => {
                min = Math.min(min, v);
                max = Math.max(max, v);
            });
        });
        if (min === Infinity) min = 0;
        if (max === -Infinity) max = 1;
        if (Math.abs(max - min) < 1e-8) max = min + 1.0;
        vtsMinMax = { min, max };
    } catch (err) {
        console.error("Error fetching VTS flow data:", err);
    }
}

async function draw() {
    const contourImg = document.getElementById("contour-image-view");
    const gridCanvas = document.getElementById("grid-canvas");
    const legend = document.getElementById("grid-legend");
    
    if (visualizerMode.startsWith("contour_")) {
        gridCanvas.style.display = "none";
        if (legend) legend.style.display = "none";
        contourImg.style.display = "block";
        
        const varName = visualizerMode.replace("contour_", "");
        let url = `/api/contour_image?var=${varName}`;
        if (playbackActiveVtm) {
            url += `&vtm=${playbackActiveVtm}`;
            const entry = playbackTimesteps.find(t => t.vtm === playbackActiveVtm);
            if (entry && entry.mtime) {
                url += `&mtime=${entry.mtime}`;
            }
        }
        contourImg.src = url;
        return;
    } else {
        contourImg.style.display = "none";
        gridCanvas.style.display = "block";
        if (legend) legend.style.display = "flex";
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Grid Visualizer Mode
    blocks.forEach(b => {
        const p1 = mapCoordToCanvas(b.xMin, b.yMin);
        const p2 = mapCoordToCanvas(b.xMax, b.yMax);
        
        const w = p2.x - p1.x;
        const h = p2.y - p1.y; // height will be negative because Y is flipped
        
        // Draw block fill
        ctx.fillStyle = "rgba(18, 22, 38, 0.5)";
        ctx.fillRect(p1.x, p1.y, w, h);
        
        // Draw internal mesh elements (thin dotted)
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.lineWidth = 0.5;
        
        // Vertical lines
        const dx = (b.xMax - b.xMin) / b.nx;
        for (let i = 1; i < b.nx; ++i) {
            const px = b.xMin + i * dx;
            const pt1 = mapCoordToCanvas(px, b.yMin);
            const pt2 = mapCoordToCanvas(px, b.yMax);
            ctx.beginPath();
            ctx.moveTo(pt1.x, pt1.y);
            ctx.lineTo(pt2.x, pt2.y);
            ctx.stroke();
        }
        // Horizontal lines
        const dy = (b.yMax - b.yMin) / b.ny;
        for (let j = 1; j < b.ny; ++j) {
            const py = b.yMin + j * dy;
            const pt1 = mapCoordToCanvas(b.xMin, py);
            const pt2 = mapCoordToCanvas(b.xMax, py);
            ctx.beginPath();
            ctx.moveTo(pt1.x, pt1.y);
            ctx.lineTo(pt2.x, pt2.y);
            ctx.stroke();
        }

        // Draw boundary edges
        const drawEdge = (x1, y1, x2, y2, bc) => {
            const pt1 = mapCoordToCanvas(x1, y1);
            const pt2 = mapCoordToCanvas(x2, y2);
            
            ctx.beginPath();
            ctx.moveTo(pt1.x, pt1.y);
            ctx.lineTo(pt2.x, pt2.y);
            
            // Set styles based on BC
            if (bc.includes(":") && !isNaN(parseInt(bc.split(":")[0]))) {
                // Connection BC (dashed cyan)
                ctx.strokeStyle = varColorHex("connection");
                ctx.setLineDash([4, 4]);
                ctx.lineWidth = 2.5;
            } else {
                ctx.setLineDash([]);
                ctx.lineWidth = 3.5;
                if (bc.startsWith("WALL")) {
                    ctx.strokeStyle = varColorHex("wall");
                } else if (bc.startsWith("INFLOW")) {
                    ctx.strokeStyle = varColorHex("inflow");
                } else if (bc.startsWith("CHARACTERISTIC")) {
                    ctx.strokeStyle = varColorHex("characteristic");
                } else {
                    ctx.strokeStyle = varColorHex("outflow"); // transmissive / outflow
                }
            }
            ctx.stroke();
            ctx.setLineDash([]); // Reset
        };

        drawEdge(b.xMin, b.yMin, b.xMin, b.yMax, b.bcL); // Left
        drawEdge(b.xMax, b.yMin, b.xMax, b.yMax, b.bcR); // Right
        drawEdge(b.xMin, b.yMin, b.xMax, b.yMin, b.bcB); // Bottom
        drawEdge(b.xMin, b.yMax, b.xMax, b.yMax, b.bcT); // Top

        // Draw block labels
        ctx.fillStyle = "rgba(230, 237, 243, 0.4)";
        ctx.font = "bold 13px 'Outfit'";
        ctx.fillText(`BLOCK ${b.id}`, p1.x + 10, p1.y + h + 20);
    });

    // Draw active connections pointers (arrows)
    drawBlockConnectionArrows();

    // Draw point probes
    probes.forEach(probe => {
        const pt = mapCoordToCanvas(probe.x, probe.y);
        ctx.fillStyle = varColorHex("characteristic");
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.stroke();
        
        ctx.fillStyle = "#fff";
        ctx.font = "10px monospace";
        ctx.fillText(probe.name, pt.x + 8, pt.y + 3);
    });

    // Draw Cylinder/Airfoil solid boundary
    drawSolidGeometries(false);

    // Show Labels & Origin overlay
    const showLabels = document.getElementById("chk-show-labels") ? document.getElementById("chk-show-labels").checked : true;
    if (showLabels) {
        ctx.fillStyle = "rgba(0, 242, 254, 0.85)";
        ctx.font = "10px monospace";
        
        const corners = [
            { x: bounds.xMin, y: bounds.yMin, label: `(${bounds.xMin.toFixed(2)}, ${bounds.yMin.toFixed(2)})`, align: 'right', baseline: 'top', dx: -5, dy: 5 },
            { x: bounds.xMax, y: bounds.yMin, label: `(${bounds.xMax.toFixed(2)}, ${bounds.yMin.toFixed(2)})`, align: 'left', baseline: 'top', dx: 5, dy: 5 },
            { x: bounds.xMin, y: bounds.yMax, label: `(${bounds.xMin.toFixed(2)}, ${bounds.yMax.toFixed(2)})`, align: 'right', baseline: 'bottom', dx: -5, dy: -5 },
            { x: bounds.xMax, y: bounds.yMax, label: `(${bounds.xMax.toFixed(2)}, ${bounds.yMax.toFixed(2)})`, align: 'left', baseline: 'bottom', dx: 5, dy: -5 }
        ];
        
        corners.forEach(c => {
            const pt = mapCoordToCanvas(c.x, c.y);
            ctx.textAlign = c.align;
            ctx.textBaseline = c.baseline;
            ctx.fillText(c.label, pt.x + c.dx, pt.y + c.dy);
        });
        
        // Cyan crosshairs for physical (0,0) origin
        if (0 >= bounds.xMin && 0 <= bounds.xMax && 0 >= bounds.yMin && 0 <= bounds.yMax) {
            const pt0 = mapCoordToCanvas(0.0, 0.0);
            ctx.strokeStyle = "rgba(0, 242, 254, 0.75)";
            ctx.lineWidth = 1.0;
            ctx.beginPath();
            ctx.moveTo(pt0.x - 15, pt0.y);
            ctx.lineTo(pt0.x + 15, pt0.y);
            ctx.moveTo(pt0.x, pt0.y - 15);
            ctx.lineTo(pt0.x, pt0.y + 15);
            ctx.stroke();
            
            ctx.fillStyle = "rgba(0, 242, 254, 0.85)";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.fillText(" (0,0)", pt0.x, pt0.y + 2);
        }
    }
}

// Render dynamic connection flowlines in grid mode
function drawBlockConnectionArrows() {
    blocks.forEach(b => {
        const checkConnection = (bcStr, faceName) => {
            if (bcStr.includes(":") && !isNaN(parseInt(bcStr.split(":")[0]))) {
                const parts = bcStr.split(":");
                const nid = parseInt(parts[0]);
                const nface = parts[1];
                const nb = blocks.find(x => x.id === nid);
                if (!nb) return;

                // Center coordinates of my face
                let myX = 0, myY = 0, nX = 0, nY = 0;
                
                if (faceName === "L") { myX = b.xMin; myY = (b.yMin+b.yMax)/2; }
                else if (faceName === "R") { myX = b.xMax; myY = (b.yMin+b.yMax)/2; }
                else if (faceName === "B") { myX = (b.xMin+b.xMax)/2; myY = b.yMin; }
                else if (faceName === "T") { myX = (b.xMin+b.xMax)/2; myY = b.yMax; }

                if (nface === "L") { nX = nb.xMin; nY = (nb.yMin+nb.yMax)/2; }
                else if (nface === "R") { nX = nb.xMax; nY = (nb.yMin+nb.yMax)/2; }
                else if (nface === "B") { nX = (nb.xMin+nb.xMax)/2; nY = nb.yMin; }
                else if (nface === "T") { nX = (nb.xMin+nb.xMax)/2; nY = nb.yMax; }

                const pt1 = mapCoordToCanvas(myX, myY);
                const pt2 = mapCoordToCanvas(nX, nY);

                // Draw arrow line between face centers
                ctx.beginPath();
                ctx.moveTo(pt1.x, pt1.y);
                ctx.lineTo(pt2.x, pt2.y);
                ctx.strokeStyle = "rgba(0, 210, 255, 0.4)";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([2, 4]);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        };
        checkConnection(b.bcL, "L");
        checkConnection(b.bcR, "R");
        checkConnection(b.bcB, "B");
        checkConnection(b.bcT, "T");
    });
}

// Render flow heatmap contours directly using 2D canvas cells
function drawFlowContours() {
    latestVtsData.forEach(block => {
        const nx = block.nx;
        const ny = block.ny;
        
        // Each element has structured points
        for (let j = 0; j < ny - 1; ++j) {
            for (let i = 0; i < nx - 1; ++i) {
                // Get corner indices
                const idx0 = j * nx + i;
                const idx1 = j * nx + (i + 1);
                const idx2 = (j + 1) * nx + (i + 1);
                const idx3 = (j + 1) * nx + i;
                
                // Get physical values
                const p0 = mapCoordToCanvas(block.x[idx0], block.y[idx0]);
                const p1 = mapCoordToCanvas(block.x[idx1], block.y[idx1]);
                const p2 = mapCoordToCanvas(block.x[idx2], block.y[idx2]);
                const p3 = mapCoordToCanvas(block.x[idx3], block.y[idx3]);
                
                const v0 = block.values[idx0];
                const v1 = block.values[idx1];
                const v2 = block.values[idx2];
                const v3 = block.values[idx3];
                
                // Draw 2 triangles to represent quad, interpolate colors
                // Triangle 1: p0, p1, p2
                const avgV1 = (v0 + v1 + v2) / 3.0;
                ctx.fillStyle = getColorForValue(avgV1);
                ctx.beginPath();
                ctx.moveTo(p0.x, p0.y);
                ctx.lineTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.closePath();
                ctx.fill();
                
                // Triangle 2: p0, p2, p3
                const avgV2 = (v0 + v2 + v3) / 3.0;
                ctx.fillStyle = getColorForValue(avgV2);
                ctx.beginPath();
                ctx.moveTo(p0.x, p0.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.lineTo(p3.x, p3.y);
                ctx.closePath();
                ctx.fill();
            }
        }
    });
}

// Convert normalized values to HSL colors (Rainbow map)
function getColorForValue(val) {
    const min = vtsMinMax.min;
    const max = vtsMinMax.max;
    
    // Normalize to [0, 1]
    let norm = (val - min) / (max - min);
    norm = Math.max(0.0, Math.min(1.0, norm));
    
    // Map to HSL: Blue (240 deg) to Red (0 deg)
    const hue = (1.0 - norm) * 240;
    return `hsl(${hue}, 100%, 45%)`;
}

// Hover inspector on active flow contour plotter
function handleContourHover(mx, my) {
    let bestVal = null;
    const coord = mapCanvasToCoord(mx, my);
    
    // Search elements to see if mouse is inside one
    for (const block of latestVtsData) {
        const nx = block.nx;
        const ny = block.ny;
        
        for (let j = 0; j < ny - 1; ++j) {
            for (let i = 0; i < nx - 1; ++i) {
                const idx0 = j * nx + i;
                const idx2 = (j + 1) * nx + (i + 1);
                
                const xMin = Math.min(block.x[idx0], block.x[idx2]);
                const xMax = Math.max(block.x[idx0], block.x[idx2]);
                const yMin = Math.min(block.y[idx0], block.y[idx2]);
                const yMax = Math.max(block.y[idx0], block.y[idx2]);
                
                if (coord.x >= xMin && coord.x <= xMax && coord.y >= yMin && coord.y <= yMax) {
                    // Simple average inside cell
                    bestVal = block.values[idx0];
                    break;
                }
            }
            if (bestVal !== null) break;
        }
        if (bestVal !== null) break;
    }
    
    if (bestVal !== null) {
        hoveredElement = { type: "contour", x: coord.x, y: coord.y, val: bestVal };
    } else {
        hoveredElement = null;
    }
    updateTooltip(mx, my);
}

// Render Solid Obstacles (cylinder / NACA airfoil profile)
function drawSolidGeometries(outlineOnly) {
    const ibEnable = inputs["ImmersedBoundary"] ? inputs["ImmersedBoundary"]["ENABLE_IB"] === "true" : false;
    if (!ibEnable) return;
    
    const shape = inputs["ImmersedBoundary"]["IB_SHAPE"];
    const cx = parseFloat(inputs["ImmersedBoundary"]["IB_CENTER_X"] || 0.0);
    const cy = parseFloat(inputs["ImmersedBoundary"]["IB_CENTER_Y"] || 0.0);
    const radius = parseFloat(inputs["ImmersedBoundary"]["IB_RADIUS"] || 0.5);

    if (shape === "CIRCLE") {
        const pt = mapCoordToCanvas(cx, cy);
        const rPix = radius * scale;
        
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, rPix, 0, Math.PI * 2);
        
        if (outlineOnly) {
            ctx.strokeStyle = varColorHex("red");
            ctx.lineWidth = 2.0;
            ctx.stroke();
        } else {
            ctx.fillStyle = "rgba(255, 59, 48, 0.25)";
            ctx.fill();
            ctx.strokeStyle = varColorHex("red");
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    } else if (shape === "NACA") {
        const nacaCode = inputs["ImmersedBoundary"]["IB_NACA_CODE"] || "0012";
        const aoaDeg = parseFloat(inputs["ImmersedBoundary"]["IB_AOA"] || 0.0);
        const chord = radius; // Chord length maps to IB_RADIUS in inputs.dat
        
        drawNacaAirfoil(cx, cy, chord, nacaCode, aoaDeg, outlineOnly);
    } else if (shape === "MULTI") {
        let t_eval = 0.0;
        if (playbackIndex >= 0 && playbackIndex < playbackTimesteps.length) {
            t_eval = playbackTimesteps[playbackIndex].time;
        } else {
            t_eval = parseFloat(inputs["IO"] ? inputs["IO"]["RESTART_TIME"] : 0.0) || 0.0;
        }
        const q_val = evaluateIBQ(t_eval);
        
        // 1. Draw Quads
        quads.forEach(q => {
            const parts = q.trim().split(/\s+/).map(parseFloat);
            if (parts.length === 8) {
                ctx.beginPath();
                let pt = mapCoordToCanvas(parts[0], parts[1]);
                ctx.moveTo(pt.x, pt.y);
                for (let j = 1; j < 4; j++) {
                    pt = mapCoordToCanvas(parts[j*2], parts[j*2 + 1]);
                    ctx.lineTo(pt.x, pt.y);
                }
                ctx.closePath();
                if (outlineOnly) {
                    ctx.strokeStyle = varColorHex("red");
                    ctx.lineWidth = 2.0;
                    ctx.stroke();
                } else {
                    ctx.fillStyle = "rgba(255, 59, 48, 0.25)";
                    ctx.fill();
                    ctx.strokeStyle = varColorHex("red");
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
            }
        });
        
        // 2. Draw Curves/Polys
        polys.forEach(p => {
            const a = (1.0 - q_val) * p.a0 + q_val * p.a1;
            const b = (1.0 - q_val) * p.b0 + q_val * p.b1;
            const c = (1.0 - q_val) * p.c0 + q_val * p.c1;
            
            const steps = 100;
            
            // Mask shading (if filled and bounds are valid)
            if (!outlineOnly) {
                ctx.fillStyle = "rgba(255, 59, 48, 0.15)";
                ctx.beginPath();
                let first = true;
                if (p.dir === "Y") {
                    for (let step = 0; step <= steps; step++) {
                        const px = bounds.xMin + (step / steps) * (bounds.xMax - bounds.xMin);
                        const py = a * px * px + b * px + c;
                        const pt = mapCoordToCanvas(px, py);
                        if (first) {
                            ctx.moveTo(pt.x, pt.y);
                            first = false;
                        } else {
                            ctx.lineTo(pt.x, pt.y);
                        }
                    }
                    const yEdge = p.side === "BELOW" ? bounds.yMin : bounds.yMax;
                    const ptR = mapCoordToCanvas(bounds.xMax, yEdge);
                    const ptL = mapCoordToCanvas(bounds.xMin, yEdge);
                    ctx.lineTo(ptR.x, ptR.y);
                    ctx.lineTo(ptL.x, ptL.y);
                } else { // dir === "X"
                    for (let step = 0; step <= steps; step++) {
                        const py = bounds.yMin + (step / steps) * (bounds.yMax - bounds.yMin);
                        const px = a * py * py + b * py + c;
                        const pt = mapCoordToCanvas(px, py);
                        if (first) {
                            ctx.moveTo(pt.x, pt.y);
                            first = false;
                        } else {
                            ctx.lineTo(pt.x, pt.y);
                        }
                    }
                    const xEdge = p.side === "LEFT" ? bounds.xMin : bounds.xMax;
                    const ptT = mapCoordToCanvas(xEdge, bounds.yMax);
                    const ptB = mapCoordToCanvas(xEdge, bounds.yMin);
                    ctx.lineTo(ptT.x, ptT.y);
                    ctx.lineTo(ptB.x, ptB.y);
                }
                ctx.closePath();
                ctx.fill();
            }
            
            // Curve Outline
            ctx.beginPath();
            let first = true;
            if (p.dir === "Y") {
                for (let step = 0; step <= steps; step++) {
                    const px = bounds.xMin + (step / steps) * (bounds.xMax - bounds.xMin);
                    const py = a * px * px + b * px + c;
                    const pt = mapCoordToCanvas(px, py);
                    if (first) {
                        ctx.moveTo(pt.x, pt.y);
                        first = false;
                    } else {
                        ctx.lineTo(pt.x, pt.y);
                    }
                }
            } else { // dir === "X"
                for (let step = 0; step <= steps; step++) {
                    const py = bounds.yMin + (step / steps) * (bounds.yMax - bounds.yMin);
                    const px = a * py * py + b * py + c;
                    const pt = mapCoordToCanvas(px, py);
                    if (first) {
                        ctx.moveTo(pt.x, pt.y);
                        first = false;
                    } else {
                        ctx.lineTo(pt.x, pt.y);
                    }
                }
            }
            ctx.strokeStyle = varColorHex("red");
            ctx.lineWidth = 2.0;
            ctx.stroke();
        });
    }
}

// Math solver to evaluate NACA profiles and draw outline
function drawNacaAirfoil(leX, leY, chord, codeStr, aoaDeg, outlineOnly) {
    const t = parseFloat(codeStr.substring(2)) / 100.0; // Thickness (last 2 digits)
    const m = parseFloat(codeStr.substring(0, 1)) / 100.0; // Max Camber (1st digit)
    const p = parseFloat(codeStr.substring(1, 2)) / 10.0; // Camber Position (2nd digit)
    
    const pointsCount = 45;
    const upperPts = [];
    const lowerPts = [];
    const rad = -aoaDeg * Math.PI / 180.0; // Angle of attack rotation in radians

    for (let i = 0; i <= pointsCount; ++i) {
        // Linear or cosine spacing
        const beta = Math.PI * i / pointsCount;
        const xc = 0.5 * (1.0 - Math.cos(beta)); // Cosine cluster spacing near leading edge
        
        // Thickness distribution
        const yt = 5.0 * t * (0.2969 * Math.sqrt(xc) - 0.1260 * xc - 0.3516 * xc * xc + 0.2843 * xc * xc * xc - 0.1015 * xc * xc * xc * xc);
        
        // Camber line solver
        let yc = 0.0;
        let theta = 0.0;
        if (m > 0 && p > 0) {
            if (xc < p) {
                yc = (m / (p * p)) * (2.0 * p * xc - xc * xc);
                const dyc = (2.0 * m / (p * p)) * (p - xc);
                theta = Math.atan(dyc);
            } else {
                yc = (m / ((1.0 - p) * (1.0 - p))) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc * xc);
                const dyc = (2.0 * m / ((1.0 - p) * (1.0 - p))) * (p - xc);
                theta = Math.atan(dyc);
            }
        }
        
        // Top and bottom coordinate offsets
        const xu = xc - yt * Math.sin(theta);
        const yu = yc + yt * Math.cos(theta);
        const xl = xc + yt * Math.sin(theta);
        const yl = yc - yt * Math.cos(theta);
        
        // Scale to physical size
        let xUpperPhy = xu * chord;
        let yUpperPhy = yu * chord;
        let xLowerPhy = xl * chord;
        let yLowerPhy = yl * chord;
        
        // Rotate by AOA around leading edge
        const rotate = (x, y) => {
            const rx = x * Math.cos(rad) - y * Math.sin(rad);
            const ry = x * Math.sin(rad) + y * Math.cos(rad);
            return { x: rx, y: ry };
        };
        
        const rUpper = rotate(xUpperPhy, yUpperPhy);
        const rLower = rotate(xLowerPhy, yLowerPhy);
        
        // Shift to leading edge offset
        upperPts.push({ x: leX + rUpper.x, y: leY + rUpper.y });
        lowerPts.push({ x: leX + rLower.x, y: leY + rLower.y });
    }

    // Connect outline points
    ctx.beginPath();
    const startPt = mapCoordToCanvas(upperPts[0].x, upperPts[0].y);
    ctx.moveTo(startPt.x, startPt.y);
    
    // Draw upper surface
    for (let i = 1; i < upperPts.length; ++i) {
        const pt = mapCoordToCanvas(upperPts[i].x, upperPts[i].y);
        ctx.lineTo(pt.x, pt.y);
    }
    // Draw lower surface back to leading edge
    for (let i = lowerPts.length - 1; i >= 0; --i) {
        const pt = mapCoordToCanvas(lowerPts[i].x, lowerPts[i].y);
        ctx.lineTo(pt.x, pt.y);
    }
    ctx.closePath();

    if (outlineOnly) {
        ctx.strokeStyle = varColorHex("red");
        ctx.lineWidth = 2.0;
        ctx.stroke();
    } else {
        ctx.fillStyle = "rgba(255, 59, 48, 0.25)";
        ctx.fill();
        ctx.strokeStyle = varColorHex("red");
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}

// Helper colors mapper
function varColorHex(name) {
    if (name === "wall") return "#6e7681";
    if (name === "inflow") return "#39ff14";
    if (name === "outflow") return "#ff5e00";
    if (name === "characteristic") return "#ff007f";
    if (name === "connection") return "#00d2ff";
    if (name === "red") return "#ff3b30";
    return "#fff";
}

// CUSTOM CANVAS LINE PLOTTER
function drawPlots() {
    // 1. Draw Residuals Plot
    const rCanvas = document.getElementById("residuals-chart");
    if (rCanvas && rCanvas.offsetParent !== null) {
        const rCtx = rCanvas.getContext("2d");
        drawSingleLinePlot(rCanvas, rCtx, residualHistory, ["rho", "rhou", "rhov", "E"], ["#00d2ff", "#39ff14", "#ffcc00", "#ff007f"], "Residuals", true);
    }
    
    // 2. Draw Probes Plot
    const pCanvas = document.getElementById("probes-chart");
    if (pCanvas && pCanvas.offsetParent !== null && probeHistory.length > 0) {
        const pCtx = pCanvas.getContext("2d");
        const keys = Object.keys(probeHistory[0]).filter(k => k !== "Time");
        const colors = ["#00d2ff", "#39ff14", "#ffcc00", "#ff007f", "#ff5e00", "#a200ff"];
        drawSingleLinePlot(pCanvas, pCtx, probeHistory, keys, colors, "Probes", false);
    }
}

// Lightweight native Canvas Line Plotter implementation (fully offline)
function drawSingleLinePlot(canvasEl, ctxEl, history, keys, colors, title, isLogScale) {
    // Resize to fit element bounding
    const rect = canvasEl.getBoundingClientRect();
    if (canvasEl.width !== rect.width || canvasEl.height !== rect.height) {
        canvasEl.width = rect.width;
        canvasEl.height = rect.height;
    }

    ctxEl.clearRect(0, 0, canvasEl.width, canvasEl.height);
    
    const marginL = 55;
    const marginR = 20;
    const marginT = 25;
    const marginB = 25;
    
    const w = canvasEl.width - marginL - marginR;
    const h = canvasEl.height - marginT - marginB;

    if (history.length === 0) {
        ctxEl.fillStyle = "#8b949e";
        ctxEl.font = "12px monospace";
        ctxEl.fillText("Waiting for simulation data...", marginL + 20, marginT + 30);
        return;
    }

    // Determine domain bounds X (Time)
    const tMin = history[0].time || history[0].Time || 0.0;
    const tMax = history[history.length - 1].time || history[history.length - 1].Time || 1.0;
    const tRange = tMax - tMin || 1.0;

    // Determine range bounds Y (Variables)
    let yMin = Infinity, yMax = -Infinity;
    history.forEach(row => {
        keys.forEach(k => {
            let v = row[k];
            if (isLogScale) {
                v = Math.log10(Math.abs(v) + 1e-18);
            }
            yMin = Math.min(yMin, v);
            yMax = Math.max(yMax, v);
        });
    });
    
    if (yMin === Infinity) yMin = -6;
    if (yMax === -Infinity) yMax = 0;
    let yRange = yMax - yMin;
    if (Math.abs(yRange) < 1e-8) {
        yMax += 1.0;
        yMin -= 1.0;
        yRange = 2.0;
    }

    // Draw Plot Background Grid
    ctxEl.fillStyle = "#05070f";
    ctxEl.fillRect(marginL, marginT, w, h);
    ctxEl.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctxEl.lineWidth = 1;
    
    // Y Gridlines
    const gridLines = 4;
    for (let i = 0; i <= gridLines; ++i) {
        const yVal = yMin + i * (yRange / gridLines);
        const cy = marginT + h - ((yVal - yMin) / yRange) * h;
        
        ctxEl.beginPath();
        ctxEl.moveTo(marginL, cy);
        ctxEl.lineTo(marginL + w, cy);
        ctxEl.stroke();
        
        // Label
        ctxEl.fillStyle = "#8b949e";
        ctxEl.font = "9px monospace";
        let lbl = "";
        if (isLogScale) {
            lbl = "1e" + Math.round(yVal);
        } else {
            lbl = yVal.toFixed(3);
        }
        ctxEl.fillText(lbl, 8, cy + 3);
    }

    // X Gridlines (Time ticks)
    for (let i = 0; i <= 5; ++i) {
        const xVal = tMin + i * (tRange / 5);
        const cx = marginL + ((xVal - tMin) / tRange) * w;
        
        ctxEl.beginPath();
        ctxEl.moveTo(cx, marginT);
        ctxEl.lineTo(cx, marginT + h);
        ctxEl.stroke();
        
        // Label
        ctxEl.fillStyle = "#8b949e";
        ctxEl.font = "9px monospace";
        ctxEl.fillText(xVal.toFixed(3), cx - 10, marginT + h + 15);
    }

    // Plot Lines
    keys.forEach((key, kIdx) => {
        ctxEl.beginPath();
        history.forEach((row, rIdx) => {
            const t = row.time || row.Time;
            let v = row[key];
            if (isLogScale) {
                v = Math.log10(Math.abs(v) + 1e-18);
            }
            
            const cx = marginL + ((t - tMin) / tRange) * w;
            const cy = marginT + h - ((v - yMin) / yRange) * h;
            
            if (rIdx === 0) ctxEl.moveTo(cx, cy);
            else ctxEl.lineTo(cx, cy);
        });
        
        ctxEl.strokeStyle = colors[kIdx % colors.length];
        ctxEl.lineWidth = 1.5;
        ctxEl.stroke();
    });

    // Draw Plot Labels / Title Legend
    ctxEl.fillStyle = "#fff";
    ctxEl.font = "bold 10px sans-serif";
    ctxEl.fillText(title, marginL + 10, marginT - 10);
    
    // Draw Keys Legend
    let legendOffset = marginL + 90;
    keys.forEach((key, kIdx) => {
        ctxEl.fillStyle = colors[kIdx % colors.length];
        ctxEl.fillRect(legendOffset, marginT - 16, 10, 6);
        ctxEl.fillStyle = "#8b949e";
        ctxEl.font = "9px sans-serif";
        ctxEl.fillText(key, legendOffset + 14, marginT - 10);
        legendOffset += key.length * 7 + 35;
    });
}

// Utility stringify parser to bypass serialization limits
function jsonStringify(obj) {
    return JSON.stringify(obj, null, 2);
}

// Contour settings management
function openContourSettingsModal(varName) {
    const modal = document.getElementById("modal-contour-settings");
    if (!modal) return;
    
    const settings = contourSettings[varName] || {
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

async function saveContourSettings() {
    const varName = visualizerMode.replace("contour_", "");
    const modal = document.getElementById("modal-contour-settings");
    
    const settings = {
        cmap: document.getElementById("contour-cmap").value,
        levels: parseInt(document.getElementById("contour-levels").value) || 50,
        range_mode: document.getElementById("contour-range-auto").checked ? "auto" : "manual",
        vmin: parseFloat(document.getElementById("contour-vmin").value) || 0.0,
        vmax: parseFloat(document.getElementById("contour-vmax").value) || 1.0,
        show_grid: document.getElementById("contour-show-grid").checked
    };
    
    contourSettings[varName] = settings;
    modal.style.display = "none";
    
    try {
        const res = await fetch("/api/webcontour", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ var: varName, settings: settings })
        });
        const data = await res.json();
        if (data.status === "success") {
            draw();
        } else {
            console.error("Failed to save .webcontour backend side:", data.message);
        }
    } catch (err) {
        console.error("Failed to POST webcontour settings:", err);
    }
}

// Preload contour images to browser cache sequentially
function preloadContourImages(varName) {
    if (!playbackTimesteps || !playbackTimesteps.length) return;
    
    // Cancel any ongoing preload queue
    if (window.currentPreloadQueue) {
        window.currentPreloadQueue.cancel = true;
    }
    
    const queue = { cancel: false };
    window.currentPreloadQueue = queue;
    
    const overlay = document.getElementById("contour-loading-overlay");
    const textLabel = document.getElementById("contour-loading-text");
    const total = playbackTimesteps.length;
    
    if (overlay && textLabel) {
        overlay.style.display = "flex";
        textLabel.textContent = `Processing: 0 / ${total} frames`;
    }
    
    let index = 0;
    function loadNext() {
        if (queue.cancel) {
            if (overlay) overlay.style.display = "none";
            return;
        }
        
        if (index >= total) {
            if (overlay) overlay.style.display = "none";
            return;
        }
        
        const entry = playbackTimesteps[index];
        const img = new Image();
        img.onload = img.onerror = () => {
            if (queue.cancel) {
                if (overlay) overlay.style.display = "none";
                return;
            }
            index++;
            if (textLabel) {
                textLabel.textContent = `Processing: ${index} / ${total} frames`;
            }
            setTimeout(loadNext, 50); // Yield main thread
        };
        img.src = `/api/contour_image?var=${varName}&vtm=${entry.vtm}&mtime=${entry.mtime}`;
    }
    
    loadNext();
}

// Playback Timeline Controls
async function fetchPlaybackHistory() {
    try {
        const res = await fetch("/api/history");
        const data = await res.json();
        const oldTimesteps = playbackTimesteps;
        playbackTimesteps = data.timesteps || [];
        
        const slider = document.getElementById("playback-slider");
        const panel = document.getElementById("playback-panel");
        
        if (playbackTimesteps.length > 0) {
            panel.style.display = "flex";
            
            const oldIndex = playbackIndex;
            const wasAtEnd = (oldIndex === -1 || oldIndex === oldTimesteps.length - 1);
            
            slider.min = "0";
            slider.max = (playbackTimesteps.length - 1).toString();
            
            if (wasAtEnd) {
                playbackIndex = playbackTimesteps.length - 1;
                playbackActiveVtm = playbackTimesteps[playbackIndex].vtm;
            } else {
                // Keep the current playbackActiveVtm if it still exists
                let found = -1;
                for (let i = 0; i < playbackTimesteps.length; i++) {
                    if (playbackTimesteps[i].vtm === playbackActiveVtm) {
                        found = i;
                        break;
                    }
                }
                if (found !== -1) {
                    playbackIndex = found;
                } else {
                    playbackIndex = Math.min(oldIndex, playbackTimesteps.length - 1);
                    playbackActiveVtm = playbackTimesteps[playbackIndex].vtm;
                }
            }
            
            slider.value = playbackIndex.toString();
            document.getElementById("playback-time-label").textContent = "t = " + playbackTimesteps[playbackIndex].time.toFixed(3);
            
            // Preload images for current variable
            const varName = visualizerMode.replace("contour_", "");
            preloadContourImages(varName);
        } else {
            panel.style.display = "none";
            playbackActiveVtm = null;
            playbackIndex = -1;
        }
    } catch (err) {
        console.error("Error fetching history:", err);
    }
}

function togglePlayback() {
    const btn = document.getElementById("btn-play-pause");
    if (playbackInterval) {
        clearInterval(playbackInterval);
        playbackInterval = null;
        btn.textContent = "▶ Play";
    } else {
        btn.textContent = "⏸ Pause";
        playbackInterval = setInterval(() => {
            playbackIndex++;
            if (playbackIndex >= playbackTimesteps.length) {
                playbackIndex = 0;
            }
            document.getElementById("playback-slider").value = playbackIndex.toString();
            updatePlaybackView();
        }, 400);
    }
}

function stepPlayback(direction) {
    if (playbackInterval) {
        togglePlayback();
    }
    playbackIndex += direction;
    if (playbackIndex < 0) playbackIndex = 0;
    if (playbackIndex >= playbackTimesteps.length) playbackIndex = playbackTimesteps.length - 1;
    
    document.getElementById("playback-slider").value = playbackIndex.toString();
    updatePlaybackView();
}

function updatePlaybackView() {
    if (playbackIndex >= 0 && playbackIndex < playbackTimesteps.length) {
        const entry = playbackTimesteps[playbackIndex];
        playbackActiveVtm = entry.vtm;
        document.getElementById("playback-time-label").textContent = "t = " + entry.time.toFixed(3);
        draw();
    }
}

// Main controller orchestrating app bootstrapping, event bindings, and polling loops

import { state } from './state.js';
import {
    fetchConfig,
    saveConfig,
    runSimulation,
    stopSimulation,
    restartSimulation,
    cleanOutputs,
    clearCache,
    fetchStatus
} from './api.js';
import {
    calculateOverallBounds,
    draw
} from './canvas_renderer.js';
import { drawPlots } from './canvas_plotter.js';
import { runValidation, showValidationToast } from './validation.js';
import {
    populateFormFields,
    toggleIBShapeFields,
    toggleIBMethodFields,
    renderQuadsTable,
    renderPolysTable,
    renderProbesTable,
    updateProbesInConfig,
    collectFormFields,
    updateStatusUI,
    appendLogLine,
    saveContourSettings,
    openBlockEditorModal,
    openBCEditorModal,
    openContourSettingsModal
} from './ui.js';
import {
    fetchPlaybackHistory,
    togglePlayback,
    stepPlayback,
    updatePlaybackView
} from './playback.js';
import {
    onCanvasMouseMove,
    onCanvasClick,
    onCanvasDblClick
} from './mouse_handlers.js';

// Initialize Page components when DOM is fully built
window.addEventListener("DOMContentLoaded", () => {
    state.canvas = document.getElementById("grid-canvas");
    state.ctx = state.canvas.getContext("2d");
    
    initializeConfig();
    startStatusPolling();
    setupEventListeners();
    resizeCanvas();
});

window.addEventListener("resize", () => {
    resizeCanvas();
    calculateOverallBounds();
    draw();
});

// Resize visualizer canvas to fit container
function resizeCanvas() {
    const container = state.canvas.parentElement;
    state.canvas.width = container.clientWidth - 40;
    state.canvas.height = container.clientHeight - 80;
}

// Fetch domain and inputs configs from backend and initialize state
async function initializeConfig() {
    try {
        const data = await fetchConfig();
        state.inputs = data.inputs || {};
        state.domain = data.domain || {};
        
        // Parse blocks from domain
        state.blocks = [];
        for (const [key, val] of Object.entries(state.domain)) {
            if (key.startsWith("Block")) {
                const id = parseInt(key.replace("Block", ""));
                state.blocks.push({
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
        state.blocks.sort((a, b) => a.id - b.id);
        
        // Update document case path
        document.getElementById("active-case-dir").textContent = window.location.pathname === "/" ? "cases/default_case" : window.location.pathname;

        populateFormFields();
        calculateOverallBounds();
        draw();
    } catch (err) {
        console.error("Error fetching config:", err);
    }
}

// Simulation Execution Actions

async function triggerSimulationRun(clean) {
    if (state.isRunning) return;
    try {
        const data = await runSimulation(clean);
        if (data.status === "success") {
            appendLogLine(`[GUI] Solver started successfully (PID: ${data.pid})...`);
            state.isRunning = true;
            updateStatusUI(true);
            await initializeConfig();
        } else {
            appendLogLine(`[GUI] Run failed: ${data.message}`);
        }
    } catch (err) {
        appendLogLine(`[GUI] Connection error launching solver: ${err}`);
    }
}

async function stopSimulationRun() {
    try {
        const data = await stopSimulation();
        if (data.status === "success") {
            appendLogLine("[GUI] STOP signal sent to solver. Waiting for clean shutdown...");
        } else {
            appendLogLine("[GUI] Stop request failed: " + data.message);
        }
    } catch (err) {
        appendLogLine("[GUI] Error stopping solver: " + err);
    }
}

async function triggerRestart() {
    appendLogLine("[GUI] Triggering restart sequence (stopping run, updating config, restarting)...");
    try {
        const data = await restartSimulation();
        if (data.status === "success") {
            appendLogLine(`[GUI] Solver restarted successfully (PID: ${data.pid}) from t=${data.restart_time.toFixed(3)} using ${data.restart_file}`);
            state.isRunning = true;
            updateStatusUI(true);
            await initializeConfig();
        } else {
            appendLogLine(`[GUI] Restart failed: ${data.message}`);
            showValidationToast("Restart failed: " + data.message, true);
        }
    } catch (err) {
        appendLogLine(`[GUI] Connection error triggering restart: ${err}`);
    }
}

async function triggerCleanOnly() {
    if (state.isRunning) return;
    if (!confirm("Are you sure you want to clean all outputs? This will delete all VTK files and logs.")) return;
    try {
        const data = await cleanOutputs();
        if (data.status === "success") {
            appendLogLine("[GUI] Case outputs cleaned successfully.");
            // Reset local histories
            state.residualHistory = [];
            state.probeHistory = [];
            state.playbackTimesteps = [];
            state.playbackActiveVtm = null;
            state.playbackIndex = -1;
            drawPlots();
            
            // Hide playback panel
            const panel = document.getElementById("playback-panel");
            if (panel) panel.style.display = "none";
            
            draw();
            showValidationToast("Outputs cleaned successfully!", false);
            await initializeConfig();
        } else {
            showValidationToast("Failed to clean outputs: " + data.message, true);
        }
    } catch (err) {
        appendLogLine(`[GUI] Connection error cleaning outputs: ${err}`);
    }
}

async function triggerContourReload() {
    try {
        appendLogLine("[GUI] Clearing visualizer cache and reloading contours...");
        const data = await clearCache();
        if (data.status === "success") {
            showValidationToast("Cache cleared! Reloading...", false);
            await fetchPlaybackHistory();
            draw();
        } else {
            showValidationToast("Failed to clear cache: " + data.message, true);
        }
    } catch (err) {
        appendLogLine(`[GUI] Connection error reloading contours: ${err}`);
    }
}

// Poll state from Python server
function startStatusPolling() {
    state.statusInterval = setInterval(async () => {
        try {
            const data = await fetchStatus();
            
            // Check running status
            state.isRunning = data.running;
            updateStatusUI(state.isRunning);

            // Update terminal logs
            if (data.logs && data.logs.length > 0) {
                const term = document.getElementById("terminal-log");
                const newText = data.logs.join("\n");
                if (term.textContent !== newText) {
                    const isAtBottom = (term.scrollHeight - term.clientHeight - term.scrollTop) < 30;
                    term.textContent = newText;
                    if (isAtBottom) {
                        term.scrollTop = term.scrollHeight;
                    }
                }
            }

            // Save histories
            state.residualHistory = data.residuals || [];
            state.probeHistory = data.probes || [];

            // Draw line plots
            drawPlots();

            // Refresh flow contours if active
            if (state.visualizerMode.startsWith("contour_")) {
                await fetchPlaybackHistory();
                draw();
            }
        } catch (err) {
            console.error("Error polling status:", err);
        }
    }, 1000);
}

// Bind event listeners to DOM controls
function setupEventListeners() {
    // Config Save Action
    document.getElementById("btn-save-config").addEventListener("click", async () => {
        collectFormFields();
        if (!runValidation()) return;

        // Clear previous Block entries in domain
        for (const key of Object.keys(state.domain)) {
            if (key.startsWith("Block")) {
                delete state.domain[key];
            }
        }

        // Re-compile domain object
        state.blocks.forEach(b => {
            const key = `Block${b.id}`;
            state.domain[key] = {
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
            const data = await saveConfig({ inputs: state.inputs, domain: state.domain });
            if (data.status === "success") {
                showValidationToast("Configuration saved successfully!", false);
                initializeConfig();
            } else {
                showValidationToast("Error: " + data.message, true);
            }
        } catch (err) {
            showValidationToast("Failed to save configuration: " + err, true);
        }
    });

    // Form selection changes
    document.getElementById("ib-shape").addEventListener("change", toggleIBShapeFields);
    document.getElementById("ib-method").addEventListener("change", toggleIBMethodFields);

    // MULTI shape Add buttons
    const btnAddQuad = document.getElementById("btn-add-quad");
    if (btnAddQuad) {
        btnAddQuad.addEventListener("click", () => {
            state.quads.push("0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0");
            renderQuadsTable();
        });
    }
    const btnAddPoly = document.getElementById("btn-add-poly");
    if (btnAddPoly) {
        btnAddPoly.addEventListener("click", () => {
            state.polys.push({
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
            state.editingBlock = null;
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
            const varName = state.visualizerMode.replace("contour_", "");
            openContourSettingsModal(varName);
        });
    }

    // Contour Modal Buttons
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
            if (state.playbackInterval) {
                togglePlayback();
            }
            state.playbackIndex = parseInt(e.target.value);
            updatePlaybackView();
        });
    }

    // Sidebar Config Tabs switcher
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", e => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            state.activeTab = btn.getAttribute("data-tab");
            document.getElementById(`tab-${state.activeTab}`).classList.add("active");
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
        state.visualizerMode = e.target.value;
        
        const gearBtn = document.getElementById("btn-contour-gear");
        const reloadBtn = document.getElementById("btn-contour-reload");
        const playbackPanel = document.getElementById("playback-panel");
        
        if (state.playbackInterval) {
            clearInterval(state.playbackInterval);
            state.playbackInterval = null;
            const btn = document.getElementById("btn-play-pause");
            if (btn) btn.textContent = "▶ Play";
        }
        
        if (state.visualizerMode.startsWith("contour_")) {
            if (gearBtn) gearBtn.style.display = "inline-block";
            if (reloadBtn) reloadBtn.style.display = "inline-block";
            await fetchPlaybackHistory();
        } else {
            if (gearBtn) gearBtn.style.display = "none";
            if (reloadBtn) reloadBtn.style.display = "none";
            if (playbackPanel) playbackPanel.style.display = "none";
            state.playbackActiveVtm = null;
        }
        
        draw();
    });

    // Probe Mode toggle
    const probeModeBtn = document.getElementById("btn-probe-mode");
    probeModeBtn.addEventListener("click", () => {
        state.isProbeModeActive = !state.isProbeModeActive;
        if (state.isProbeModeActive) {
            probeModeBtn.textContent = "Cancel Probe";
            probeModeBtn.classList.add("active");
            state.canvas.parentElement.classList.add("probe-mode");
        } else {
            probeModeBtn.textContent = "Add Probe";
            probeModeBtn.classList.remove("active");
            state.canvas.parentElement.classList.remove("probe-mode");
        }
    });

    // Custom Probe form
    document.getElementById("btn-add-probe-form").addEventListener("click", () => {
        const name = prompt("Enter probe name (e.g. PROBE_MY):");
        if (!name) return;
        const x = parseFloat(prompt("Enter X coordinate:"));
        const y = parseFloat(prompt("Enter Y coordinate:"));
        const variable = prompt("Enter sampling variable (Density, Pressure, Mach, Sigma):", "Pressure");
        if (isNaN(x) || isNaN(y) || !variable) return;
        
        state.probes.push({ name, x, y, variable });
        renderProbesTable();
        updateProbesInConfig();
        draw();
    });

    // Simulation Execution Buttons
    document.getElementById("btn-run").addEventListener("click", () => triggerSimulationRun(false));
    document.getElementById("btn-restart").addEventListener("click", triggerRestart);
    document.getElementById("btn-clean").addEventListener("click", () => triggerSimulationRun(true));
    document.getElementById("btn-clean-only").addEventListener("click", triggerCleanOnly);
    document.getElementById("btn-stop").addEventListener("click", stopSimulationRun);
    
    const reloadBtn = document.getElementById("btn-contour-reload");
    if (reloadBtn) {
        reloadBtn.addEventListener("click", triggerContourReload);
    }

    // Canvas Mouse Listeners
    state.canvas.addEventListener("mousemove", onCanvasMouseMove);
    state.canvas.addEventListener("click", onCanvasClick);
    state.canvas.addEventListener("dblclick", onCanvasDblClick);

    // BC Modal buttons
    document.getElementById("btn-bc-save").addEventListener("click", () => {
        const selectVal = document.getElementById("bc-type-select").value;
        let finalBC = "";
        
        if (selectVal === "BLOCK_CONNECT") {
            const blockId = document.getElementById("bc-connect-block").value;
            const face = document.getElementById("bc-connect-face").value;
            finalBC = `${blockId}:${face}`;
        } else if (["CHARACTERISTIC", "INFLOW_SUPERSONIC", "STATIC_PRESSURE"].includes(selectVal)) {
            const stateBC = document.getElementById("bc-state-input").value;
            finalBC = `${selectVal}:${stateBC}`;
        } else {
            finalBC = selectVal;
        }
        
        const targetBlock = state.blocks.find(x => x.id === state.editingBC.blockId);
        if (targetBlock) {
            if (state.editingBC.face === "L") targetBlock.bcL = finalBC;
            else if (state.editingBC.face === "R") targetBlock.bcR = finalBC;
            else if (state.editingBC.face === "B") targetBlock.bcB = finalBC;
            else if (state.editingBC.face === "T") targetBlock.bcT = finalBC;
        }
        
        document.getElementById("modal-edit-bc").style.display = "none";
        draw();
    });

    document.getElementById("btn-bc-cancel").addEventListener("click", () => {
        document.getElementById("modal-edit-bc").style.display = "none";
    });

    // Block editor modal buttons
    document.getElementById("btn-eb-save").addEventListener("click", () => {
        const nx = parseInt(document.getElementById("eb-nx").value);
        const ny = parseInt(document.getElementById("eb-ny").value);
        const xMin = parseFloat(document.getElementById("eb-xmin").value);
        const xMax = parseFloat(document.getElementById("eb-xmax").value);
        const yMin = parseFloat(document.getElementById("eb-ymin").value);
        const yMax = parseFloat(document.getElementById("eb-ymax").value);
        
        if (state.editingBlock) {
            state.editingBlock.nx = nx;
            state.editingBlock.ny = ny;
            state.editingBlock.xMin = xMin;
            state.editingBlock.xMax = xMax;
            state.editingBlock.yMin = yMin;
            state.editingBlock.yMax = yMax;
            
            const blockName = `Block${state.editingBlock.id}`;
            if (state.domain[blockName]) {
                state.domain[blockName]["N_ELEM_X"] = nx.toString();
                state.domain[blockName]["N_ELEM_Y"] = ny.toString();
                state.domain[blockName]["X_MIN"] = xMin.toString();
                state.domain[blockName]["X_MAX"] = xMax.toString();
                state.domain[blockName]["Y_MIN"] = yMin.toString();
                state.domain[blockName]["Y_MAX"] = yMax.toString();
            }
        } else {
            let newId = 0;
            state.blocks.forEach(b => {
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
            state.blocks.push(newBlock);
            
            const blockName = `Block${newId}`;
            state.domain[blockName] = {
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
            if (state.editingBlock) {
                const idx = state.blocks.indexOf(state.editingBlock);
                if (idx !== -1) {
                    state.blocks.splice(idx, 1);
                }
                const blockName = `Block${state.editingBlock.id}`;
                delete state.domain[blockName];
            }
            document.getElementById("modal-edit-block").style.display = "none";
            calculateOverallBounds();
            draw();
        });
    }

    // Modal selection changes for BC Type
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
                document.getElementById("bc-state-input").value = "0.75"; // Default static pressure
            }
        } else {
            connectGroup.style.display = "none";
            stateGroup.style.display = "none";
        }
    });
}

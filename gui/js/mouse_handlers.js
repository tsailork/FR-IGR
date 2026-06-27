// Interactive canvas mouse interaction handlers

import { state } from './state.js';
import { mapCanvasToCoord, mapCoordToCanvas } from './canvas_renderer.js';
import { openBCEditorModal, openBlockEditorModal, renderProbesTable, updateProbesInConfig } from './ui.js';
import { showValidationToast } from './validation.js';

// Mouse Move: hover inspector
export function onCanvasMouseMove(e) {
    const rect = state.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    state.lastMousePos = { x: mx, y: my };
    
    if (state.visualizerMode.startsWith("contour_")) {
        handleContourHover(mx, my);
        return;
    }

    const coord = mapCanvasToCoord(mx, my);
    let hovered = null;
    const edgeTolerance = 12;

    for (const b of state.blocks) {
        const bLx = mapCoordToCanvas(b.xMin, 0).x;
        const bRx = mapCoordToCanvas(b.xMax, 0).x;
        const bBy = mapCoordToCanvas(0, b.yMin).y;
        const bTy = mapCoordToCanvas(0, b.yMax).y;

        // Mouse inside block?
        if (coord.x >= b.xMin && coord.x <= b.xMax && coord.y >= b.yMin && coord.y <= b.yMax) {
            hovered = { type: "block", blockId: b.id, block: b };

            // Check Left boundary
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

    state.hoveredElement = hovered;
    updateTooltip(mx, my);
}

// Tooltip display manager
export function updateTooltip(mx, my) {
    const tooltip = document.getElementById("canvas-tooltip");
    if (!state.hoveredElement) {
        tooltip.style.opacity = 0;
        return;
    }
    
    let content = "";
    if (state.hoveredElement.type === "block") {
        const b = state.hoveredElement.block;
        content = `<strong>Block ${b.id}</strong>\nElements: ${b.nx} x ${b.ny}\nBounds X: [${b.xMin.toFixed(2)}, ${b.xMax.toFixed(2)}]\nBounds Y: [${b.yMin.toFixed(2)}, ${b.yMax.toFixed(2)}]`;
    } else if (state.hoveredElement.type === "boundary") {
        const faceName = state.hoveredElement.face === "L" ? "Left" : state.hoveredElement.face === "R" ? "Right" : state.hoveredElement.face === "B" ? "Bottom" : "Top";
        content = `<strong>Boundary: Block ${state.hoveredElement.blockId} ${faceName} Face</strong>\nBC Type: ${state.hoveredElement.bc}`;
    } else if (state.hoveredElement.type === "contour") {
        content = `X: ${state.hoveredElement.x.toFixed(4)}\nY: ${state.hoveredElement.y.toFixed(4)}\nValue: ${state.hoveredElement.val.toFixed(5)}`;
    }
    
    tooltip.innerHTML = content.replace(/\n/g, "<br>");
    tooltip.style.left = (mx + 20) + "px";
    tooltip.style.top = (my + 10) + "px";
    tooltip.style.opacity = 0.95;
}

// Hover inspector on active flow contour plotter (used in canvas based contours)
export function handleContourHover(mx, my) {
    let bestVal = null;
    const coord = mapCanvasToCoord(mx, my);
    
    for (const block of state.latestVtsData) {
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
                    bestVal = block.values[idx0];
                    break;
                }
            }
            if (bestVal !== null) break;
        }
        if (bestVal !== null) break;
    }
    
    if (bestVal !== null) {
        state.hoveredElement = { type: "contour", x: coord.x, y: coord.y, val: bestVal };
    } else {
        state.hoveredElement = null;
    }
    updateTooltip(mx, my);
}

// Canvas Click: add probes or edit BCs
export function onCanvasClick(e) {
    if (state.isProbeModeActive) {
        // Place point probe
        const rect = state.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const coord = mapCanvasToCoord(mx, my);

        // Check if inside domain
        let inside = false;
        for (const b of state.blocks) {
            if (coord.x >= b.xMin && coord.x <= b.xMax && coord.y >= b.yMin && coord.y <= b.yMax) {
                inside = true;
                break;
            }
        }

        if (inside) {
            const name = prompt("Point probe placed! Enter unique name:", `PROBE_${state.probes.length + 1}`);
            if (name) {
                const variable = prompt("Enter variable name (Density, Pressure, Mach, Sigma):", "Pressure");
                if (variable) {
                    state.probes.push({ name, x: coord.x, y: coord.y, variable });
                    renderProbesTable();
                    updateProbesInConfig();
                    showValidationToast(`Probe added at physical (${coord.x.toFixed(3)}, ${coord.y.toFixed(3)})`, false);
                }
            }
        } else {
            showValidationToast("Cannot place probe: click must be inside the grid block bounds.");
        }
        
        // Deactivate probe mode
        document.getElementById("btn-probe-mode").click();
        return;
    }

    if (state.hoveredElement && state.hoveredElement.type === "boundary") {
        // Edit boundary condition click
        state.editingBC = { blockId: state.hoveredElement.blockId, face: state.hoveredElement.face, currentBC: state.hoveredElement.bc };
        openBCEditorModal();
    }
}

// Canvas Double Click: Edit block resolution
export function onCanvasDblClick() {
    if (state.hoveredElement && state.hoveredElement.type === "block") {
        state.editingBlock = state.hoveredElement.block;
        openBlockEditorModal();
    }
}

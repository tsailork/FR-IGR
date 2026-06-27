// Canvas coordinate mapping and visual element rendering logic

import { state } from './state.js';

// Calculate bounding box of the multi-block grid
export function calculateOverallBounds() {
    if (state.blocks.length === 0) {
        state.bounds = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
        return;
    }
    
    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;
    
    state.blocks.forEach(b => {
        xMin = Math.min(xMin, b.xMin);
        xMax = Math.max(xMax, b.xMax);
        yMin = Math.min(yMin, b.yMin);
        yMax = Math.max(yMax, b.yMax);
    });
    
    state.bounds = { xMin, xMax, yMin, yMax };
    
    // Fit to Canvas: Calculate scale and offsets
    const margin = 45;
    const drawW = state.canvas.width - margin * 2;
    const drawH = state.canvas.height - margin * 2;
    
    const domainW = xMax - xMin;
    const domainH = yMax - yMin;
    
    // Scale to fit maintaining aspect ratio
    state.scale = Math.min(drawW / domainW, drawH / domainH);
    
    // Center it on canvas
    state.offsetX = margin + (drawW - domainW * state.scale) / 2 - xMin * state.scale;
    // Remember canvas coordinates: Y increases down, physical coordinates Y increases up
    state.offsetY = state.canvas.height - margin - (drawH - domainH * state.scale) / 2 + yMin * state.scale;
}

// Map physical coordinate (x, y) to screen pixels (cx, cy)
export function mapCoordToCanvas(x, y) {
    const cx = state.offsetX + x * state.scale;
    const cy = state.offsetY - y * state.scale;
    return { x: cx, y: cy };
}

// Map screen pixels (cx, cy) back to physical coordinates (x, y)
export function mapCanvasToCoord(cx, cy) {
    const x = (cx - state.offsetX) / state.scale;
    const y = (state.offsetY - cy) / state.scale;
    return { x, y };
}

export async function draw() {
    const contourImg = document.getElementById("contour-image-view");
    const gridCanvas = document.getElementById("grid-canvas");
    const legend = document.getElementById("grid-legend");
    
    if (state.visualizerMode.startsWith("contour_")) {
        gridCanvas.style.display = "none";
        if (legend) legend.style.display = "none";
        contourImg.style.display = "block";
        
        const varName = state.visualizerMode.replace("contour_", "");
        let url = `/api/contour_image?var=${varName}`;
        if (state.playbackActiveVtm) {
            url += `&vtm=${state.playbackActiveVtm}`;
            const entry = state.playbackTimesteps.find(t => t.vtm === state.playbackActiveVtm);
            if (entry && entry.mtime) {
                url += `&mtime=${entry.mtime}`;
            }
        }
        if (contourImg.getAttribute("data-src") !== url) {
            contourImg.setAttribute("data-src", url);
            contourImg.src = url;
        }
        return;
    } else {
        contourImg.style.display = "none";
        gridCanvas.style.display = "block";
        if (legend) legend.style.display = "flex";
    }

    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);

    // Grid Visualizer Mode
    state.blocks.forEach(b => {
        const p1 = mapCoordToCanvas(b.xMin, b.yMin);
        const p2 = mapCoordToCanvas(b.xMax, b.yMax);
        
        const w = p2.x - p1.x;
        const h = p2.y - p1.y; // height will be negative because Y is flipped
        
        // Draw block fill
        state.ctx.fillStyle = "rgba(18, 22, 38, 0.5)";
        state.ctx.fillRect(p1.x, p1.y, w, h);
        
        // Draw internal mesh elements (thin dotted)
        state.ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        state.ctx.lineWidth = 0.5;
        
        // Vertical lines
        const dx = (b.xMax - b.xMin) / b.nx;
        for (let i = 1; i < b.nx; ++i) {
            const px = b.xMin + i * dx;
            const pt1 = mapCoordToCanvas(px, b.yMin);
            const pt2 = mapCoordToCanvas(px, b.yMax);
            state.ctx.beginPath();
            state.ctx.moveTo(pt1.x, pt1.y);
            state.ctx.lineTo(pt2.x, pt2.y);
            state.ctx.stroke();
        }
        // Horizontal lines
        const dy = (b.yMax - b.yMin) / b.ny;
        for (let j = 1; j < b.ny; ++j) {
            const py = b.yMin + j * dy;
            const pt1 = mapCoordToCanvas(b.xMin, py);
            const pt2 = mapCoordToCanvas(b.xMax, py);
            state.ctx.beginPath();
            state.ctx.moveTo(pt1.x, pt1.y);
            state.ctx.lineTo(pt2.x, pt2.y);
            state.ctx.stroke();
        }

        // Draw boundary edges
        const drawEdge = (x1, y1, x2, y2, bc) => {
            const pt1 = mapCoordToCanvas(x1, y1);
            const pt2 = mapCoordToCanvas(x2, y2);
            
            state.ctx.beginPath();
            state.ctx.moveTo(pt1.x, pt1.y);
            state.ctx.lineTo(pt2.x, pt2.y);
            
            // Set styles based on BC
            if (bc.includes(":") && !isNaN(parseInt(bc.split(":")[0]))) {
                // Connection BC (dashed cyan)
                state.ctx.strokeStyle = varColorHex("connection");
                state.ctx.setLineDash([4, 4]);
                state.ctx.lineWidth = 2.5;
            } else {
                state.ctx.setLineDash([]);
                state.ctx.lineWidth = 3.5;
                if (bc.startsWith("WALL")) {
                    state.ctx.strokeStyle = varColorHex("wall");
                } else if (bc.startsWith("INFLOW")) {
                    state.ctx.strokeStyle = varColorHex("inflow");
                } else if (bc.startsWith("CHARACTERISTIC")) {
                    state.ctx.strokeStyle = varColorHex("characteristic");
                } else {
                    state.ctx.strokeStyle = varColorHex("outflow"); // transmissive / outflow
                }
            }
            state.ctx.stroke();
            state.ctx.setLineDash([]); // Reset
        };

        drawEdge(b.xMin, b.yMin, b.xMin, b.yMax, b.bcL); // Left
        drawEdge(b.xMax, b.yMin, b.xMax, b.yMax, b.bcR); // Right
        drawEdge(b.xMin, b.yMin, b.xMax, b.yMin, b.bcB); // Bottom
        drawEdge(b.xMin, b.yMax, b.xMax, b.yMax, b.bcT); // Top

        // Draw block labels
        state.ctx.fillStyle = "rgba(230, 237, 243, 0.4)";
        state.ctx.font = "bold 13px 'Outfit'";
        state.ctx.fillText(`BLOCK ${b.id}`, p1.x + 10, p1.y + h + 20);
    });

    // Draw active connections pointers (arrows)
    drawBlockConnectionArrows();

    // Draw point probes
    state.probes.forEach(probe => {
        const pt = mapCoordToCanvas(probe.x, probe.y);
        state.ctx.fillStyle = varColorHex("characteristic");
        state.ctx.beginPath();
        state.ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        state.ctx.fill();
        state.ctx.strokeStyle = "#fff";
        state.ctx.lineWidth = 1;
        state.ctx.stroke();
        
        state.ctx.fillStyle = "#fff";
        state.ctx.font = "10px monospace";
        state.ctx.fillText(probe.name, pt.x + 8, pt.y + 3);
    });

    // Draw Cylinder/Airfoil solid boundary
    drawSolidGeometries(false);

    // Show Labels & Origin overlay
    const showLabels = document.getElementById("chk-show-labels") ? document.getElementById("chk-show-labels").checked : true;
    if (showLabels) {
        state.ctx.fillStyle = "rgba(0, 242, 254, 0.85)";
        state.ctx.font = "10px monospace";
        
        const corners = [
            { x: state.bounds.xMin, y: state.bounds.yMin, label: `(${state.bounds.xMin.toFixed(2)}, ${state.bounds.yMin.toFixed(2)})`, align: 'right', baseline: 'top', dx: -5, dy: 5 },
            { x: state.bounds.xMax, y: state.bounds.yMin, label: `(${state.bounds.xMax.toFixed(2)}, ${state.bounds.yMin.toFixed(2)})`, align: 'left', baseline: 'top', dx: 5, dy: 5 },
            { x: state.bounds.xMin, y: state.bounds.yMax, label: `(${state.bounds.xMin.toFixed(2)}, ${state.bounds.yMax.toFixed(2)})`, align: 'right', baseline: 'bottom', dx: -5, dy: -5 },
            { x: state.bounds.xMax, y: state.bounds.yMax, label: `(${state.bounds.xMax.toFixed(2)}, ${state.bounds.yMax.toFixed(2)})`, align: 'left', baseline: 'bottom', dx: 5, dy: -5 }
        ];
        
        corners.forEach(c => {
            const pt = mapCoordToCanvas(c.x, c.y);
            state.ctx.textAlign = c.align;
            state.ctx.textBaseline = c.baseline;
            state.ctx.fillText(c.label, pt.x + c.dx, pt.y + c.dy);
        });
        
        // Cyan crosshairs for physical (0,0) origin
        if (0 >= state.bounds.xMin && 0 <= state.bounds.xMax && 0 >= state.bounds.yMin && 0 <= state.bounds.yMax) {
            const pt0 = mapCoordToCanvas(0.0, 0.0);
            state.ctx.strokeStyle = "rgba(0, 242, 254, 0.75)";
            state.ctx.lineWidth = 1.0;
            state.ctx.beginPath();
            state.ctx.moveTo(pt0.x - 15, pt0.y);
            state.ctx.lineTo(pt0.x + 15, pt0.y);
            state.ctx.moveTo(pt0.x, pt0.y - 15);
            state.ctx.lineTo(pt0.x, pt0.y + 15);
            state.ctx.stroke();
            
            state.ctx.fillStyle = "rgba(0, 242, 254, 0.85)";
            state.ctx.textAlign = "left";
            state.ctx.textBaseline = "top";
            state.ctx.fillText(" (0,0)", pt0.x, pt0.y + 2);
        }
    }
}

// Render dynamic connection flowlines in grid mode
function drawBlockConnectionArrows() {
    state.blocks.forEach(b => {
        const checkConnection = (bcStr, faceName) => {
            if (bcStr.includes(":") && !isNaN(parseInt(bcStr.split(":")[0]))) {
                const parts = bcStr.split(":");
                const nid = parseInt(parts[0]);
                const nface = parts[1];
                const nb = state.blocks.find(x => x.id === nid);
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
                state.ctx.beginPath();
                state.ctx.moveTo(pt1.x, pt1.y);
                state.ctx.lineTo(pt2.x, pt2.y);
                state.ctx.strokeStyle = "rgba(0, 210, 255, 0.4)";
                state.ctx.lineWidth = 1.5;
                state.ctx.setLineDash([2, 4]);
                state.ctx.stroke();
                state.ctx.setLineDash([]);
            }
        };
        checkConnection(b.bcL, "L");
        checkConnection(b.bcR, "R");
        checkConnection(b.bcB, "B");
        checkConnection(b.bcT, "T");
    });
}

export function evaluateIBQ(t) {
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

// Render Solid Obstacles (cylinder / NACA airfoil profile)
export function drawSolidGeometries(outlineOnly) {
    const ibEnable = state.inputs["ImmersedBoundary"] ? state.inputs["ImmersedBoundary"]["ENABLE_IB"] === "true" : false;
    if (!ibEnable) return;
    
    const shape = state.inputs["ImmersedBoundary"]["IB_SHAPE"];
    const cx = parseFloat(state.inputs["ImmersedBoundary"]["IB_CENTER_X"] || 0.0);
    const cy = parseFloat(state.inputs["ImmersedBoundary"]["IB_CENTER_Y"] || 0.0);
    const radius = parseFloat(state.inputs["ImmersedBoundary"]["IB_RADIUS"] || 0.5);

    if (shape === "CIRCLE") {
        const pt = mapCoordToCanvas(cx, cy);
        const rPix = radius * state.scale;
        
        state.ctx.beginPath();
        state.ctx.arc(pt.x, pt.y, rPix, 0, Math.PI * 2);
        
        if (outlineOnly) {
            state.ctx.strokeStyle = varColorHex("red");
            state.ctx.lineWidth = 2.0;
            state.ctx.stroke();
        } else {
            state.ctx.fillStyle = "rgba(255, 59, 48, 0.25)";
            state.ctx.fill();
            state.ctx.strokeStyle = varColorHex("red");
            state.ctx.lineWidth = 1.5;
            state.ctx.stroke();
        }
    } else if (shape === "NACA") {
        const nacaCode = state.inputs["ImmersedBoundary"]["IB_NACA_CODE"] || "0012";
        const aoaDeg = parseFloat(state.inputs["ImmersedBoundary"]["IB_AOA"] || 0.0);
        const chord = radius; // Chord length maps to IB_RADIUS in parameters
        
        drawNacaAirfoil(cx, cy, chord, nacaCode, aoaDeg, outlineOnly);
    } else if (shape === "MULTI") {
        let t_eval = 0.0;
        if (state.playbackIndex >= 0 && state.playbackIndex < state.playbackTimesteps.length) {
            t_eval = state.playbackTimesteps[state.playbackIndex].time;
        } else {
            t_eval = parseFloat(state.inputs["IO"] ? state.inputs["IO"]["RESTART_TIME"] : 0.0) || 0.0;
        }
        const q_val = evaluateIBQ(t_eval);
        
        // 1. Draw Quads
        state.quads.forEach(q => {
            const parts = q.trim().split(/\s+/).map(parseFloat);
            if (parts.length === 8) {
                state.ctx.beginPath();
                let pt = mapCoordToCanvas(parts[0], parts[1]);
                state.ctx.moveTo(pt.x, pt.y);
                for (let j = 1; j < 4; j++) {
                    pt = mapCoordToCanvas(parts[j*2], parts[j*2 + 1]);
                    state.ctx.lineTo(pt.x, pt.y);
                }
                state.ctx.closePath();
                if (outlineOnly) {
                    state.ctx.strokeStyle = varColorHex("red");
                    state.ctx.lineWidth = 2.0;
                    state.ctx.stroke();
                } else {
                    state.ctx.fillStyle = "rgba(255, 59, 48, 0.25)";
                    state.ctx.fill();
                    state.ctx.strokeStyle = varColorHex("red");
                    state.ctx.lineWidth = 1.5;
                    state.ctx.stroke();
                }
            }
        });
        
        // 2. Draw Curves/Polys
        state.polys.forEach(p => {
            const a = (1.0 - q_val) * p.a0 + q_val * p.a1;
            const b = (1.0 - q_val) * p.b0 + q_val * p.b1;
            const c = (1.0 - q_val) * p.c0 + q_val * p.c1;
            
            const steps = 100;
            
            // Mask shading (if filled and bounds are valid)
            if (!outlineOnly) {
                state.ctx.fillStyle = "rgba(255, 59, 48, 0.15)";
                state.ctx.beginPath();
                let first = true;
                if (p.dir === "Y") {
                    for (let step = 0; step <= steps; step++) {
                        const px = state.bounds.xMin + (step / steps) * (state.bounds.xMax - state.bounds.xMin);
                        const py = a * px * px + b * px + c;
                        const pt = mapCoordToCanvas(px, py);
                        if (first) {
                            state.ctx.moveTo(pt.x, pt.y);
                            first = false;
                        } else {
                            state.ctx.lineTo(pt.x, pt.y);
                        }
                    }
                    const yEdge = p.side === "BELOW" ? state.bounds.yMin : state.bounds.yMax;
                    const ptR = mapCoordToCanvas(state.bounds.xMax, yEdge);
                    const ptL = mapCoordToCanvas(state.bounds.xMin, yEdge);
                    state.ctx.lineTo(ptR.x, ptR.y);
                    state.ctx.lineTo(ptL.x, ptL.y);
                } else { // dir === "X"
                    for (let step = 0; step <= steps; step++) {
                        const py = state.bounds.yMin + (step / steps) * (state.bounds.yMax - state.bounds.yMin);
                        const px = a * py * py + b * py + c;
                        const pt = mapCoordToCanvas(px, py);
                        if (first) {
                            state.ctx.moveTo(pt.x, pt.y);
                            first = false;
                        } else {
                            state.ctx.lineTo(pt.x, pt.y);
                        }
                    }
                    const xEdge = p.side === "LEFT" ? state.bounds.xMin : state.bounds.xMax;
                    const ptT = mapCoordToCanvas(xEdge, state.bounds.yMax);
                    const ptB = mapCoordToCanvas(xEdge, state.bounds.yMin);
                    state.ctx.lineTo(ptT.x, ptT.y);
                    state.ctx.lineTo(ptB.x, ptB.y);
                }
                state.ctx.closePath();
                state.ctx.fill();
            }
            
            // Curve Outline
            state.ctx.beginPath();
            let first = true;
            if (p.dir === "Y") {
                for (let step = 0; step <= steps; step++) {
                    const px = state.bounds.xMin + (step / steps) * (state.bounds.xMax - state.bounds.xMin);
                    const py = a * px * px + b * px + c;
                    const pt = mapCoordToCanvas(px, py);
                    if (first) {
                        state.ctx.moveTo(pt.x, pt.y);
                        first = false;
                    } else {
                        state.ctx.lineTo(pt.x, pt.y);
                    }
                }
            } else { // dir === "X"
                for (let step = 0; step <= steps; step++) {
                    const py = state.bounds.yMin + (step / steps) * (state.bounds.yMax - state.bounds.yMin);
                    const px = a * py * py + b * py + c;
                    const pt = mapCoordToCanvas(px, py);
                    if (first) {
                        state.ctx.moveTo(pt.x, pt.y);
                        first = false;
                    } else {
                        state.ctx.lineTo(pt.x, pt.y);
                    }
                }
            }
            state.ctx.strokeStyle = varColorHex("red");
            state.ctx.lineWidth = 2.0;
            state.ctx.stroke();
        });
    }
}

// Math solver to evaluate NACA profiles and draw outline
export function drawNacaAirfoil(leX, leY, chord, codeStr, aoaDeg, outlineOnly) {
    const t = parseFloat(codeStr.substring(2)) / 100.0; // Thickness (last 2 digits)
    const m = parseFloat(codeStr.substring(0, 1)) / 100.0; // Max Camber (1st digit)
    const p = parseFloat(codeStr.substring(1, 2)) / 10.0; // Camber Position (2nd digit)
    
    const pointsCount = 45;
    const upperPts = [];
    const lowerPts = [];
    const rad = -aoaDeg * Math.PI / 180.0; // Angle of attack rotation in radians

    for (let i = 0; i <= pointsCount; ++i) {
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

    state.ctx.beginPath();
    const startPt = mapCoordToCanvas(upperPts[0].x, upperPts[0].y);
    state.ctx.moveTo(startPt.x, startPt.y);
    
    // Draw upper surface
    for (let i = 1; i < upperPts.length; ++i) {
        const pt = mapCoordToCanvas(upperPts[i].x, upperPts[i].y);
        state.ctx.lineTo(pt.x, pt.y);
    }
    // Draw lower surface back to leading edge
    for (let i = lowerPts.length - 1; i >= 0; --i) {
        const pt = mapCoordToCanvas(lowerPts[i].x, lowerPts[i].y);
        state.ctx.lineTo(pt.x, pt.y);
    }
    state.ctx.closePath();

    if (outlineOnly) {
        state.ctx.strokeStyle = varColorHex("red");
        state.ctx.lineWidth = 2.0;
        state.ctx.stroke();
    } else {
        state.ctx.fillStyle = "rgba(255, 59, 48, 0.25)";
        state.ctx.fill();
        state.ctx.strokeStyle = varColorHex("red");
        state.ctx.lineWidth = 1.5;
        state.ctx.stroke();
    }
}

export function varColorHex(name) {
    if (name === "wall") return "#6e7681";
    if (name === "inflow") return "#39ff14";
    if (name === "outflow") return "#ff5e00";
    if (name === "characteristic") return "#ff007f";
    if (name === "connection") return "#00d2ff";
    if (name === "red") return "#ff3b30";
    return "#fff";
}

// Render flow heatmap contours directly using 2D canvas cells (defined but unused)
export function drawFlowContours() {
    state.latestVtsData.forEach(block => {
        const nx = block.nx;
        const ny = block.ny;
        
        for (let j = 0; j < ny - 1; ++j) {
            for (let i = 0; i < nx - 1; ++i) {
                const idx0 = j * nx + i;
                const idx1 = j * nx + (i + 1);
                const idx2 = (j + 1) * nx + (i + 1);
                const idx3 = (j + 1) * nx + i;
                
                const p0 = mapCoordToCanvas(block.x[idx0], block.y[idx0]);
                const p1 = mapCoordToCanvas(block.x[idx1], block.y[idx1]);
                const p2 = mapCoordToCanvas(block.x[idx2], block.y[idx2]);
                const p3 = mapCoordToCanvas(block.x[idx3], block.y[idx3]);
                
                const v0 = block.values[idx0];
                const v1 = block.values[idx1];
                const v2 = block.values[idx2];
                const v3 = block.values[idx3];
                
                // Triangle 1: p0, p1, p2
                const avgV1 = (v0 + v1 + v2) / 3.0;
                state.ctx.fillStyle = getColorForValue(avgV1);
                state.ctx.beginPath();
                state.ctx.moveTo(p0.x, p0.y);
                state.ctx.lineTo(p1.x, p1.y);
                state.ctx.lineTo(p2.x, p2.y);
                state.ctx.closePath();
                state.ctx.fill();
                
                // Triangle 2: p0, p2, p3
                const avgV2 = (v0 + v2 + v3) / 3.0;
                state.ctx.fillStyle = getColorForValue(avgV2);
                state.ctx.beginPath();
                state.ctx.moveTo(p0.x, p0.y);
                state.ctx.lineTo(p2.x, p2.y);
                state.ctx.lineTo(p3.x, p3.y);
                state.ctx.closePath();
                state.ctx.fill();
            }
        }
    });
}

export function getColorForValue(val) {
    const min = state.vtsMinMax.min;
    const max = state.vtsMinMax.max;
    
    let norm = (val - min) / (max - min);
    norm = Math.max(0.0, Math.min(1.0, norm));
    
    const hue = (1.0 - norm) * 240;
    return `hsl(${hue}, 100%, 45%)`;
}

// Diagnostic line charting implementation on native canvas

import { state } from './state.js';

export function drawPlots() {
    // 1. Draw Residuals Plot
    const rCanvas = document.getElementById("residuals-chart");
    if (rCanvas && rCanvas.offsetParent !== null) {
        const rCtx = rCanvas.getContext("2d");
        drawSingleLinePlot(rCanvas, rCtx, state.residualHistory, ["rho", "rhou", "rhov", "E"], ["#00d2ff", "#39ff14", "#ffcc00", "#ff007f"], "Residuals", true);
    }
    
    // 2. Draw Probes Plot
    const pCanvas = document.getElementById("probes-chart");
    if (pCanvas && pCanvas.offsetParent !== null && state.probeHistory.length > 0) {
        const pCtx = pCanvas.getContext("2d");
        const keys = Object.keys(state.probeHistory[0]).filter(k => k !== "Time");
        const colors = ["#00d2ff", "#39ff14", "#ffcc00", "#ff007f", "#ff5e00", "#a200ff"];
        drawSingleLinePlot(pCanvas, pCtx, state.probeHistory, keys, colors, "Probes", false);
    }
}

export function drawSingleLinePlot(canvasEl, ctxEl, history, keys, colors, title, isLogScale) {
    // Resize to fit element bounding container dynamically
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

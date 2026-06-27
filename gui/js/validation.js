// Client-side validation checks for block bounds, connectivity symmetry, and probes locations

import { state } from './state.js';

export function runValidation() {
    // 1. Validate element counts
    for (const b of state.blocks) {
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
    for (const b of state.blocks) {
        const checkFaceBC = (bcStr, faceName) => {
            if (bcStr.includes(":") && !isNaN(parseInt(bcStr.split(":")[0]))) {
                const parts = bcStr.split(":");
                const targetId = parseInt(parts[0]);
                const targetFace = parts[1];
                
                const targetBlock = state.blocks.find(x => x.id === targetId);
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
    for (const p of state.probes) {
        let inside = false;
        for (const b of state.blocks) {
            if (p.x >= b.xMin && p.x <= b.xMax && p.y >= b.yMin && p.y <= b.yMax) {
                inside = true;
                break;
            }
        }
        if (!inside) {
            showValidationToast(`Warning: Point probe '${p.name}' is located at (${p.x.toFixed(3)}, ${p.y.toFixed(3)}), which is outside the computational domain.`, false);
        }
    }

    return true;
}

export function showValidationToast(msg, isError = true) {
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

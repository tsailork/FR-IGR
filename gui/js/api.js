// API Gateway for communicating with the Python server endpoints

export async function fetchConfig() {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

export async function saveConfig(payload) {
    const res = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

export async function saveWebcontour(varName, settings) {
    const res = await fetch("/api/webcontour", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ var: varName, settings: settings })
    });
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

export async function runSimulation(clean) {
    const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clean })
    });
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

export async function stopSimulation() {
    const res = await fetch("/api/stop", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

export async function restartSimulation() {
    const res = await fetch("/api/restart", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

export async function cleanOutputs() {
    const res = await fetch("/api/clean", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

export async function clearCache() {
    const res = await fetch("/api/clear_cache", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

export async function fetchStatus() {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

export async function fetchHistory() {
    const res = await fetch("/api/history");
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

export async function fetchVTSData(varName) {
    const res = await fetch(`/api/vts_data?var=${varName}`);
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return await res.json();
}

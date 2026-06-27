// Playback caching and timeline preloading engine

import { state } from './state.js';
import { fetchHistory } from './api.js';
import { draw } from './canvas_renderer.js';

// Preload contour images to browser cache sequentially
export function preloadContourImages(varName) {
    if (!state.playbackTimesteps || !state.playbackTimesteps.length) return;
    
    // Cancel any ongoing preload queue
    if (window.currentPreloadQueue) {
        window.currentPreloadQueue.cancel = true;
    }
    
    const queue = { cancel: false };
    window.currentPreloadQueue = queue;
    
    const overlay = document.getElementById("contour-loading-overlay");
    const textLabel = document.getElementById("contour-loading-text");
    const total = state.playbackTimesteps.length;
    
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
        
        const entry = state.playbackTimesteps[index];
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
            setTimeout(loadNext, 50); // Yield main thread to prevent interface freeze
        };
        img.src = `/api/contour_image?var=${varName}&vtm=${entry.vtm}&mtime=${entry.mtime}`;
    }
    
    loadNext();
}

// Fetch historical output time steps from the server and update timeline controls
export async function fetchPlaybackHistory() {
    try {
        const data = await fetchHistory();
        const oldTimesteps = state.playbackTimesteps;
        state.playbackTimesteps = data.timesteps || [];
        
        const slider = document.getElementById("playback-slider");
        const panel = document.getElementById("playback-panel");
        
        if (state.playbackTimesteps.length > 0) {
            panel.style.display = "flex";
            
            const oldIndex = state.playbackIndex;
            const wasAtEnd = (oldIndex === -1 || oldIndex === oldTimesteps.length - 1);
            
            slider.min = "0";
            slider.max = (state.playbackTimesteps.length - 1).toString();
            
            if (wasAtEnd) {
                state.playbackIndex = state.playbackTimesteps.length - 1;
                state.playbackActiveVtm = state.playbackTimesteps[state.playbackIndex].vtm;
            } else {
                // Keep current active VTM if it still exists
                let found = -1;
                for (let i = 0; i < state.playbackTimesteps.length; i++) {
                    if (state.playbackTimesteps[i].vtm === state.playbackActiveVtm) {
                        found = i;
                        break;
                    }
                }
                if (found !== -1) {
                    state.playbackIndex = found;
                } else {
                    state.playbackIndex = Math.min(oldIndex, state.playbackTimesteps.length - 1);
                    state.playbackActiveVtm = state.playbackTimesteps[state.playbackIndex].vtm;
                }
            }
            
            slider.value = state.playbackIndex.toString();
            document.getElementById("playback-time-label").textContent = "t = " + state.playbackTimesteps[state.playbackIndex].time.toFixed(3);
            
            // Preload images for current variable
            const varName = state.visualizerMode.replace("contour_", "");
            preloadContourImages(varName);
        } else {
            panel.style.display = "none";
            state.playbackActiveVtm = null;
            state.playbackIndex = -1;
        }
    } catch (err) {
        console.error("Error fetching history:", err);
    }
}

export function togglePlayback() {
    const btn = document.getElementById("btn-play-pause");
    if (state.playbackInterval) {
        clearInterval(state.playbackInterval);
        state.playbackInterval = null;
        btn.textContent = "▶ Play";
    } else {
        btn.textContent = "⏸ Pause";
        state.playbackInterval = setInterval(() => {
            state.playbackIndex++;
            if (state.playbackIndex >= state.playbackTimesteps.length) {
                state.playbackIndex = 0;
            }
            document.getElementById("playback-slider").value = state.playbackIndex.toString();
            updatePlaybackView();
        }, 400);
    }
}

export function stepPlayback(direction) {
    if (state.playbackInterval) {
        togglePlayback();
    }
    state.playbackIndex += direction;
    if (state.playbackIndex < 0) state.playbackIndex = 0;
    if (state.playbackIndex >= state.playbackTimesteps.length) state.playbackIndex = state.playbackTimesteps.length - 1;
    
    document.getElementById("playback-slider").value = state.playbackIndex.toString();
    updatePlaybackView();
}

export function updatePlaybackView() {
    if (state.playbackIndex >= 0 && state.playbackIndex < state.playbackTimesteps.length) {
        const entry = state.playbackTimesteps[state.playbackIndex];
        state.playbackActiveVtm = entry.vtm;
        document.getElementById("playback-time-label").textContent = "t = " + entry.time.toFixed(3);
        draw();
    }
}

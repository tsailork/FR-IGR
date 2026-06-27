// Global Application State Container for FR-IGR Web GUI

export const state = {
    // Configuration options
    inputs: {},
    domain: {},
    blocks: [],
    probes: [],
    quads: [],
    polys: [],

    // Status & execution
    statusInterval: null,
    activeTab: "physics",
    visualizerMode: "grid", // 'grid', 'contour_rho', 'contour_press', 'contour_mach', 'contour_sigma', 'contour_phi'
    isRunning: false,
    isProbeModeActive: false,

    // Timeline playback settings
    playbackActiveVtm: null,
    playbackTimesteps: [],
    playbackIndex: -1,
    playbackInterval: null,
    contourSettings: {}, // cached settings by varName

    // Canvas drawing context (populated on initialization)
    canvas: null,
    ctx: null,
    scale: 1.0,
    offsetX: 0.0,
    offsetY: 0.0,
    bounds: { xMin: 0, xMax: 1, yMin: 0, yMax: 1 },
    lastMousePos: { x: 0, y: 0 },
    hoveredElement: null, // { type: 'boundary'/'block'/'contour', blockId, face/details }

    // Physical diagnostics data histories
    residualHistory: [],
    probeHistory: [],

    // Transient UI modals configurations
    editingBC: null,    // { blockId, face, currentBC }
    editingBlock: null, // block object references
    latestVtsData: [],  // parsed raw grid arrays
    vtsMinMax: { min: 0, max: 1 }
};

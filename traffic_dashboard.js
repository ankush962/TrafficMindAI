const DIRECTIONS = ["North", "East", "South", "West"];
const COLORS = { fixed: "#8398b2", queue: "#09a76c", ml: "#267cf3" };
const STRATEGIES = [
    { id: "fixed", label: "Fixed-Time", long: "Fixed-Time" },
    { id: "queue", label: "Queue-Based", long: "Queue-Based Adaptive" },
    { id: "ml", label: "ML-Assisted", long: "ML-Assisted Adaptive" },
];
const SCENARIOS = {
    balanced: {
        label: "Midday · balanced demand",
        description: "Balanced midday demand with two simulated ambulance calls from different roads.",
        rates: { North: 8, East: 6, South: 6, West: 8 },
        lanes: { North: 2, East: 2, South: 2, West: 2 },
        capacityFactor: 1,
        ambulanceCount: 2,
    },
    peak: {
        label: "Morning peak",
        description: "Morning rush demand is highest on the north and west approaches; two ambulances are scheduled.",
        rates: { North: 12, East: 8, South: 7, West: 12 },
        lanes: { North: 2, East: 2, South: 2, West: 2 },
        capacityFactor: 1,
        ambulanceCount: 2,
    },
    incident: {
        label: "East lane incident",
        description: "An incident closes one eastbound lane; two ambulances arrive from different roads.",
        rates: { North: 8, East: 14, South: 6, West: 8 },
        lanes: { North: 2, East: 1, South: 2, West: 2 },
        capacityFactor: 1,
        ambulanceCount: 2,
    },
    rain: {
        label: "Rain · evening peak",
        description: "Rain reduces discharge capacity by 25%; two ambulances arrive from different roads.",
        rates: { North: 11, East: 9, South: 8, West: 11 },
        lanes: { North: 2, East: 2, South: 2, West: 2 },
        capacityFactor: 0.75,
        ambulanceCount: 2,
    },
    emergency: {
        label: "Emergency response",
        description: "Four ambulances arrive from different roads. Each approach receives the next safe green in turn.",
        rates: { North: 8, East: 7, South: 6, West: 8 },
        lanes: { North: 2, East: 2, South: 2, West: 2 },
        capacityFactor: 1,
        ambulanceCount: 4,
    },
};
const STEP_SEC = 5;
const DEFAULT_LANES = 2;
const MAX_GREEN_STEPS = 9;
const MIN_GREEN_STEPS = 2;
const state = { results: null, arrivals: null, scenario: "balanced", selected: "queue", chartMode: "all", seed: 11, steps: 240, currentStep: 0, queueChartMax: 10, directionChartMax: 10, timer: null, playing: false };

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 1) => Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function seededRandom(seed) {
    let x = (Number(seed) >>> 0) || 1;
    return function () {
        x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
        return (x >>> 0) / 4294967296;
    };
}
function poisson(lambda, random) {
    const limit = Math.exp(-lambda);
    let product = 1, count = 0;
    do { count += 1; product *= random(); } while (product > limit);
    return count - 1;
}
function makeArrivals(steps, ratesPerMinute, seed) {
    const random = seededRandom(seed);
    const arrivals = Object.fromEntries(DIRECTIONS.map((d) => [d, []]));
    for (let step = 0; step < steps; step += 1) {
        for (const direction of DIRECTIONS) {
            const lambda = ratesPerMinute[direction] * STEP_SEC / 60;
            arrivals[direction].push(poisson(lambda, random));
        }
    }
    return arrivals;
}
function makeAmbulanceSchedule(scenario, steps) {
    const count = scenario.ambulanceCount || 2;
    const emergencyFractions = [0.05, 0.29, 0.53, 0.77];
    const routineFractions = [0.12, 0.62];
    const emergencyDirections = ["East", "North", "West", "South"];
    const routineDirections = ["East", "West"];
    const fractions = count > 2 ? emergencyFractions : routineFractions;
    const directions = count > 2 ? emergencyDirections : routineDirections;
    return Array.from({ length: count }, (_, index) => ({
        id: `AMB-${index + 1}`,
        direction: directions[index % directions.length],
        arrivalStep: clamp(Math.round(steps * fractions[index]), 1, steps - 1),
    }));
}
async function requestModelPredictions(observations) {
    const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ observations }),
    });
    if (!response.ok) {
        const details = await response.json().catch(() => ({}));
        throw new Error(details.error || `ML backend returned HTTP ${response.status}.`);
    }
    const result = await response.json();
    if (!Array.isArray(result.green_seconds) || result.green_seconds.length !== observations.length) {
        throw new Error("ML backend returned an invalid Random Forest response.");
    }
    return result.green_seconds.map(Number);
}
async function runController(type, arrivals, scenario, steps, ambulanceSchedule) {
    const queues = Object.fromEntries(DIRECTIONS.map((d) => [d, 0]));
    const waitSec = Object.fromEntries(DIRECTIONS.map((d) => [d, 0]));
    const history = Object.fromEntries(DIRECTIONS.map((d) => [d, [0, 0, 0, 0, 0]]));
    const serviceCredit = Object.fromEntries(DIRECTIONS.map((d) => [d, 0]));
    let active = "North";
    let remainingGreenSteps = 6;
    let clearanceRemaining = 0;
    let cumulativeDelay = 0;
    let lastReason = "Initial North phase: 30-second fixed green";
    const records = [];
    let servedTotal = 0;
    let arrivalTotal = 0;
    const ambulances = ambulanceSchedule.map((ambulance) => ({ ...ambulance, servedAtStep: null, greenStartedAtStep: null }));

    for (let step = 0; step < steps; step += 1) {
        for (const direction of DIRECTIONS) {
            const newAmbulances = ambulances.filter((ambulance) => ambulance.arrivalStep === step && ambulance.direction === direction);
            const regularIncoming = arrivals[direction][step] || 0;
            const incoming = regularIncoming + newAmbulances.length;
            queues[direction] += incoming;
            arrivalTotal += incoming;
            history[direction].push(regularIncoming);
            history[direction].shift();
        }

        const currentAmbulance = ambulances.find((ambulance) => step >= ambulance.arrivalStep && ambulance.servedAtStep === null) || null;
        const emergencyWaiting = Boolean(currentAmbulance);
        if (emergencyWaiting && active !== currentAmbulance.direction && clearanceRemaining === 0) {
            remainingGreenSteps = 0;
        }
        let served = 0;
        let clearance = false;
        let decision = step === 0;
        let allocation = step === 0 ? 30 : null;
        if (clearanceRemaining > 0) {
            clearance = true;
            clearanceRemaining -= 1;
        } else {
            if (remainingGreenSteps <= 0) {
                const previous = active;
                let allocatedGreenSec = 30;
                let modelPredictions = null;
                if (type === "fixed") {
                    active = DIRECTIONS[(DIRECTIONS.indexOf(active) + 1) % DIRECTIONS.length];
                    lastReason = emergencyWaiting
                        ? `Emergency override pauses the fixed schedule. ${currentAmbulance.direction} receives the next green after a safe clearance step.`
                        : `Fixed rotation gives ${active} a 30-second green.`;
                } else if (type === "queue") {
                    const scores = Object.fromEntries(DIRECTIONS.map((d) => [d, queues[d] + waitSec[d] * 0.04]));
                    active = DIRECTIONS.reduce((best, d) => scores[d] > scores[best] ? d : best, DIRECTIONS[0]);
                    allocatedGreenSec = clamp(10 + queues[active] * 1.2, 10, 45);
                    lastReason = `${active} has the strongest queue/fairness score (${scores[active].toFixed(1)}): ${queues[active]} vehicles queued and ${waitSec[active]}s since its last green.`;
                } else {
                    const observations = DIRECTIONS.map((d) => ({
                        direction: d,
                        queue_length: queues[d],
                        arrival_rate_veh_per_step: history[d].reduce((sum, n) => sum + n, 0) / history[d].length,
                        approach_wait_sec: waitSec[d],
                        lanes_open: scenario.lanes[d],
                    }));
                    const predictions = await requestModelPredictions(observations);
                    modelPredictions = Object.fromEntries(DIRECTIONS.map((d, index) => [d, predictions[index]]));
                    const scores = Object.fromEntries(DIRECTIONS.map((d) => [d, modelPredictions[d] + waitSec[d] * 0.05]));
                    active = DIRECTIONS.reduce((best, d) => scores[d] > scores[best] ? d : best, DIRECTIONS[0]);
                    allocatedGreenSec = modelPredictions[active];
                    lastReason = `The Random Forest recommends ${active}: ${allocatedGreenSec.toFixed(1)}s green, ${queues[active]} vehicles queued, ${waitSec[active]}s approach wait, ${scenario.lanes[active]} lane(s) open.`;
                }
                if (emergencyWaiting) {
                    active = currentAmbulance.direction;
                    allocatedGreenSec = type === "ml"
                        ? clamp(modelPredictions[active], 15, 35)
                        : type === "queue"
                            ? clamp(Math.max(15, 10 + queues[active] * 1.2), 15, 35)
                            : 30;
                    lastReason = `Ambulance ${currentAmbulance.id} detected from the ${active} road. ${active} receives the next safe green (${allocatedGreenSec.toFixed(1)}s).`;
                }
                remainingGreenSteps = clamp(Math.round(allocatedGreenSec / STEP_SEC), MIN_GREEN_STEPS, MAX_GREEN_STEPS);
                allocation = allocatedGreenSec;
                decision = true;
                if (active !== previous) {
                    clearance = true;
                    clearanceRemaining = 0; // One clearance step is represented by this transition step.
                }
            }
            if (!clearance) {
                const stepCapacity = 0.5 * STEP_SEC * scenario.lanes[active] * (scenario.capacityFactor || 1);
                serviceCredit[active] += stepCapacity;
                let canServe = Math.floor(serviceCredit[active] + 1e-9);
                let ambulanceClearedNow = false;
                if (emergencyWaiting && active === currentAmbulance.direction) {
                    currentAmbulance.greenStartedAtStep ??= step;
                    if (canServe > 0 && queues[active] > 0) {
                        served = 1;
                        serviceCredit[active] -= 1;
                        canServe -= 1;
                        currentAmbulance.servedAtStep = step;
                        ambulanceClearedNow = true;
                    }
                }
                const regularQueue = queues[active] - (ambulanceClearedNow ? 1 : 0);
                const regularServed = emergencyWaiting && active === currentAmbulance.direction && !ambulanceClearedNow
                    ? 0
                    : Math.min(regularQueue, canServe);
                served += regularServed;
                serviceCredit[active] -= regularServed;
                queues[active] -= served;
                if (queues[active] === 0) serviceCredit[active] = 0;
                remainingGreenSteps -= 1;
            }
        }

        servedTotal += served;
        const totalQueue = DIRECTIONS.reduce((sum, d) => sum + queues[d], 0);
        cumulativeDelay += totalQueue * STEP_SEC;
        for (const d of DIRECTIONS) {
            if (d === active && !clearance) waitSec[d] = 0;
            else waitSec[d] += STEP_SEC;
        }
        const ambulanceServedThisStep = ambulances.find((ambulance) => ambulance.servedAtStep === step) || null;
        const recentlyClearedAmbulance = ambulances.find((ambulance) => ambulance.servedAtStep !== null && step <= ambulance.servedAtStep + 3) || null;
        const emergencyFocus = currentAmbulance || ambulanceServedThisStep || recentlyClearedAmbulance;
        records.push({
            step,
            totalQueue,
            queueByDir: { ...queues },
            waitByDir: { ...waitSec },
            cumulativeDelay,
            served,
            active,
            clearance,
            decision,
            allocation,
            reason: lastReason,
            remainingGreenSec: clearance ? 0 : remainingGreenSteps * STEP_SEC,
            emergencyWaiting: Boolean(currentAmbulance && currentAmbulance.servedAtStep === null),
            emergencyPriority: Boolean(currentAmbulance || ambulanceServedThisStep),
            emergencyDirection: emergencyFocus?.direction || null,
            emergencyId: emergencyFocus?.id || null,
            emergencyVehicles: ambulances.map((ambulance) => ({
                ...ambulance,
                arrived: step >= ambulance.arrivalStep,
                waiting: step >= ambulance.arrivalStep && ambulance.servedAtStep === null,
                servedThisStep: ambulance.servedAtStep === step,
                recentlyCleared: ambulance.servedAtStep !== null && step <= ambulance.servedAtStep + 3,
            })),
        });
    }
    const avgTotalQueue = records.reduce((sum, r) => sum + r.totalQueue, 0) / records.length;
    const ambulancesServed = ambulances.filter((ambulance) => ambulance.servedAtStep !== null).length;
    const avgEmergencyDelaySec = ambulances.reduce((sum, ambulance) => {
        const endStep = ambulance.servedAtStep === null ? steps - 1 : ambulance.servedAtStep;
        return sum + Math.max(0, endStep - ambulance.arrivalStep) * STEP_SEC;
    }, 0) / Math.max(1, ambulances.length);
    return {
        records,
        metrics: {
            avgQueuePerApproach: avgTotalQueue / DIRECTIONS.length,
            finalQueue: records[records.length - 1].totalQueue,
            served: servedTotal,
            arrived: arrivalTotal,
            delayVehicleHours: records[records.length - 1].cumulativeDelay / 3600,
            emergencyDelaySec: avgEmergencyDelaySec,
            emergencyServed: ambulancesServed === ambulances.length,
            ambulancesServed,
            ambulanceCount: ambulances.length,
        },
    };
}
async function runSimulation(autoplay = true) {
    stopPlayback();
    $("runButton").disabled = true;
    $("runStatus").textContent = "Running Random Forest controller…";
    state.scenario = $("scenarioSelect").value;
    state.chartMode = $("strategySelect").value;
    if (state.chartMode !== "all") state.selected = state.chartMode;
    state.seed = clamp(parseInt($("seedInput").value, 10) || 11, 1, 999999);
    state.steps = clamp(parseInt($("stepsInput").value, 10) || 240, 60, 720);
    state.queueChartMax = 10;
    state.directionChartMax = 10;
    $("seedInput").value = state.seed;
    $("stepsInput").value = state.steps;
    const scenario = SCENARIOS[state.scenario];
    state.arrivals = makeArrivals(state.steps, scenario.rates, state.seed);
    const ambulanceSchedule = makeAmbulanceSchedule(scenario, state.steps);
    try {
        state.results = {
            fixed: await runController("fixed", state.arrivals, scenario, state.steps, ambulanceSchedule),
            queue: await runController("queue", state.arrivals, scenario, state.steps, ambulanceSchedule),
            ml: await runController("ml", state.arrivals, scenario, state.steps, ambulanceSchedule),
        };
        $("modelStatusTitle").textContent = "Random Forest connected";
        $("modelStatusDetail").textContent = "Loaded traffic_rf_model.pkl · Python backend";
        state.currentStep = 0;
        $("timelineRange").max = String(state.steps - 1);
        $("timelineRange").value = "0";
        $("runMeta").textContent = `Seed ${state.seed} · ${state.steps} steps · ${fmt(state.steps * STEP_SEC / 60, 1)} minutes`;
        $("comparisonScenario").textContent = `${scenario.label} · ${scenario.description}`;
        renderDashboard();
        if (autoplay) startPlayback();
        toastMessage("Simulation complete · ML decisions came from your Random Forest model.");
    } catch (error) {
        $("modelStatusTitle").textContent = "Random Forest not connected";
        $("modelStatusDetail").textContent = "Start traffic_backend.py and open the dashboard at http://127.0.0.1:8765";
        $("runStatus").textContent = "Python backend connection required";
        toastMessage(error.message || "Could not reach the Random Forest backend.");
    } finally {
        $("runButton").disabled = false;
    }
}
function avgQueueByDirection(result, direction) {
    return result.records.reduce((sum, r) => sum + r.queueByDir[direction], 0) / result.records.length;
}
function avgQueueByDirectionThrough(result, direction, step) {
    const records = result.records.slice(0, step + 1);
    if (!records.length) return 0;
    return records.reduce((sum, record) => sum + record.queueByDir[direction], 0) / records.length;
}
function chartStrategies() {
    return state.chartMode === "all" ? STRATEGIES : STRATEGIES.filter((strategy) => strategy.id === state.chartMode);
}
function syncChartLegends() {
    document.querySelectorAll("[data-chart-strategy]").forEach((item) => {
        item.hidden = state.chartMode !== "all" && item.dataset.chartStrategy !== state.chartMode;
    });
}
function bestStrategy() {
    return STRATEGIES.reduce((best, item) => state.results[item.id].metrics.avgQueuePerApproach < state.results[best].metrics.avgQueuePerApproach ? item.id : best, STRATEGIES[0].id);
}
function pctChange(value, baseline) {
    if (!baseline) return 0;
    return ((baseline - value) / baseline) * 100;
}
function setTrend(id, text, good = true) {
    const el = $(id); el.textContent = text; el.classList.toggle("bad", !good);
}
function renderKpis() {
    const selected = state.results[state.selected].metrics;
    const fixed = state.results.fixed.metrics;
    const queuePct = pctChange(selected.avgQueuePerApproach, fixed.avgQueuePerApproach);
    const finalPct = pctChange(selected.finalQueue, fixed.finalQueue);
    const servedPct = fixed.served ? ((selected.served - fixed.served) / fixed.served) * 100 : 0;
    $("avgQueueValue").textContent = fmt(selected.avgQueuePerApproach, 1);
    $("finalQueueValue").textContent = fmt(selected.finalQueue, 0);
    $("servedValue").textContent = fmt(selected.served, 0);
    $("delayValue").textContent = fmt(selected.delayVehicleHours, 2);
    if (state.selected === "fixed") {
        setTrend("avgQueueTrend", "Fixed-time baseline", true);
        setTrend("finalQueueTrend", "Baseline result", true);
        setTrend("servedTrend", "Reference for comparison", true);
        setTrend("delayTrend", "Baseline vehicle-hours", true);
    } else {
        setTrend("avgQueueTrend", `${queuePct >= 0 ? "↓" : "↑"} ${Math.abs(queuePct).toFixed(1)}% vs. fixed`, queuePct >= 0);
        setTrend("finalQueueTrend", `${finalPct >= 0 ? "↓" : "↑"} ${Math.abs(finalPct).toFixed(1)}% vs. fixed`, finalPct >= 0);
        setTrend("servedTrend", `${servedPct >= 0 ? "↑" : "↓"} ${Math.abs(servedPct).toFixed(1)}% vs. fixed`, servedPct >= 0);
        const delayPct = pctChange(selected.delayVehicleHours, fixed.delayVehicleHours);
        setTrend("delayTrend", `${delayPct >= 0 ? "↓" : "↑"} ${Math.abs(delayPct).toFixed(1)}% vs. fixed`, delayPct >= 0);
    }
}
function renderTables() {
    const winner = bestStrategy();
    const labels = { fixed: "Fixed-Time", queue: "Queue-Based Adaptive", ml: "ML-Assisted Adaptive" };
    const fixedQueue = state.results.fixed.metrics.avgQueuePerApproach;
    const ranked = [...STRATEGIES].sort((a, b) => state.results[a.id].metrics.avgQueuePerApproach - state.results[b.id].metrics.avgQueuePerApproach);
    const body = ranked.map((s, index) => {
        const m = state.results[s.id].metrics;
        const queueChange = pctChange(m.avgQueuePerApproach, fixedQueue);
        const changeText = s.id === "fixed" ? "Baseline" : `${queueChange >= 0 ? "↓" : "↑"} ${Math.abs(queueChange).toFixed(1)}%`;
        const changeClass = s.id === "fixed" ? "baseline" : (queueChange >= 0 ? "positive" : "negative");
        const rank = `<span class="rank-chip ${index === 0 ? "first" : ""}">#${index + 1}</span>`;
        const recommendation = s.id === winner ? '<span class="recommend-pill">Lowest average queue</span>' : "";
        const emergencyDelay = `${m.emergencyServed ? "" : "≥"}${fmt(m.emergencyDelaySec, 0)}s`;
        return `<tr class="${s.id === winner ? "recommended" : ""}"><td><span class="strategy-name"><i class="strategy-mark ${s.id}"></i>${labels[s.id]}</span>${rank}${recommendation}</td><td>${fmt(m.avgQueuePerApproach, 2)}</td><td>${fmt(m.finalQueue, 0)}</td><td>${fmt(m.served, 0)}</td><td>${fmt(m.delayVehicleHours, 2)} vh</td><td>${m.ambulancesServed}/${m.ambulanceCount}</td><td>${emergencyDelay}</td><td><span class="delta-tag ${changeClass}">${changeText}</span></td></tr>`;
    }).join("");
    $("comparisonBody").innerHTML = body;
    $("directionBody").innerHTML = DIRECTIONS.map((d) => `<tr><td><span class="dir-dot dir-${d.toLowerCase()}"></span>${d}</td><td>${fmt(avgQueueByDirection(state.results.fixed, d), 1)}</td><td>${fmt(avgQueueByDirection(state.results.queue, d), 1)}</td><td>${fmt(avgQueueByDirection(state.results.ml, d), 1)}</td></tr>`).join("");
    const winningMetrics = state.results[winner].metrics;
    const winningLabel = labels[winner];
    const queueImprovement = pctChange(winningMetrics.avgQueuePerApproach, fixedQueue);
    const delayImprovement = pctChange(winningMetrics.delayVehicleHours, state.results.fixed.metrics.delayVehicleHours);
    const fixedMetrics = state.results.fixed.metrics;
    const emergencyComparison = ` Ambulances: ${winningMetrics.ambulancesServed}/${winningMetrics.ambulanceCount} cleared with ${fmt(winningMetrics.emergencyDelaySec, 0)}s average response delay for ${winningLabel}; fixed-time cleared ${fixedMetrics.ambulancesServed}/${fixedMetrics.ambulanceCount} with ${fmt(fixedMetrics.emergencyDelaySec, 0)}s average delay.`;
    $("bestStrategyLabel").textContent = `${winningLabel} produced the lowest average queue`;
    if (winner === "fixed") {
        $("outcomeSummary").textContent = `The fixed-time baseline had the lowest average queue in this scenario. Neither adaptive controller improved this measure.${emergencyComparison}`;
        $("improvementValue").textContent = "Baseline result";
        $("improvementValue").className = "neutral";
    } else {
        const delayText = `${Math.abs(delayImprovement).toFixed(1)}% ${delayImprovement >= 0 ? "lower" : "higher"} accumulated vehicle delay`;
        $("outcomeSummary").textContent = `Across the full ${fmt(state.steps * STEP_SEC / 60, 1)}-minute run, average queue was ${Math.abs(queueImprovement).toFixed(1)}% lower than fixed-time; ${delayText}.${emergencyComparison}`;
        $("improvementValue").textContent = `${Math.abs(queueImprovement).toFixed(1)}% ${queueImprovement >= 0 ? "lower" : "higher"}`;
        $("improvementValue").className = `summary-result ${queueImprovement >= 0 ? "positive" : "negative"}`;
    }
}
function getCanvasContext(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(250, rect.width);
    const height = Math.max(180, rect.height);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
}
function drawQueueChart() {
    const canvas = $("queueChart");
    const { ctx, width: w, height: h } = getCanvasContext(canvas);
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 39, r: 8, t: 12, b: 30 };
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    const shownStrategies = chartStrategies();
    const visibleQueues = shownStrategies.flatMap((s) => state.results[s.id].records.slice(0, state.currentStep + 1).map((r) => r.totalQueue));
    state.queueChartMax = Math.max(state.queueChartMax, Math.ceil(Math.max(10, ...visibleQueues) / 50) * 50);
    const max = state.queueChartMax;
    ctx.font = "9px Inter, system-ui, sans-serif";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
        const y = pad.t + ph * i / 4;
        const value = Math.round(max * (4 - i) / 4);
        ctx.strokeStyle = "#eaf1f8"; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        ctx.fillStyle = "#8197b0"; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(String(value), pad.l - 6, y);
    }
    ctx.strokeStyle = "#c7d9eb"; ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ph); ctx.lineTo(w - pad.r, pad.t + ph); ctx.stroke();
    for (let i = 0; i <= 4; i += 1) {
        const x = pad.l + pw * i / 4;
        const minute = state.steps * STEP_SEC / 60 * i / 4;
        ctx.fillStyle = "#8197b0"; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(minute.toFixed(0), x, pad.t + ph + 6);
    }
    for (const strategy of shownStrategies) {
        const rows = state.results[strategy.id].records.slice(0, state.currentStep + 1);
        ctx.strokeStyle = COLORS[strategy.id]; ctx.lineWidth = strategy.id === state.selected ? 2.5 : 1.8;
        ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.beginPath();
        rows.forEach((r, index) => {
            const x = pad.l + pw * index / Math.max(1, state.steps - 1);
            const y = pad.t + ph * (1 - r.totalQueue / max);
            if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        if (rows.length) {
            const latest = rows[rows.length - 1];
            const x = pad.l + pw * (rows.length - 1) / Math.max(1, state.steps - 1);
            const y = pad.t + ph * (1 - latest.totalQueue / max);
            ctx.fillStyle = COLORS[strategy.id];
            ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
        }
    }
    const playX = pad.l + pw * state.currentStep / Math.max(1, state.steps - 1);
    ctx.strokeStyle = "rgba(22,119,248,.6)"; ctx.setLineDash([3, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(playX, pad.t); ctx.lineTo(playX, pad.t + ph); ctx.stroke(); ctx.setLineDash([]);
    $("chartCurrentTime").textContent = `Playhead at ${clockLabel(state.currentStep * STEP_SEC)}`;
}
function drawDirectionChart() {
    const canvas = $("directionChart");
    const { ctx, width: w, height: h } = getCanvasContext(canvas);
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 36, r: 8, t: 12, b: 29 };
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    const shownStrategies = chartStrategies();
    const values = shownStrategies.flatMap((s) => DIRECTIONS.map((d) => avgQueueByDirectionThrough(state.results[s.id], d, state.currentStep)));
    state.directionChartMax = Math.max(state.directionChartMax, Math.ceil(Math.max(10, ...values) / 20) * 20);
    const max = state.directionChartMax;
    ctx.font = "9px Inter, system-ui, sans-serif";
    for (let i = 0; i <= 4; i += 1) {
        const y = pad.t + ph * i / 4;
        ctx.strokeStyle = "#eaf1f8"; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        ctx.fillStyle = "#8197b0"; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(String(Math.round(max * (4 - i) / 4)), pad.l - 5, y);
    }
    ctx.strokeStyle = "#c7d9eb"; ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ph); ctx.lineTo(w - pad.r, pad.t + ph); ctx.stroke();
    const groupWidth = pw / DIRECTIONS.length;
    const barWidth = shownStrategies.length === 1 ? Math.min(24, groupWidth * .38) : Math.min(15, groupWidth * .18);
    DIRECTIONS.forEach((d, group) => {
        shownStrategies.forEach((s, si) => {
            const v = avgQueueByDirectionThrough(state.results[s.id], d, state.currentStep);
            const barH = ph * v / max;
            const center = pad.l + groupWidth * (group + .5);
            const offset = (si - (shownStrategies.length - 1) / 2) * (barWidth + 3);
            const x = center + offset - barWidth / 2;
            ctx.fillStyle = COLORS[s.id];
            roundRect(ctx, x, pad.t + ph - barH, barWidth, barH, 3); ctx.fill();
        });
        ctx.fillStyle = "#6f88a5"; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(d, pad.l + groupWidth * (group + .5), pad.t + ph + 6);
    });
}
function roundRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r); ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r); ctx.closePath();
}
function clockLabel(seconds) {
    const mins = Math.floor(seconds / 60), secs = seconds % 60;
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}
function svgCar(direction, index) {
    const fills = ["#2487ef", "#0ba879", "#7666e8", "#ef9e26", "#18a6c8", "#486f99"];
    const fill = fills[index % fills.length];
    let x, y, w, h;
    if (direction === "North") { x = 280; y = 9 + index * 22; w = 19; h = 16; }
    else if (direction === "South") { x = 318; y = 314 - index * 22; w = 19; h = 16; }
    else if (direction === "West") { x = 9 + index * 23; y = 153; w = 18; h = 12; }
    else { x = 573 - index * 23; y = 177; w = 18; h = 12; }
    return `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${fill}" stroke="#fff" stroke-width="1.8"/><rect x="${x + w * .28}" y="${y + h * .18}" width="${w * .44}" height="${h * .26}" rx="2" fill="#dff3ff" opacity=".9"/><circle cx="${x + 4}" cy="${y + h - 1}" r="1.6" fill="#32475a"/><circle cx="${x + w - 4}" cy="${y + h - 1}" r="1.6" fill="#32475a"/></g>`;
}
function renderEmergencyPriority(record) {
    const focus = record.emergencyVehicles.find((ambulance) => ambulance.id === record.emergencyId) || null;
    const visibleAmbulances = record.emergencyVehicles.filter((ambulance) => ambulance.arrived && (ambulance.waiting || ambulance.recentlyCleared));
    const directionStacks = {};
    const banner = $("emergencyBanner");
    const layer = $("emergencyVehiclesLayer");
    banner.hidden = !focus;
    layer.innerHTML = visibleAmbulances.map((ambulance) => {
        const direction = ambulance.direction;
        const stack = directionStacks[direction] || 0;
        directionStacks[direction] = stack + 1;
        const status = ambulance.waiting ? "priority" : "cleared";
        return `<div class="emergency-vehicle-marker from-${direction.toLowerCase()} ${ambulance.waiting ? "is-priority" : "is-cleared"}" style="--marker-offset:${stack * 22}px" role="img" aria-label="Ambulance ${ambulance.id} from ${direction} road, ${status}"><svg class="ambulance-vehicle-icon" viewBox="0 0 40 24" aria-hidden="true"><rect x="3" y="7" width="25" height="12" rx="2.5" fill="#fff" stroke="#a62e38" stroke-width="1.4"/><path d="M28 10h5l4 4v5H28z" fill="#fff" stroke="#a62e38" stroke-width="1.4"/><path d="M30 11h3l2.4 2.5H30z" fill="#9bdcff"/><rect x="6" y="11" width="12" height="2.2" rx="1" fill="#dc3447"/><path d="M10 8.8h2v2h2v2h-2v2h-2v-2H8v-2h2z" fill="#fff"/><circle cx="10" cy="19" r="2.5" fill="#24364a"/><circle cx="10" cy="19" r="1" fill="#d7e3ed"/><circle cx="31" cy="19" r="2.5" fill="#24364a"/><circle cx="31" cy="19" r="1" fill="#d7e3ed"/><rect x="15" y="4.5" width="4" height="2.4" rx="1" fill="#2588ef"/><rect x="20" y="4.5" width="4" height="2.4" rx="1" fill="#ef4554"/></svg><span>FROM ${direction.toUpperCase()} ROAD</span></div>`;
    }).join("");
    banner.classList.remove("priority-next", "priority-green");
    if (!focus) return;

    const direction = focus.direction;
    $("emergencyBannerTitle").textContent = record.emergencyWaiting
        ? `AMBULANCE ${focus.id} · FROM ${direction.toUpperCase()} ROAD`
        : `AMBULANCE ${focus.id} CLEARED · ${direction.toUpperCase()} GREEN`;
    if (record.emergencyWaiting && record.clearance && record.active === direction) {
        banner.classList.add("priority-next");
        $("emergencyBannerText").textContent = `All signals are red for a safe clearance step. ${direction} road is next for green.`;
        $("emergencyBannerState").textContent = "NEXT GREEN";
    } else if (record.active === direction && !record.clearance) {
        banner.classList.add("priority-green");
        $("emergencyBannerText").textContent = record.emergencyWaiting
            ? `${direction} signal is green. The ambulance is highlighted on its incoming road and clears first.`
            : `The ambulance cleared. The ${direction} signal now follows the selected controller.`;
        $("emergencyBannerState").textContent = record.emergencyWaiting ? "EMS GREEN" : "CLEARED";
    } else {
        banner.classList.add("priority-next");
        $("emergencyBannerText").textContent = `Emergency priority is active. The next safe green is assigned to the ${direction} road.`;
        $("emergencyBannerState").textContent = "NEXT GREEN";
    }
}
function renderIntersection() {
    const record = state.results[state.selected].records[state.currentStep];
    if (!record) return;
    const emergencyWaiting = record.emergencyWaiting;
    const emergencyPriority = record.emergencyPriority;
    const emergencyDirection = record.emergencyDirection;
    const ambulancesWaitingByDirection = Object.fromEntries(DIRECTIONS.map((direction) => [
        direction,
        record.emergencyVehicles.filter((ambulance) => ambulance.arrived && ambulance.waiting && ambulance.direction === direction).length,
    ]));
    for (const d of DIRECTIONS) {
        $(`map${d}`).textContent = record.queueByDir[d];
    }
    const carParts = [];
    for (const d of DIRECTIONS) {
        const normalVehicles = record.queueByDir[d] - ambulancesWaitingByDirection[d];
        const count = Math.min(6, Math.max(0, normalVehicles));
        for (let i = 0; i < count; i += 1) carParts.push(svgCar(d, i));
        const bulb = $(`sig${d}`);
        const isGreen = !record.clearance && record.active === d;
        bulb.setAttribute("fill", isGreen ? "#13bd79" : "#f05458");
        const priorityNext = emergencyWaiting && d === emergencyDirection && record.clearance;
        const priorityGreen = emergencyPriority && d === emergencyDirection && isGreen;
        bulb.classList.toggle("priority-green", priorityGreen);
        const signalRow = document.querySelector(`.signal-row[data-dir="${d}"]`);
        signalRow.classList.toggle("emergency-next", priorityNext);
        signalRow.classList.toggle("emergency-green", priorityGreen);
        signalRow.setAttribute("aria-label", priorityNext
            ? `${d} signal is red; next green for emergency vehicle`
            : priorityGreen
                ? `${d} signal is green for emergency vehicle`
                : `${d} signal ${isGreen ? "green" : "red"}`);
    }
    $("carsLayer").innerHTML = carParts.join("");
    document.querySelectorAll(".signal-row").forEach((row) => {
        const d = row.dataset.dir;
        const active = !record.clearance && record.active === d;
        const dot = row.querySelector(".signal-dot");
        const label = row.querySelector(".signal-state");
        const timer = row.querySelector(".signal-timer");
        const priorityNext = row.classList.contains("emergency-next");
        const priorityGreen = row.classList.contains("emergency-green");
        dot.className = `signal-dot ${active ? "green" : "red"}`;
        label.className = `signal-state ${active ? "green" : priorityNext ? "next" : "red"}`;
        label.textContent = priorityGreen ? (record.emergencyWaiting ? "EMS GREEN" : "EMS PASSED") : priorityNext ? "NEXT GREEN" : active ? "Green" : "Red";
        timer.textContent = active ? `${record.remainingGreenSec}s` : `${record.waitByDir[d]}s`;
    });
    renderEmergencyPriority(record);
    $("phaseStatus").textContent = emergencyWaiting && record.clearance
        ? `ALL RED · ${emergencyDirection.toUpperCase()} NEXT`
        : emergencyPriority && record.active === emergencyDirection
            ? (emergencyWaiting ? `${emergencyDirection.toUpperCase()} EMS GREEN` : `EMS PASSED · ${emergencyDirection.toUpperCase()} GREEN`)
            : record.clearance ? "ALL RED · PHASE CHANGE" : `${record.remainingGreenSec}s left`;
    $("decisionReason").textContent = record.reason;
    $("junctionTag").textContent = `${STRATEGIES.find((s) => s.id === state.selected).long} · current state`;
    $("currentStrategyLabel").textContent = STRATEGIES.find((s) => s.id === state.selected).long;
    $("timelineRange").value = String(state.currentStep);
    const currentSeconds = state.currentStep * STEP_SEC;
    const totalSeconds = state.steps * STEP_SEC;
    $("timeReadout").textContent = `${clockLabel(currentSeconds)} / ${clockLabel(totalSeconds)}`;
    $("runStatus").textContent = state.playing ? "Simulation running" : (state.currentStep >= state.steps - 1 ? "Simulation complete" : "Ready · timeline paused");
    $("playButton").innerHTML = `<svg class="icon"><use href="#${state.playing ? "i-pause" : "i-play"}"/></svg>`;
    $("playButton").setAttribute("aria-label", state.playing ? "Pause simulation" : "Play simulation");
    drawQueueChart();
    drawDirectionChart();
}
function renderDashboard() {
    if (!state.results) return;
    syncChartLegends();
    renderKpis(); renderTables(); renderIntersection();
    $("runStatus").textContent = state.playing ? "Simulation running" : "Ready · timeline paused";
}
function setCurrentStep(step) {
    state.currentStep = clamp(step, 0, state.steps - 1);
    renderIntersection();
}
function startPlayback() {
    if (!state.results) return;
    if (state.currentStep >= state.steps - 1) state.currentStep = 0;
    state.playing = true;
    renderIntersection();
    state.timer = window.setInterval(() => {
        if (state.currentStep >= state.steps - 1) { stopPlayback(); renderIntersection(); return; }
        state.currentStep += 1;
        renderIntersection();
    }, parseInt($("playbackSpeed").value, 10) || 300);
}
function stopPlayback() {
    if (state.timer) window.clearInterval(state.timer);
    state.timer = null; state.playing = false;
}
function toastMessage(message) {
    const toast = $("toast"); toast.textContent = message; toast.classList.add("show");
    window.clearTimeout(toastMessage.timer); toastMessage.timer = window.setTimeout(() => toast.classList.remove("show"), 2600);
}

$("runButton").addEventListener("click", () => runSimulation(true));
$("strategySelect").addEventListener("change", (event) => {
    state.chartMode = event.target.value;
    if (state.chartMode !== "all") state.selected = state.chartMode;
    state.queueChartMax = 10;
    state.directionChartMax = 10;
    syncChartLegends();
    if (state.results) renderDashboard();
});
$("timelineRange").addEventListener("input", (event) => {
    stopPlayback(); setCurrentStep(parseInt(event.target.value, 10));
});
$("playButton").addEventListener("click", () => state.playing ? (stopPlayback(), renderIntersection()) : startPlayback());
window.addEventListener("resize", () => { if (state.results) { drawQueueChart(); drawDirectionChart(); } });
document.querySelectorAll(".nav-link").forEach((link) => link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
}));

// Load a repeatable preview so the dashboard is useful before the first click.
runSimulation(false);





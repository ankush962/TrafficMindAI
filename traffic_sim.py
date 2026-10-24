"""A repeatable, synthetic traffic-signal control prototype.

The Random Forest imitates a heuristic green-time target generated below. It is
an ML-assisted controller for demonstration, not a model trained on measured
traffic or proven optimal signal plans.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


SEED = 42
DIRECTIONS = ["North", "East", "South", "West"]

# Traffic rates in the scenarios are vehicles per minute per approach.
# Convert to vehicles per 5-second simulation step when generating arrivals.
STEP_DURATION_SEC = 5
SATURATION_FLOW_RATE = 0.5  # vehicles per second per open lane
DEFAULT_LANES_PER_APPROACH = 2
YELLOW_CLEARANCE_STEPS = 1
MIN_GREEN_STEPS = 2
MAX_GREEN_STEPS = 9
FIXED_GREEN_SEC = 30

SCENARIOS = {
    "Balanced": {
        "description": "Moderate demand across all approaches.",
        "arrival_rates_veh_per_min": {
            "North": 8.0,
            "East": 6.0,
            "South": 6.0,
            "West": 8.0,
        },
        "lanes_open": {direction: DEFAULT_LANES_PER_APPROACH for direction in DIRECTIONS},
    },
    "Peak": {
        "description": "Higher demand, with the north and west approaches busiest.",
        "arrival_rates_veh_per_min": {
            "North": 12.0,
            "East": 8.0,
            "South": 7.0,
            "West": 12.0,
        },
        "lanes_open": {direction: DEFAULT_LANES_PER_APPROACH for direction in DIRECTIONS},
    },
    "East lane incident": {
        "description": "An incident leaves one eastbound lane open while demand rises.",
        "arrival_rates_veh_per_min": {
            "North": 8.0,
            "East": 14.0,
            "South": 6.0,
            "West": 8.0,
        },
        "lanes_open": {
            "North": DEFAULT_LANES_PER_APPROACH,
            "East": 1,
            "South": DEFAULT_LANES_PER_APPROACH,
            "West": DEFAULT_LANES_PER_APPROACH,
        },
    },
}

FEATURE_COLUMNS = [
    "queue_length",
    "avg_speed_kmh",
    "occupancy_pct",
    "arrival_rate_veh_per_step",
    "approach_wait_sec",
    "lanes_open",
]


def estimate_observation(queue_length, arrival_rate_veh_per_step, approach_wait_sec, lanes_open):
    """Build estimated detector features from the simulated state.

    Speed and occupancy are proxies derived from queue length; they are not
    independent sensor measurements in this prototype.
    """
    return {
        "queue_length": queue_length,
        "avg_speed_kmh": float(np.clip(50.0 - 0.8 * queue_length, 5.0, 60.0)),
        "occupancy_pct": float(np.clip(queue_length * 2.0, 0.0, 100.0)),
        "arrival_rate_veh_per_step": arrival_rate_veh_per_step,
        "approach_wait_sec": approach_wait_sec,
        "lanes_open": lanes_open,
    }


def generate_synthetic_traffic_dataset(samples=6000):
    """Generate training data for a heuristic green-time target."""
    rng = np.random.default_rng(SEED)
    queue_length = rng.integers(0, 50, size=samples)
    avg_speed_kmh = np.clip(
        50.0 - 0.8 * queue_length + rng.normal(0, 3, samples), 5.0, 60.0
    )
    occupancy_pct = np.clip(queue_length * 2.0 + rng.uniform(-5, 5, samples), 0, 100)
    arrival_rate = rng.uniform(0.25, 3.0, size=samples)  # vehicles per 5-second step
    approach_wait_sec = rng.uniform(0, 180, size=samples)
    lanes_open = rng.integers(1, DEFAULT_LANES_PER_APPROACH + 1, size=samples)

    # This is a hand-built teaching heuristic, not a label from real traffic.
    required_green_sec = (
        (queue_length / (SATURATION_FLOW_RATE * lanes_open)) * 0.6
        + (100.0 - avg_speed_kmh) * 0.15
        + approach_wait_sec * 0.1
        + arrival_rate * 2.0
    )
    heuristic_green_sec = np.clip(required_green_sec, 10.0, 45.0)

    return pd.DataFrame(
        {
            "queue_length": queue_length,
            "avg_speed_kmh": avg_speed_kmh,
            "occupancy_pct": occupancy_pct,
            "arrival_rate_veh_per_step": arrival_rate,
            "approach_wait_sec": approach_wait_sec,
            "lanes_open": lanes_open,
            "heuristic_green_sec": heuristic_green_sec,
        }
    )


def train_and_save_ml_model(model_filename="traffic_rf_model.pkl"):
    """Train a Random Forest to imitate the synthetic green-time heuristic."""
    print("Training the Random Forest on a synthetic heuristic dataset...")
    data = generate_synthetic_traffic_dataset()
    X = data[FEATURE_COLUMNS]
    y = data["heuristic_green_sec"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test.to_numpy() - predictions) ** 2)))
    r2 = r2_score(y_test, predictions)
    print("Held-out fit to the synthetic heuristic target (not traffic performance):")
    print(f" - RMSE: {rmse:.2f} seconds")
    print(f" - R²:   {r2:.3f}")

    joblib.dump(model, model_filename)
    return model


def generate_shared_arrivals(steps, arrival_rates_veh_per_min, seed=SEED):
    """Generate the same Poisson arrivals for each controller in a scenario."""
    rng = np.random.default_rng(seed)
    seconds_per_step = STEP_DURATION_SEC
    return {
        direction: rng.poisson(
            arrival_rates_veh_per_min[direction] * seconds_per_step / 60.0,
            size=steps,
        )
        for direction in DIRECTIONS
    }


def simulate_intersection(controller_type, arrivals, ml_model=None, scenario=None, total_steps=240):
    """Simulate one four-way intersection for 20 minutes by default."""
    if controller_type not in {"fixed", "queue_adaptive", "ml_adaptive"}:
        raise ValueError(f"Unknown controller type: {controller_type}")
    if controller_type == "ml_adaptive" and ml_model is None:
        raise ValueError("ml_model is required for the ml_adaptive controller")

    scenario = scenario or {}
    lanes_open = {
        direction: int(scenario.get("lanes_open", {}).get(direction, DEFAULT_LANES_PER_APPROACH))
        for direction in DIRECTIONS
    }
    if any(count < 1 for count in lanes_open.values()):
        raise ValueError("Each approach must have at least one open lane")

    queues = {direction: 0 for direction in DIRECTIONS}
    discharge_credit = {direction: 0.0 for direction in DIRECTIONS}
    # Approach wait is elapsed time since that approach last received green.
    approach_wait_sec = {direction: 0 for direction in DIRECTIONS}
    arrival_history = {direction: [0] * 5 for direction in DIRECTIONS}

    active_dir = "North"
    remaining_green_steps = FIXED_GREEN_SEC // STEP_DURATION_SEC
    clearance_steps_remaining = 0
    total_delay_vehicle_sec = 0.0
    records = []

    for step in range(total_steps):
        for direction in DIRECTIONS:
            new_vehicles = int(arrivals[direction][step])
            queues[direction] += new_vehicles
            arrival_history[direction].append(new_vehicles)
            arrival_history[direction].pop(0)

        served = 0
        clearance_this_step = False
        decision_made = step == 0
        green_allocation_sec = FIXED_GREEN_SEC if step == 0 else np.nan
        decision_reason = "Initial North phase: 30-second fixed green" if step == 0 else ""

        if clearance_steps_remaining > 0:
            clearance_this_step = True
            clearance_steps_remaining -= 1
        else:
            if remaining_green_steps <= 0:
                previous_dir = active_dir

                if controller_type == "fixed":
                    next_idx = (DIRECTIONS.index(active_dir) + 1) % len(DIRECTIONS)
                    active_dir = DIRECTIONS[next_idx]
                    allocated_green_sec = FIXED_GREEN_SEC
                    decision_reason = f"Fixed rotation: serve {active_dir} for {allocated_green_sec}s"

                elif controller_type == "queue_adaptive":
                    priority_scores = {
                        direction: queues[direction] + approach_wait_sec[direction] * 0.04
                        for direction in DIRECTIONS
                    }
                    active_dir = max(priority_scores, key=priority_scores.get)
                    allocated_green_sec = float(
                        np.clip(10 + queues[active_dir] * 1.2, 10, 45)
                    )
                    decision_reason = (
                        f"Largest queue/fairness score ({priority_scores[active_dir]:.1f}); "
                        f"queue {queues[active_dir]} vehicles, approach wait "
                        f"{approach_wait_sec[active_dir]}s"
                    )

                else:  # ML-assisted adaptive controller
                    predictions = {}
                    for direction in DIRECTIONS:
                        features = estimate_observation(
                            queues[direction],
                            float(np.mean(arrival_history[direction])),
                            approach_wait_sec[direction],
                            lanes_open[direction],
                        )
                        prediction = float(ml_model.predict(pd.DataFrame([features])[FEATURE_COLUMNS])[0])
                        # The small wait bonus helps prevent a low-demand approach from starving.
                        predictions[direction] = prediction + approach_wait_sec[direction] * 0.05

                    active_dir = max(predictions, key=predictions.get)
                    features = estimate_observation(
                        queues[active_dir],
                        float(np.mean(arrival_history[active_dir])),
                        approach_wait_sec[active_dir],
                        lanes_open[active_dir],
                    )
                    allocated_green_sec = float(
                        ml_model.predict(pd.DataFrame([features])[FEATURE_COLUMNS])[0]
                    )
                    decision_reason = (
                        f"RF green estimate {allocated_green_sec:.1f}s; queue "
                        f"{queues[active_dir]}, approach wait {approach_wait_sec[active_dir]}s, "
                        f"{lanes_open[active_dir]} lane(s) open"
                    )

                remaining_green_steps = max(
                    MIN_GREEN_STEPS,
                    min(
                        MAX_GREEN_STEPS,
                        int(round(allocated_green_sec / STEP_DURATION_SEC)),
                    ),
                )
                decision_made = True
                green_allocation_sec = allocated_green_sec

                if active_dir != previous_dir:
                    # The transition step itself is the first clearance step.
                    clearance_this_step = YELLOW_CLEARANCE_STEPS > 0
                    clearance_steps_remaining = max(0, YELLOW_CLEARANCE_STEPS - 1)

            if not clearance_this_step:
                # Carry fractional capacity forward so a one-lane approach can serve
                # 2 then 3 vehicles over successive 5-second steps (2.5 on average).
                step_capacity = SATURATION_FLOW_RATE * STEP_DURATION_SEC * lanes_open[active_dir]
                discharge_credit[active_dir] += step_capacity
                available_capacity = int(np.floor(discharge_credit[active_dir] + 1e-9))
                served = min(queues[active_dir], available_capacity)
                queues[active_dir] -= served
                discharge_credit[active_dir] -= served
                if queues[active_dir] == 0:
                    # Do not bank unused green capacity while the approach is empty.
                    discharge_credit[active_dir] = 0.0
                remaining_green_steps -= 1

        # Count every vehicle left in every queue, including the green approach.
        total_delay_vehicle_sec += sum(queues.values()) * STEP_DURATION_SEC

        for direction in DIRECTIONS:
            if direction == active_dir and not clearance_this_step:
                approach_wait_sec[direction] = 0
            else:
                approach_wait_sec[direction] += STEP_DURATION_SEC

        records.append(
            {
                "step": step,
                "time_min": step * STEP_DURATION_SEC / 60.0,
                "controller": controller_type,
                "active_direction": active_dir,
                "signal_state": "CLEARANCE" if clearance_this_step else "GREEN",
                "decision_made": decision_made,
                "decision_reason": decision_reason,
                "green_allocation_sec": green_allocation_sec,
                "vehicles_served": served,
                "total_queue": sum(queues.values()),
                "cum_delay_vehicle_sec": total_delay_vehicle_sec,
            }
        )

    return pd.DataFrame(records)


def summarize_scenario(scenario_name, controller_name, frame):
    return {
        "Scenario": scenario_name,
        "Controller": controller_name,
        "Average queue (vehicles)": round(frame["total_queue"].mean(), 2),
        "Maximum queue (vehicles)": int(frame["total_queue"].max()),
        "Vehicles served": int(frame["vehicles_served"].sum()),
        "Total delay (vehicle-hours)": round(
            frame["cum_delay_vehicle_sec"].iloc[-1] / 3600.0, 3
        ),
    }


def main():
    model_file = "traffic_rf_model.pkl"
    rf_model = train_and_save_ml_model(model_file)
    controller_names = {
        "fixed": "Fixed-Time",
        "queue_adaptive": "Queue-Adaptive",
        "ml_adaptive": "ML-Assisted Random Forest",
    }
    controller_colors = {
        "fixed": "#e74c3c",
        "queue_adaptive": "#f39c12",
        "ml_adaptive": "#2ecc71",
    }

    results = []
    scenario_runs = {}
    for scenario_index, (scenario_name, scenario) in enumerate(SCENARIOS.items()):
        arrivals = generate_shared_arrivals(
            steps=240,
            arrival_rates_veh_per_min=scenario["arrival_rates_veh_per_min"],
            seed=SEED + scenario_index,
        )
        run = {
            "fixed": simulate_intersection("fixed", arrivals, scenario=scenario),
            "queue_adaptive": simulate_intersection(
                "queue_adaptive", arrivals, scenario=scenario
            ),
            "ml_adaptive": simulate_intersection(
                "ml_adaptive", arrivals, ml_model=rf_model, scenario=scenario
            ),
        }
        scenario_runs[scenario_name] = run
        for controller_type, frame in run.items():
            results.append(
                summarize_scenario(
                    scenario_name, controller_names[controller_type], frame
                )
            )

        print(f"\n{scenario_name}: {scenario['description']}")
        ml_decisions = run["ml_adaptive"].loc[
            run["ml_adaptive"]["decision_made"],
            ["time_min", "active_direction", "signal_state", "green_allocation_sec", "decision_reason"],
        ].head(6)
        print("Sample ML controller recommendations:")
        print(ml_decisions.to_string(index=False))

    print("\nSIMULATION BENCHMARK SUMMARY")
    print(pd.DataFrame(results).to_string(index=False))

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, axes = plt.subplots(
        len(scenario_runs), 2, figsize=(13, 4.2 * len(scenario_runs)), sharex="col"
    )
    if len(scenario_runs) == 1:
        axes = np.array([axes])

    for row, (scenario_name, run) in enumerate(scenario_runs.items()):
        queue_ax, delay_ax = axes[row]
        for controller_type, frame in run.items():
            label = controller_names[controller_type]
            color = controller_colors[controller_type]
            queue_ax.plot(frame["time_min"], frame["total_queue"], label=label, color=color)
            delay_ax.plot(
                frame["time_min"],
                frame["cum_delay_vehicle_sec"] / 60.0,
                label=label,
                color=color,
            )
        queue_ax.set_title(f"{scenario_name}: {SCENARIOS[scenario_name]['description']}")
        queue_ax.set_ylabel("Queue (vehicles)")
        delay_ax.set_ylabel("Cumulative delay (vehicle-minutes)")
        delay_ax.set_xlabel("Simulation time (minutes)")
        queue_ax.legend(loc="upper left", fontsize=8)
        delay_ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Traffic Controller Comparison Across Repeatable Scenarios", fontsize=14)
    fig.tight_layout()
    output_plot = Path("simulation_benchmark.png")
    fig.savefig(output_plot, dpi=250, bbox_inches="tight")
    print(f"\nComparison chart saved to: {output_plot.resolve()}")


if __name__ == "__main__":
    main()

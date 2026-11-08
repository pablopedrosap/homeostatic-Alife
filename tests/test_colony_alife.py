import numpy as np

from homeostatic_colony.envs.colony_env import ColonyConfig, ColonySimulation


def _base_cfg(enable_evolution: bool) -> ColonyConfig:
    return ColonyConfig(
        width=12.0,
        height=12.0,
        n_initial_cells=4,
        max_cells=40,
        n_nutrients=6,
        n_toxins=0,
        n_cool=2,
        nutrient_radius=3.0,
        toxin_radius=2.0,
        cool_radius=2.0,
        basal_cost=0.0005,
        move_cost=0.0005,
        nutrient_gain=0.50,
        heat_move=0.0015,
        heat_hazard=0.0,
        cooling_rate=0.02,
        dmg_toxin=0.0,
        dmg_overheat=0.01,
        repair_rate=0.02,
        t_crit=1.0,
        t_fatal=1.3,
        division_energy=0.58,
        division_cost=0.10,
        min_division_age=1,
        max_steps=80,
        seed=123,
        use_signaling=True,
        enable_resource_dynamics=False,
        predictive_steering=True,
        enable_evolution=enable_evolution,
        initial_trait_std=0.0,
        mutation_std=0.2,
    )


def test_colony_evolution_generates_new_traits_and_generations():
    sim = ColonySimulation(_base_cfg(enable_evolution=True))
    history = sim.run(max_steps=80)
    assert history, "simulation history should not be empty"

    final = history[-1]
    assert final["births_total"] > 0, "expected reproduction events"
    assert final["max_generation"] >= 1, "expected lineage depth growth"
    assert final["lineages_alive"] >= 1
    assert final["diversity_entropy"] > 0.0, "mutation should create trait diversity"


def test_colony_clonal_mode_keeps_traits_identical():
    sim = ColonySimulation(_base_cfg(enable_evolution=False))
    history = sim.run(max_steps=80)
    assert history, "simulation history should not be empty"
    assert history[-1]["births_total"] > 0, "expected reproduction events in clonal mode too"

    alive_traits = np.array([c.traits for c in sim.cells if c.alive], dtype=float)
    assert alive_traits.shape[0] >= 2, "need at least two survivors to compare clonal traits"
    # In clonal mode with zero initial variance, all descendants stay identical.
    assert float(np.max(np.std(alive_traits, axis=0))) < 1e-6

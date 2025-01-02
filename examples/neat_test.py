import dataclasses
import jax
import numpy as np
import pytest
from evojax.task.slimevolley import SlimeVolley
from examples.neat import (
    NEAT,
    Config,
    Genome,
    NEATConfig,
    NEATPolicy,
    calculate_depth,
    calculate_distance,
    calculate_species,
    create_empty_genome_fn,
    crossover,
    generate_new_population_fn,
    mutate_add_connection,
    mutate_add_node,
    mutate_attributes,
)
import jax.numpy as jnp


# def test_generate_new_population_fn():

#     # SlimeVolley
#     max_steps = 3000
#     train_task = SlimeVolley(test=False, max_steps=max_steps)
#     test_task = SlimeVolley(test=True, max_steps=max_steps)

#     assert len(train_task.obs_shape) == 1
#     assert len(train_task.act_shape) == 1
#     config = Config(
#         input_dim=train_task.obs_shape[0],
#         output_dim=train_task.act_shape[0],
#         max_nodes=100,
#         max_edges=1000,
#         max_depth=10,
#     )
#     neat_config = NEATConfig(config, 100)

#     policy = NEATPolicy(config)
#     solver = NEAT(neat_config)

#     generate_new_population_fn(neat_config, policy, solver)

max_steps = 3000
train_task = SlimeVolley(test=False, max_steps=max_steps)
config = Config(
    input_dim=train_task.obs_shape[0],
    output_dim=train_task.act_shape[0],
    max_nodes=20,
    max_edges=50,
    max_depth=10,
)
neat_config = NEATConfig(config, 100)


def test_mutate_add_node():
    genome = create_empty_genome_fn(config)
    new_genome = mutate_add_node(config, genome, jax.random.PRNGKey(0), 100)
    print(new_genome)
    # TODO: Add assertions.


@pytest.mark.parametrize(
    ["src", "dst", "max_depth", "expected"],
    [
        pytest.param(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            10,
            10,
            id="max_depth_equal_to_depth",
        ),
        pytest.param(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            9,
            -1,
            id="max_depth_less_than_depth",
        ),
        # 0 1 2
        # |\| |
        # 5 | |
        # |/  |
        # 3   4
        pytest.param(
            [0, 0, 1, 2, 5],
            [5, 3, 3, 4, 3],
            11,
            3,
            id="two_way",
        ),
        pytest.param(
            [5, 6],
            [6, 5],
            11,
            -1,
            id="cycle",
        ),
    ],
)
def test_calculate_depth(src, dst, max_depth, expected):
    config = Config(
        input_dim=3,
        output_dim=2,
        max_nodes=10,
        max_edges=20,
        max_depth=max_depth,
    )

    depth = calculate_depth(
        config,
        jnp.array(src),
        jnp.array(dst),
    )

    assert depth == expected


def test_mutate_add_connection():
    config = Config(
        input_dim=3,
        output_dim=2,
        max_nodes=10,
        max_edges=20,
        max_depth=10,
    )

    genome = mutate_add_connection(
        config,
        create_empty_genome_fn(config),
        jax.random.PRNGKey(0),
        jnp.array(100),
    )


def test_calculate_distance():
    # ONly conn_ids and weights are used
    genome1 = Genome(
        node_ids=jnp.array([]),
        node_activation=jnp.array([]),
        conn_ids=jnp.array([0, 1, 2, 5, -1, -1]),
        conn_in=jnp.array([]),
        conn_out=jnp.array([]),
        conn_weights=jnp.array([0.1, 0.2, 0.3, -0.4, -0.5, -0.6]),
        conn_enabled=jnp.array([]),
    )

    genome2 = dataclasses.replace(
        genome1,
        conn_ids=jnp.array([0, 2, 3, 4, 5, -1]),
        conn_weights=jnp.array([-0.1, -0.2, -0.3, 0.4, 0.5, 0.6]),
    )

    assert calculate_distance(neat_config, genome1, genome1) == 0.0
    assert calculate_distance(neat_config, genome2, genome2) == 0.0
    # - common conn:   0.5 * AVG(|0.1 - (-0.1)| + |0.3  - (-0.2)| + |-0.4 - 0.5|)
    # - disjoint conn: 1.0 * (3 / 9)
    expected = 0.5 * (0.2 + 0.5 + 0.9) / 3 + 1.0 * (3 / 9)
    assert calculate_distance(neat_config, genome1, genome2) == expected


def test_calculate_species():
    config = Config(
        input_dim=3,
        output_dim=2,
        max_nodes=10,
        max_edges=20,
        max_depth=10,
    )
    neat_config = NEATConfig(config, 100, num_species=3)
    genome1, edge = mutate_add_node(
        config, create_empty_genome_fn(config), jax.random.key(0), 100
    )
    genome2, edge = mutate_add_node(config, genome1, jax.random.key(1), edge)
    genome3, edge = mutate_add_connection(config, genome2, jax.random.key(2), edge)
    genome4 = mutate_attributes(neat_config, genome2, jax.random.key(3))
    genome5, edge = mutate_add_node(config, genome2, jax.random.key(4), edge)
    genome6 = mutate_attributes(neat_config, genome5, jax.random.key(5))

    genomes = [genome1, genome2, genome3, genome4, genome5, genome6]

    # Stack genomes into batched structure
    stacked_genomes = jax.tree.map(lambda *arrays: jnp.stack(arrays), *genomes)

    # Create double-vmapped distance function
    pairwise_distance = jax.vmap(
        lambda g1: jax.vmap(lambda g2: calculate_distance(neat_config, g1, g2))(
            stacked_genomes
        )
    )(stacked_genomes)

    # Result shape: (n_genomes, n_genomes) distance matrix
    assert pairwise_distance.shape == (len(genomes), len(genomes))
    print(pairwise_distance)
    assert jnp.all(pairwise_distance.diagonal() == 0)

    species = calculate_species(neat_config, stacked_genomes, jax.random.key(0))

    np.testing.assert_array_equal(species, jnp.array([2, 0, 0, 0, 1, 1]))


def test_crossover():
    config = Config(
        input_dim=3,
        output_dim=2,
        max_nodes=10,
        max_edges=20,
        max_depth=10,
    )
    neat_config = NEATConfig(config, 6, num_species=3, prob_update_conn_weights=1)
    genome1, edge = mutate_add_node(
        config, create_empty_genome_fn(config), jax.random.key(0), 100
    )
    genome2, edge = mutate_add_node(config, genome1, jax.random.key(1), edge)
    genome3, edge = mutate_add_node(config, genome2, jax.random.key(2), edge)
    genome4 = mutate_attributes(neat_config, genome2, jax.random.key(3))
    genome5, edge = mutate_add_node(config, genome2, jax.random.key(4), edge)
    genome6 = mutate_attributes(neat_config, genome5, jax.random.key(5))

    genomes = [genome1, genome2, genome3, genome4, genome5, genome6]

    # Stack genomes into batched structure
    stacked_genomes = jax.tree.map(lambda *arrays: jnp.stack(arrays), *genomes)

    crossovered = crossover(
        neat_config=neat_config,
        key=jax.random.key(2),
        fitness=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        pop=stacked_genomes,
        idx=jnp.array([2, 2, 5, 2, 2, 5]),
    )

    np.testing.assert_array_equal(crossovered.node_ids, genome5.node_ids)
    np.testing.assert_array_equal(crossovered.node_activation, genome5.node_activation)
    np.testing.assert_array_equal(crossovered.conn_ids, genome5.conn_ids)
    np.testing.assert_array_equal(crossovered.conn_in, genome5.conn_in)
    np.testing.assert_array_equal(crossovered.conn_out, genome5.conn_out)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        crossovered.conn_weights,
        genome5.conn_weights,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        crossovered.conn_weights,
        genome3.conn_weights,
    )

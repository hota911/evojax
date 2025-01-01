from dataclasses import dataclass
import dataclasses
from datetime import datetime
from functools import partial
import math
import os
import jax
import numpy as np
from evojax import util
from evojax.algo.base import NEAlgorithm
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.task.base import TaskState

import jax.numpy as jnp

from evojax.task.slimevolley import Game, SlimeVolley, State
from evojax.trainer import Trainer

from PIL import Image


##############################################################
# Patch the SlimeVolley class to add the render method.
##############################################################


def render(state: State, task_id: int = 0) -> Image:
    """Render a specified task."""
    task_game_state = jax.tree.map(lambda x: x[task_id], state.game_state)
    game = Game(task_game_state)
    canvas = game.display()
    img = Image.fromarray(canvas)
    return img


SlimeVolley.render = render  # type: ignore

##############################################################
# Data classes for the NEAT algorithm.
##############################################################

# Data classes for the NEAT algorithm.
@partial(
    jax.tree_util.register_dataclass,
    # This is required to run on colab
    data_fields=[
        "node_ids",
        "node_activation",
        "conn_ids",
        "conn_in",
        "conn_out",
        "conn_weights",
        "conn_enabled",
    ],
    meta_fields=[],
)
@dataclass
class Genome:
    """Stores node and connection info in JAX arrays."""

    node_ids: jax.Array  # shape [num_nodes]
    # 0: relu
    node_activation: jax.Array  # shape [num_nodes]

    # Connections.
    conn_ids: jax.Array  # shape [num_connections]
    conn_in: jax.Array  # shape [num_connections]
    conn_out: jax.Array  # shape [num_connections]
    conn_weights: jax.Array  # shape [num_connections]
    conn_enabled: jax.Array  # shape [num_connections]

# Type alias for the genome population.
# To make it easier to understand the genome is only a single genome or a population of genomes.
GenomePopulation = Genome

@dataclass(frozen=True)
class Config:
    """Stores static info for the NEAT algorithm."""

    input_dim: int
    output_dim: int
    max_nodes: int
    max_edges: int
    max_depth: int


##############################################################
# Helper functions for data classes.
##############################################################


def create_genome(
    config: Config,
    node_ids,
    node_activation,
    conn_ids,
    conn_in,
    conn_out,
    conn_weights,
    conn_enabled,
) -> Genome:
    node_pad_config = [(0, config.max_nodes - node_ids.shape[0], 0)]
    node_ids = jax.lax.pad(node_ids, -1, node_pad_config)
    node_activation = jax.lax.pad(node_activation, -1, node_pad_config)

    conn_pad_config = [(0, config.max_edges - conn_ids.shape[0], 0)]
    conn_ids = jax.lax.pad(conn_ids, -1, conn_pad_config)
    conn_in = jax.lax.pad(conn_in, -1, conn_pad_config)
    conn_out = jax.lax.pad(conn_out, -1, conn_pad_config)
    conn_weights = jax.lax.pad(conn_weights, jnp.nan, conn_pad_config)
    conn_enabled = jax.lax.pad(conn_enabled, False, conn_pad_config)
    return Genome(
        node_ids,
        node_activation,
        conn_ids,
        conn_in,
        conn_out,
        conn_weights,
        conn_enabled,
    )


def create_empty_genome_fn(config: Config) -> Genome:
    """Create an empty genome.

    Args:
        neat_config (NEATConfig): The NEAT configuration.

    Returns:
        jax.Array: The empty genome.
    """
    nodes = config.input_dim + config.output_dim
    conns = config.input_dim * config.output_dim
    node_ids = jnp.arange(0, nodes)
    node_activation = jnp.array([0] * (nodes))
    # input -> output
    # E.g. input: 2, output: 2
    # conn_in:  [0, 1, 0, 1]
    # conn_out: [2, 2, 3, 3]
    conn_ids = jnp.arange(0, conns)
    conn_in = jnp.tile(jnp.arange(0, config.input_dim), config.output_dim)
    conn_out = jnp.repeat(
        jnp.arange(config.input_dim, config.input_dim + config.output_dim),
        config.input_dim,
    )
    conn_weights = jnp.zeros(conns)
    conn_enabled = jnp.ones(conns, dtype=bool)

    return create_genome(
        config,
        node_ids,
        node_activation,
        conn_ids,
        conn_in,
        conn_out,
        conn_weights,
        conn_enabled,
    )


def genome_to_mermaid(genome: Genome, config: Config, show_disabled: bool) -> str:
    """Convert genome to mermaid graph definition."""
    mermaid = [
        "graph TD;",
    ]

    # Add nodes with styling
    def node_type(node_id):
        if node_id < config.input_dim:
            return "i"
        elif node_id < config.input_dim + config.output_dim:
            return "o"
        else:
            return "h"

    for node_id in genome.node_ids:
        if node_id >= 0:
            type = node_type(node_id)
            if node_id == 0:
                mermaid.append("    subgraph Input")
            elif node_id == config.input_dim:
                mermaid.append("    end")
                mermaid.append("    subgraph Output")
            elif node_id == config.input_dim + config.output_dim:
                mermaid.append("    end")
                mermaid.append("    subgraph Hidden")
            if type == "i":
                mermaid.append(f'    {type}{node_id}(["Input {node_id}"]):::input')
            elif type == "o":
                mermaid.append(f'    {type}{node_id}(["Output {node_id}"]):::output')
            else:
                mermaid.append(
                    f'    {type}{node_id}(["Hidden {node_id}"]):::hidden_node'
                )
    mermaid.append("    end")

    # Add connections
    for id, src, dst, w, enabled in zip(
        genome.conn_ids,
        genome.conn_in,
        genome.conn_out,
        genome.conn_weights,
        genome.conn_enabled,
    ):
        if id >= 0 and (show_disabled or enabled):
            src_prefix = node_type(src)
            dst_prefix = node_type(dst)
            weight = f"{w:.2f}"
            arrow = "-->" if enabled else "-.->"
            mermaid.append(f"    {src_prefix}{src} {arrow}|{weight}| {dst_prefix}{dst}")

    # Add styles
    mermaid.extend(
        [
            "    classDef input fill:#61DAFB,stroke:#333,stroke-width:2px;",
            "    classDef output fill:#4EC9B0,stroke:#333,stroke-width:2px;",
            "    classDef hidden_node fill:#9CA3AF,stroke:#333,stroke-width:2px;",
            "    linkStyle default stroke:#E5E7EB,stroke-width:2px;",
        ]
    )

    return "\n".join(mermaid)


def get_params_format_fn(config: Config):
    """Generate the number of parameters and format function for the genome.

    Parameters of NetworkPolicy must be Array and cannot be Pytree.
    Hence, we need to flatten the Pytree to Array.
    """
    init_params = create_empty_genome_fn(config)
    flat, tree = jax.tree.flatten(init_params)

    # calculate the indexes to split the parameters.
    params_sizes = np.cumsum([np.prod(p.shape) for p in flat])[0:-1]
    info = [p.dtype for p in flat]
    print(info)
    print(params_sizes)

    def params_format_fn(params: jnp.ndarray) -> Genome:
        parts = jnp.split(params, params_sizes)
        leaves = [jnp.asarray(p, dtype=d) for p, d in zip(parts, info)]

        return jax.tree.unflatten(tree, leaves)

    return params_sizes[-1], params_format_fn


def convert_to_array_fn(genome: GenomePopulation | Genome) -> jax.Array:
    """Convert the genome as a flat array."""
    return jnp.concatenate(jax.tree.leaves(genome), dtype=jnp.float32, axis=-1)


##############################################################
# Feed-forward function.
##############################################################


def get_adjacency_matrix_fn(config: Config, genome: Genome):
    matrix = jnp.zeros((config.max_nodes, config.max_nodes))
    matrix = matrix + jnp.pad(
        jnp.identity(config.input_dim),
        (
            (0, config.max_nodes - config.input_dim),
            (0, config.max_nodes - config.input_dim),
        ),
    )
    matrix = matrix.at[genome.conn_out, genome.conn_in].set(
        jnp.where(genome.conn_enabled, genome.conn_weights, 0.0)
    )
    return matrix


ACTIVATION_FUNCTIONS = [lambda x: x, jax.nn.relu, jax.nn.sigmoid, jax.nn.tanh]


def matmul_fn(adj, x):
    return jnp.matmul(adj, x)


def forward_fn(config: Config, genome: Genome, x: jax.Array):
    """Calculate the forward network.

    Args:
        config (Config): The configuration. Used to pad the input.
        genome (Genome): _description_
        x (jax.Array): _description_

    Returns:
        _type_: _description_
    """
    x = jnp.pad(x, (0, config.max_nodes - x.shape[0]))
    adj = get_adjacency_matrix_fn(config, genome)
    for _ in range(config.max_depth):
        x = matmul_fn(adj, x)

        # Apply the activation function.
        def activate_fn(activation, x):
            return jax.lax.switch(activation, ACTIVATION_FUNCTIONS, x)

        activate = jax.vmap(activate_fn)
        x = activate(genome.node_activation, x)
    return x


# For testing
forward = jax.jit(jax.vmap(forward_fn, in_axes=(None, 0, 0)), static_argnums=(0,))


@dataclass
class NEATPolicy(PolicyNetwork):
    def __init__(self, config: Config):
        self._config = config

        def forward_with_config_fn(obs: jax.Array, params: Genome):
            result = forward_fn(config, params, obs)
            return result[config.input_dim : config.input_dim + config.output_dim]

        self._forward_with_config_fn = jax.jit(jax.vmap(forward_with_config_fn))

        # Get the number of parameters
        self.num_params, format_params_fn = get_params_format_fn(config)
        self._format_params = jax.jit(jax.vmap(format_params_fn))

    def get_actions(
        self, t_states: TaskState, params: jax.Array, p_states: PolicyState
    ) -> tuple[jax.Array, PolicyState]:
        params = self._format_params(params)
        return self._forward_with_config_fn(t_states.obs, params), p_states


##############################################################
# NEAT algorithm.
##############################################################
@dataclass(frozen=True)
class NEATConfig:
    """Stores static info for the NEAT algorithm."""

    # Internal config used by the NEAT algorithm and NEATPolicy.
    config: Config

    # Population size.
    pop_size: int

    # Rate of the population to keep.
    rate_keep_best: float = 0.2

    # ------------------------------
    # Structure mutation parameters.
    # ------------------------------
    # Rate of adding a node.
    prob_add_node: float = 0.1

    # Rate of adding a connection.
    prob_add_connection: float = 0.2

    # ------------------------------
    # Attribute mutation parameters.
    # ------------------------------
    prob_update_activation: float = 0.1
    prob_update_conn_weights: float = 0.1
    prob_update_conn_enabled: float = 0.1

    # ------------------------------
    # Crossover parameters.
    # ------------------------------
    coefficient_common_conn_weight: float = 0.5
    coefficient_disjoint_conn: float = 1.0
    num_species: int = 5

@jax.jit
def pick_one(key: jax.Array, mask: jax.Array) -> int:
    """Pick up one index where the mask is true.

    Args:
        mask (jax.Array): The mask. 1-d array of bool.
    Returns:
        jax.Array: The selected index.
    """
    idx = jnp.arange(mask.shape[0])
    # Filter the indices where mask is True.
    filtered = jnp.sort(jnp.where(mask, idx, -1), descending=True)
    # Pick up one index from the filtered indices.
    chosen = jax.random.randint(key, (1,), 0, mask.sum())
    return filtered[chosen][0]  # type: ignore


def select(cond: jax.Array, genome_true: Genome, genome_false: Genome) -> Genome:
    """Select the genome based on the condition."""
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), genome_true, genome_false)


# @partial(jax.jit, static_argnums=(0,))
def calculate_depth(
    config: Config,
    conn_in: jax.Array,
    conn_out: jax.Array,
) -> jax.Array:
    """Calculate the depth of the nodes.

    Returns the maximum depth of the nodes or -1 if
    - the depth is greater than the max_depth
    - or there is a cycle in the network.
    """
    depth = jax.lax.pad(
        jnp.ones(config.input_dim, dtype=jnp.int32),
        0,
        [(0, config.max_nodes - config.input_dim, 0)],
    )
    for _ in range(config.max_depth + 1):
        # Update the depth of the nodes.
        depth = depth.at[conn_out].set(
            jnp.where(
                conn_out >= 0,
                jnp.maximum(depth[conn_out], depth[conn_in] + 1),
                depth[conn_out],
            )
        )

    max_depth = jnp.max(depth)
    return jnp.where(max_depth > config.max_depth, -1, max_depth)


def mutate_add_node(
    config: Config, genome: Genome, key: jax.Array, next_edge_id: jax.Array
) -> tuple[Genome, jax.Array]:
    """Add nodes to the genome. This function is not jit-compiled."""
    valid = jnp.array(True)

    conn_idx = pick_one(key, genome.conn_enabled)
    valid = jnp.logical_and(valid, conn_idx >= 0)

    # Get the selected connection values
    in_node = genome.conn_in[conn_idx]
    out_node = genome.conn_out[conn_idx]
    weight = genome.conn_weights[conn_idx]

    # Add a new node
    new_node_id = genome.node_ids.max() + 1
    valid = jnp.logical_and(valid, new_node_id < config.max_nodes)

    node_ids = genome.node_ids.at[new_node_id].set(new_node_id)
    node_activation = genome.node_activation.at[new_node_id].set(0)

    # Disable the connection
    conn_enabled = genome.conn_enabled.at[conn_idx].set(False)

    # Get the new connection indexes
    new_edge1_idx = jnp.count_nonzero(genome.conn_ids >= 0)
    new_edge_idx = jnp.array([new_edge1_idx, new_edge1_idx + 1])

    conn_ids = genome.conn_ids.at[new_edge_idx].set(
        jnp.array([next_edge_id, next_edge_id + 1])
    )
    conn_in = genome.conn_in.at[new_edge_idx].set(jnp.array([in_node, new_node_id]))
    conn_out = genome.conn_out.at[new_edge_idx].set(jnp.array([new_node_id, out_node]))
    conn_weights = genome.conn_weights.at[new_edge_idx].set(jnp.array([1.0, weight]))
    conn_enabled = conn_enabled.at[new_edge_idx].set(jnp.array([True, True]))

    return (
        select(
            valid,
            Genome(
                node_ids=node_ids,
                node_activation=node_activation,
                conn_ids=conn_ids,
                conn_in=conn_in,
                conn_out=conn_out,
                conn_weights=conn_weights,
                conn_enabled=conn_enabled,
            ),
            genome,
        ),
        jnp.where(valid, next_edge_id + 2, next_edge_id),
    )


@partial(jax.jit, static_argnums=(0,))
def mutate_add_connection(
    config: Config, genome: Genome, key: jax.Array, next_edge_id: jax.Array
) -> tuple[Genome, jax.Array]:
    """Add connection"""
    key, key_src, key_dst, key_weight = jax.random.split(key, 4)

    src = pick_one(
        key_src,
        jnp.logical_or(
            # input
            jnp.logical_and(0 <= genome.node_ids, genome.node_ids < config.input_dim),
            # or hidden
            genome.node_ids >= config.input_dim + config.output_dim,
        ),
    )
    # output or hidden
    dst = pick_one(key_dst, genome.node_ids >= config.input_dim)

    # New graph
    new_edge_idx = jnp.count_nonzero(genome.conn_ids >= 0)
    conn_ids = genome.conn_ids.at[new_edge_idx].set(next_edge_id)
    conn_in = genome.conn_in.at[new_edge_idx].set(src)
    conn_out = genome.conn_out.at[new_edge_idx].set(dst)
    conn_weights = genome.conn_weights.at[new_edge_idx].set(
        jax.random.uniform(key_weight, minval=-1.0, maxval=1.0)
    )
    conn_enabled = genome.conn_enabled.at[new_edge_idx].set(True)

    # Check
    # - src and dest are not the same
    valid = src != dst
    valid = jnp.logical_and(
        valid,
        jnp.logical_not(
            jnp.any(jnp.logical_and(genome.conn_in == src, genome.conn_out == dst))
        ),
    )
    # - there is no cycle
    valid = jnp.logical_and(valid, calculate_depth(config, conn_in, conn_out) != -1)

    return (
        Genome(
            node_ids=genome.node_ids,
            node_activation=genome.node_activation,
            conn_ids=jnp.where(valid, conn_ids, genome.conn_ids),
            conn_in=jnp.where(valid, conn_in, genome.conn_in),
            conn_out=jnp.where(valid, conn_out, genome.conn_out),
            conn_weights=jnp.where(valid, conn_weights, genome.conn_weights),
            conn_enabled=jnp.where(valid, conn_enabled, genome.conn_enabled),
        ),
        jnp.where(valid, next_edge_id + 2, next_edge_id),
    )


def mutate_none(
    genome: Genome, key: jax.Array, next_edge_id: jax.Array
) -> tuple[Genome, jax.Array]:
    return (
        Genome(
            node_ids=genome.node_ids,
            node_activation=genome.node_activation,
            conn_ids=genome.conn_ids,
            conn_in=genome.conn_in,
            conn_out=genome.conn_out,
            conn_weights=genome.conn_weights,
            conn_enabled=genome.conn_enabled,
        ),
        next_edge_id,
    )


def validate_genome(config: Config, genome: Genome) -> jax.Array:
    """Validate the genome.

    Returns: Array(shape=(), dtype=bool)"""
    valid = calculate_depth(config, genome.conn_in, genome.conn_out) != -1
    # TODO: Add more validation if needed.
    return valid


def mutate_attributes(
    neat_config: NEATConfig, genome: Genome, key: jax.Array
) -> Genome:
    """Mutate the attributes of the genome.

    - Change the activation function of the nodes.
    - Change the weights of the connections.
    - Change the enabled status of the connections.
    """
    # Update attributes
    (
        key,
        key_update_activation_choice,
        key_update_activation,
        key_update_conn_weights_choice,
        key_update_conn_weights,
        key_update_conn_enabled_choice,
    ) = jax.random.split(key, 6)
    node_activations = jnp.where(
        jnp.logical_and(
            genome.node_ids >= 0,
            jax.random.uniform(
                key_update_activation_choice, genome.node_activation.shape
            )
            < neat_config.prob_update_activation,
        ),
        jax.random.randint(
            key_update_activation,
            genome.node_activation.shape,
            minval=0,
            maxval=len(ACTIVATION_FUNCTIONS),
        ),
        genome.node_activation,
    )
    conn_weights = jnp.where(
        jnp.logical_and(
            genome.conn_ids >= 0,
            jax.random.uniform(
                key_update_conn_weights_choice, genome.conn_weights.shape
            )
            < neat_config.prob_update_conn_weights,
        ),
        jax.random.uniform(
            key_update_conn_weights, genome.conn_weights.shape, minval=-1.0, maxval=1.0
        ),
        genome.conn_weights,
    )
    conn_enabled = jnp.where(
        jnp.logical_and(
            genome.conn_ids >= 0,
            jax.random.uniform(
                key_update_conn_enabled_choice, genome.conn_enabled.shape
            )
            < neat_config.prob_update_conn_enabled,
        ),
        # Flip the connection.
        jnp.logical_not(genome.conn_enabled),
        genome.conn_enabled,
    )
    return dataclasses.replace(
        genome,
        node_activation=node_activations,
        conn_weights=conn_weights,
        conn_enabled=conn_enabled,
    )


def mutate_genome(
    neat_config: NEATConfig, genome: Genome, key: jax.Array, next_edge_id: jax.Array
) -> tuple[Genome, jax.Array]:
    """Mutate the genome."""

    key, key_choice, key_mutate = jax.random.split(key, 3)

    p = jax.random.uniform(key_choice)

    # TODO: Use for loop.
    index = jnp.where(
        p < neat_config.prob_add_node,
        0,
        jnp.where(
            p < neat_config.prob_add_node + neat_config.prob_add_connection,
            1,
            2,
        ),
    )

    branches = [
        lambda: mutate_add_node(neat_config.config, genome, key_mutate, next_edge_id),
        lambda: mutate_add_connection(
            neat_config.config, genome, key_mutate, next_edge_id
        ),
        lambda: mutate_none(genome, key_mutate, next_edge_id),
    ]
    # return (a, b, c)
    (genome, new_edge_id) = jax.lax.switch(index, branches)

    genome = mutate_attributes(neat_config, genome, key_mutate)

    return (genome, new_edge_id)

# Species

int32_max = np.iinfo(np.int32).max


def get_common_idx(array1: jax.Array, array2: jax.Array) -> jax.Array:
    """Get the index of the common elements between two arrays.

    Returns: for each i, the index of array1 where array2[i] is found, or -1 if not found.

    ```python
    ret = []
    for v1 in array2:
        ret.append(array1.find(v1))
    return ret
    """
    array1_sorted = jnp.where(array1 >= 0, array1, int32_max)
    possible_idx = jnp.searchsorted(array1_sorted, array2)
    return jnp.where(array1[possible_idx] == array2, possible_idx, -1)


def calculate_distance(
    neat_config: NEATConfig, genome1: Genome, genome2: Genome
) -> jax.Array:
    """Calculate the distance between two genomes."""
    common_idx = get_common_idx(genome1.conn_ids, genome2.conn_ids)

    diff_weight = jnp.mean(
        jnp.abs(genome1.conn_weights[common_idx] - genome2.conn_weights),
        where=common_idx >= 0,
    )

    num_common = jnp.count_nonzero(common_idx >= 0)
    num_conn = jnp.count_nonzero(genome1.conn_ids >= 0) + jnp.count_nonzero(
        genome2.conn_ids >= 0
    )

    disjoint_rate = (num_conn - 2 * num_common) / num_conn

    return (
        neat_config.coefficient_common_conn_weight * diff_weight
        + neat_config.coefficient_disjoint_conn * disjoint_rate
    )


def calculate_species(
    neat_config: NEATConfig,
    genomes: GenomePopulation,
    key: jax.Array,
) -> jax.Array:
    """Calculate the species of the genomes."""

    one = jax.random.randint(key, (), 0, neat_config.pop_size)
    calculate_distance_vmapped = jax.vmap(calculate_distance, in_axes=(None, 0, None))

    idx = jnp.array([one])
    distance: None | jax.Array = None

    for i in range(neat_config.num_species):
        selected = jax.tree.map(lambda x: x[idx[-1]], genomes)
        new_distance = jnp.array(
            [calculate_distance_vmapped(neat_config, genomes, selected)]
        )
        distance = (
            new_distance
            if distance is None
            else jnp.append(distance, new_distance, axis=0)
        )
        jax.debug.print("distance: {}", distance)
        if i == neat_config.num_species - 1:
            break
        furthest = jnp.argmax(jnp.prod(distance, axis=0))
        idx = jnp.append(idx, furthest)

    return jnp.argmin(distance, axis=0)


def generate_new_population_fn(
    neat_config: NEATConfig,
    fitness: jax.typing.ArrayLike,
    pop: GenomePopulation,
    key: jax.Array,
) -> GenomePopulation:
    """Generate a new population."""
    # Calculate the best genomes.
    k = math.ceil(neat_config.pop_size * neat_config.rate_keep_best)
    _, top_idx = jax.lax.top_k(fitness, k)
    top_genomes = jax.tree_map(lambda x: x[top_idx], pop)

    # Repeat the top genomes.
    n_repeats = (neat_config.pop_size + k - 1) // k
    pop = jax.tree_map(
        lambda x: jnp.repeat(
            x, n_repeats, axis=0, total_repeat_length=neat_config.pop_size
        ),
        top_genomes,
    )

    new_edge_id = jnp.max(pop.conn_ids) + 1

    keys = jax.random.split(key, neat_config.pop_size)

    # Mutate the genes. To assign the edge id sequentially, we use scan.
    def mutate(
        carry: tuple[int, jax.Array], genome: Genome
    ) -> tuple[tuple[int, jax.Array], Genome]:
        (i, new_edge_id) = carry
        (genome, new_edge_id) = mutate_genome(neat_config, genome, keys[i], new_edge_id)
        return ((i + 1, new_edge_id), genome)

    _, pop = jax.lax.scan(mutate, (0, new_edge_id), pop, neat_config.pop_size)

    return pop


def ask_fn(neat_config: NEATConfig) -> GenomePopulation:
    """Generate a new population.

    Returns:
        vmapped Genome.
    """

    # For now create an empty genome.
    genome = create_empty_genome_fn(neat_config.config)

    # Repeat the genome for the population size.
    return jax.tree.map(lambda x: jnp.tile(x, (neat_config.pop_size, 1)), genome)


def get_best_params_fn(fitness: jax.typing.ArrayLike, pops: GenomePopulation) -> Genome:
    maxi = jax.lax.argmax(fitness, 0, jnp.int32)
    best_params = jax.tree.map(lambda leaf: leaf[maxi], pops)
    # Calculate next generation.

    return best_params


class NEAT(NEAlgorithm):

    def __init__(self, config: NEATConfig, key: jax.Array = jax.random.key(0)):
        self.pop_size = config.pop_size
        self._config = config

        self._pops = ask_fn(config)

        self._convert_to_array = jax.jit(convert_to_array_fn)

        self._get_best_params = jax.jit(get_best_params_fn)
        self._generate_new_population = jax.jit(
            generate_new_population_fn, static_argnums=(0)
        )
        self._key = key

    def ask(self):
        return self._convert_to_array(self._pops)

    def tell(self, fitness: jax.typing.ArrayLike):
        # Select the best parameters.
        self._best_genome = self._get_best_params(fitness, self._pops)
        self._best_params = self._convert_to_array(self._best_genome)

        # Generate a new population.
        self._key, key = jax.random.split(self._key)
        self._pops = self._generate_new_population(
            self._config, fitness, self._pops, key
        )
        jax.debug.print(
            """\
            node_counts: {}
            edge_counts: {}
            """,
            jnp.count_nonzero(self._pops.node_ids > 0),
            jnp.count_nonzero(self._pops.conn_ids > 0),
        )

    @property
    def best_params(self) -> jax.Array:
        return self._best_params

    @best_params.setter
    def best_params(self, params: jax.typing.ArrayLike):
        self._best_params = jnp.array(params)

    @property
    def best_genome(self) -> Genome:
        return self._best_genome


##############################################################
# Example usage.
##############################################################

log_dir = "./log/slimevolley_" + datetime.now().strftime("%Y%m%d-%H%M%S")
logger = util.create_logger(name="SlimeVolley", log_dir=log_dir, debug=False)


def test3():
    # SlimeVolley
    task = SlimeVolley()

    assert len(task.obs_shape) == 1
    assert len(task.act_shape) == 1
    config = Config(
        input_dim=task.obs_shape[0],
        output_dim=task.act_shape[0],
        max_nodes=50,
        max_edges=200,
        max_depth=10,
    )
    neat_config = NEATConfig(config, pop_size=300)
    policy = NEATPolicy(config)
    solver = NEAT(neat_config)

    max_steps = 3000
    train_task = SlimeVolley(test=False, max_steps=max_steps)
    test_task = SlimeVolley(test=True, max_steps=max_steps)

    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=30,
        log_interval=1,
        test_interval=10,
        n_repeats=1,
        n_evaluations=1,
        seed=42,
        log_dir=log_dir,
        logger=logger,
    )

    trainer.run(demo_mode=False)

    # Visualize the policy.
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)
    best_params = jax.tree.map(lambda l: l[None, :], trainer.solver.best_params)
    key = jax.random.PRNGKey(0)[None, :]
    print(genome_to_mermaid(solver.best_genome, config, show_disabled=True))
    print(best_params)

    task_state = task_reset_fn(key)
    policy_state = policy_reset_fn(task_state)
    screens = []
    for _ in range(max_steps):
        action, policy_state = action_fn(task_state, best_params, policy_state)
        task_state, reward, done = step_fn(task_state, action)
        screens.append(SlimeVolley.render(task_state))

    gif_file = os.path.join(log_dir, "slimevolley.gif")
    screens[0].save(
        gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0
    )
    logger.info("GIF saved to {}.".format(gif_file))


if __name__ == "__main__":
    # test2()
    # test3()
    pass

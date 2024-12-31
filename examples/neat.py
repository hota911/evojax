from dataclasses import dataclass
import dataclasses
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

from evojax.task.slimevolley import SlimeVolley
from evojax.trainer import Trainer
from evojax.util import get_params_format_fn


# Data classes for the NEAT algorithm.
@jax.tree_util.register_dataclass
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
    conn_weights = jnp.ones(conns)
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


# def get_params_format_fn(config: Config):
#     """Generate the number of parameters and format function.

#     Parameters of NetworkPolicy must be Array and cannot be Pytree.
#     Hence, we need to flatten the Pytree to Array.
#     """
#     init_params = create_empty_genome_fn(config)
#     flat, tree = jax.tree.flatten(init_params)
#     params_sizes = np.cumsum([np.prod(p.shape) for p in flat])

#     def params_format_fn(params: jnp.ndarray) -> Genome:
#         params = jax.tree.map(
#             lambda x, y: x.reshape(y.shape),
#             jnp.split(params, params_sizes, axis=-1)[:-1],
#             flat,
#         )
#         return jax.tree.unflatten(tree, params)

#     return params_sizes[-1], params_format_fn


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
    print("Forward fn")
    print(x.shape)
    print(config.max_nodes - x.shape[0])
    x = jnp.pad(x, (0, config.max_nodes - x.shape[0]))
    print(x.shape)
    adj = get_adjacency_matrix_fn(config, genome)
    for _ in range(config.max_depth):
        x = matmul_fn(adj, x)

        # Apply the activation function.
        def activate_fn(activation, x):
            return jax.lax.switch(activation, ACTIVATION_FUNCTIONS, x)

        activate = jax.vmap(activate_fn)
        print(genome.node_activation.shape)
        print(genome.node_ids.shape)
        x = activate(genome.node_activation, x)
    return x


# For testing
forward = jax.jit(jax.vmap(forward_fn, in_axes=(None, 0, 0)), static_argnums=(0,))


@dataclass
class NEATPolicy(PolicyNetwork):
    def __init__(self, config: Config):
        self._config = config

        def forward_with_config_fn(obs: jax.Array, params: Genome):
            print(obs)
            result = forward_fn(config, params, obs)
            return result[config.input_dim : config.input_dim + config.output_dim]

        self._forward_with_config_fn = jax.jit(jax.vmap(forward_with_config_fn))

        # Get the number of parameters
        # self.num_params, format_params_fn = get_params_format_fn(config)
        # self._format_params = jax.jit(jax.vmap(format_params_fn))

    def get_actions(
        self, t_states: TaskState, params: Genome, p_states: PolicyState
    ) -> tuple[jax.Array, PolicyState]:
        print(params.node_ids.shape)
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


def ask_fn(neat_config: NEATConfig):
    """Generate a new population."""

    # For now create an empty genome.
    genome = create_empty_genome_fn(neat_config.config)
    # Repeat the genome for the population size.
    return jax.tree.map(lambda x: jnp.tile(x, (neat_config.pop_size, 1)), genome)


# Type alias for the genome population.
# To make it easier to understand the genome is only a single genome or a population of genomes.
GenomePopulation = Genome


def tell_fn(fitness: jax.typing.ArrayLike, pops: GenomePopulation) -> Genome:
    maxi = jax.lax.argmax(fitness, 0, jnp.int32)
    best_params = jax.tree.map(lambda leaf: leaf[maxi], pops)

    # Calculate next generation.

    return best_params


@jax.jit
def pick_one(key: jax.Array, mask: jax.Array) -> int:
    """Pick up one index where the mask is truefp.

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
    conn_idx = pick_one(key, genome.conn_enabled)

    # Get the selected connection values
    in_node = genome.conn_in[conn_idx]
    out_node = genome.conn_out[conn_idx]
    weight = genome.conn_weights[conn_idx]

    # Add a new node
    new_node_id = genome.node_ids.max() + 1

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
        Genome(
            node_ids=jnp.where(
                new_node_id >= config.max_nodes, genome.node_ids, node_ids
            ),
            node_activation=jnp.where(
                new_node_id >= config.max_nodes, genome.node_activation, node_activation
            ),
            conn_ids=jnp.where(
                new_node_id >= config.max_nodes, genome.conn_ids, conn_ids
            ),
            conn_in=jnp.where(new_node_id >= config.max_nodes, genome.conn_in, conn_in),
            conn_out=jnp.where(
                new_node_id >= config.max_nodes, genome.conn_out, conn_out
            ),
            conn_weights=jnp.where(
                new_node_id >= config.max_nodes, genome.conn_weights, conn_weights
            ),
            conn_enabled=jnp.where(
                new_node_id >= config.max_nodes, genome.conn_enabled, conn_enabled
            ),
        ),
        next_edge_id + 2,
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

    jax.debug.print("src: {}, dst: {}", src, dst)

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
    jax.debug.print("valid: {}", valid)
    jax.debug.print("depth: {}", calculate_depth(config, conn_in, conn_out))
    # - there is no cycle
    valid = jnp.logical_and(valid, calculate_depth(config, conn_in, conn_out) != -1)
    jax.debug.print("valid: {}", valid)

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
    jax.debug.print("p: {}, index: {}", p, index)

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


def generate_new_population_fn(
    neat_config: NEATConfig, fitness: jax.typing.ArrayLike, pop: GenomePopulation
) -> GenomePopulation:
    """Generate a new population."""
    # Calculate the best genomes.
    k = math.ceil(neat_config.pop_size * neat_config.rate_keep_best)
    _, top_idx = jax.lax.top_k(fitness, k)
    top_genomes = jax.tree_map(lambda x: x[top_idx], pop)

    # Repeat the top genomes.
    n_repeats = (neat_config.pop_size + k - 1) // k
    return jax.tree_map(
        lambda x: jnp.repeat(
            x, n_repeats, axis=0, total_repeat_length=neat_config.pop_size
        ),
        top_genomes,
    )


class NEAT(NEAlgorithm):
    def __init__(self, config: NEATConfig):
        self.pop_size = config.pop_size
        self._config = config

        self._ask = jax.jit(ask_fn, static_argnums=(0))

        self._tell = jax.jit(tell_fn)

    def ask(self):
        self._pops = self._ask(self._config)
        return self._pops

    def tell(self, fitness: jax.typing.ArrayLike):
        # Select the best parameters.
        self._best_params = self._tell(fitness, self._pops)

    @property
    def best_params(self):
        return self._best_params

    @best_params.setter
    def best_params(self, params: jax.typing.ArrayLike):
        self._best_params = jnp.array(params)


##############################################################
# Example usage.
##############################################################


# def test1():
#     config = Config(input_dim=2, output_dim=1, max_nodes=10, max_edges=20, max_depth=10)

#     # 0 1
#     # | |
#     # 4 3
#     #  \|
#     #   2
#     node_ids = jnp.array([0, 1, 2, 3, 4])
#     node_activation = jnp.array([0, 0, 2, 0, 0])
#     conn_ids = jnp.array([0, 1, 2, 3])
#     conn_in = jnp.array([0, 1, 4, 3])
#     conn_out = jnp.array([4, 3, 2, 2])
#     conn_weights = jnp.array([1.0, 1.0, 1.0, 1.0])
#     conn_enabled = jnp.array([True, True, True, True])
#     genome = create_genome(
#         config,
#         node_ids,
#         node_activation,
#         conn_ids,
#         conn_in,
#         conn_out,
#         conn_weights,
#         conn_enabled,
#     )
#     genome = jax.tree.map(lambda x: jnp.array([x]), genome)

#     x = jnp.stack([jnp.array([1, 1])])
#     print(forward(config, genome, x))

#     # SlimeVolley
#     task = SlimeVolley()

#     task_reset = jax.jit(task.reset)
#     t_state = task_reset(jax.random.PRNGKey(seed=0)[None, :])

#     policy = NEATPolicy(config)
#     p_state = policy.reset(t_state)
#     get_actions = jax.jit(jax.vmap(policy.get_actions))
#     output, p_state = get_actions(t_state, genome, p_state)
#     output.block_until_ready()

#     assert output.shape == (1, 1)
#     np.testing.assert_allclose(output, jnp.array([[0.880797]]), atol=1e-5)


def test2():
    # SlimeVolley
    task = SlimeVolley()

    task_reset = jax.jit(task.reset)
    t_state = task_reset(jax.random.split(jax.random.PRNGKey(seed=0), 3))

    assert len(task.obs_shape) == 1
    assert len(task.act_shape) == 1
    config = Config(
        input_dim=task.obs_shape[0],
        output_dim=task.act_shape[0],
        max_nodes=100,
        max_edges=1000,
        max_depth=10,
    )
    neat_config = NEATConfig(config, 3)
    print(neat_config)
    neat = NEAT(neat_config)
    pops = neat.ask()
    print(pops)

    policy = NEATPolicy(config)
    p_state = policy.reset(t_state)
    get_actions = jax.jit(policy.get_actions)
    output, p_state = get_actions(t_state, pops, p_state)

    print("-------------------")
    print(output.block_until_ready())
    print("-------------------")


log_dir = "./log/slimevolley"
logger = util.create_logger(name="SlimeVolley", log_dir=log_dir, debug=True)


def test3():
    # SlimeVolley
    task = SlimeVolley()

    assert len(task.obs_shape) == 1
    assert len(task.act_shape) == 1
    config = Config(
        input_dim=task.obs_shape[0],
        output_dim=task.act_shape[0],
        max_nodes=100,
        max_edges=1000,
        max_depth=10,
    )
    neat_config = NEATConfig(config, 100)
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
        max_iter=10,
        log_interval=3,
        test_interval=2,
        n_repeats=3,
        n_evaluations=4,
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

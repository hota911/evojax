from dataclasses import dataclass
import jax
from evojax.algo.base import NEAlgorithm
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.task.base import TaskState

import jax.numpy as jnp

from evojax.task.slimevolley import SlimeVolley


# Data classes for the NEAT algorithm.
@jax.tree_util.register_dataclass
@dataclass
class Genome:
    """Stores node and connection info in JAX arrays."""

    node_ids: jax.Array  # shape [num_nodes]
    # 0: relu
    node_activation: jax.Array  # shape [num_nodes]

    # Connections.
    conn_id: jax.Array  # shape [num_connections]
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
    conn_id,
    conn_in,
    conn_out,
    conn_weights,
    conn_enabled,
) -> Genome:
    node_pad_config = [(0, config.max_nodes - node_ids.shape[0], 0)]
    node_ids = jax.lax.pad(node_ids, -1, node_pad_config)
    node_activation = jax.lax.pad(node_activation, -1, node_pad_config)

    conn_pad_config = [(0, config.max_edges - conn_id.shape[0], 0)]
    conn_id = jax.lax.pad(conn_id, -1, conn_pad_config)
    conn_in = jax.lax.pad(conn_in, -1, conn_pad_config)
    conn_out = jax.lax.pad(conn_out, -1, conn_pad_config)
    conn_weights = jax.lax.pad(conn_weights, jnp.nan, conn_pad_config)
    conn_enabled = jax.lax.pad(conn_enabled, False, conn_pad_config)
    return Genome(
        node_ids,
        node_activation,
        conn_id,
        conn_in,
        conn_out,
        conn_weights,
        conn_enabled,
    )


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


##############################################################
# NEAT algorithm.
##############################################################
@dataclass
class NEATPolicy(PolicyNetwork):
    def __init__(self, config: Config):
        self._config = config

        def forward_with_config_fn(t_states, params: Genome):
            result = forward_fn(config, params, t_states)
            return result[config.input_dim : config.input_dim + config.output_dim]

        self._forward_with_config_fn = jax.jit(forward_with_config_fn)

    def get_actions(self, t_states, params: Genome, p_states):
        return self._forward_with_config_fn(t_states, params), p_states


class NEAT(NEAlgorithm):
    def __init__(self, config: Config):
        self._config = config

        def ask_fn():
            return None

        self._ask = jax.jit(ask_fn)

        def tell_fn(fitness):
            return None

        self._tell = jax.jit(tell_fn)

    def ask(self):
        return self._ask()

    def tell(self, fitness: jax.typing.ArrayLike):
        return self._tell(fitness)


##############################################################
# Example usage.
##############################################################

if __name__ == "__main__":
    config = Config(input_dim=2, output_dim=1, max_nodes=10, max_edges=20, max_depth=10)

    # 0 1
    # | |
    # 4 3
    #  \|
    #   2
    node_ids = jnp.array([0, 1, 2, 3, 4])
    node_activation = jnp.array([0, 0, 2, 0, 0])
    conn_id = jnp.array([0, 1, 2, 3])
    conn_in = jnp.array([0, 1, 4, 3])
    conn_out = jnp.array([4, 3, 2, 2])
    conn_weights = jnp.array([1.0, 1.0, 1.0, 1.0])
    conn_enabled = jnp.array([True, True, True, True])
    genome = create_genome(
        config,
        node_ids,
        node_activation,
        conn_id,
        conn_in,
        conn_out,
        conn_weights,
        conn_enabled,
    )
    genome = jax.tree.map(lambda x: jnp.array([x]), genome)

    x = jnp.stack([jnp.array([1, 1])])
    print(forward(config, genome, x))

    # SlimeVolley
    task = SlimeVolley()

    task_reset = jax.jit(task.reset)
    t_state = task_reset(jax.random.PRNGKey(seed=0)[None, :])

    policy = NEATPolicy(config)
    p_state = policy.reset(t_state)
    get_actions = jax.jit(jax.vmap(policy.get_actions))
    print(get_actions(x, genome, p_state))

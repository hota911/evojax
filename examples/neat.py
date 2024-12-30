from dataclasses import dataclass
import jax
import numpy as np
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


def ask_fn(neat_config: NEATConfig):
    """Generate a new population."""

    # For now create an empty genome.
    genome = create_empty_genome_fn(neat_config.config)
    # Repeat the genome for the population size.
    return jax.tree.map(lambda x: jnp.tile(x, (neat_config.pop_size, 1)), genome)


def tell_fn(fitness):
    return None


class NEAT(NEAlgorithm):
    def __init__(self, config: NEATConfig):
        self.pop_size = config.pop_size
        self._config = config

        self._ask = jax.jit(ask_fn, static_argnums=(0))

        self._tell = jax.jit(tell_fn)

    def ask(self):
        self._params = self._ask(self._config)
        return self._params

    def tell(self, fitness: jax.typing.ArrayLike):
        maxi = jax.lax.argmax(fitness, 0, jnp.int32)

        # Select the best parameters.
        self._best_params = jax.tree.map(lambda leaf: leaf[maxi], self._params)

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
    neat_config = NEATConfig(config, 3)
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
        max_iter=20,
        log_interval=4,
        test_interval=5,
        n_repeats=3,
        n_evaluations=4,
        seed=42,
        log_dir="./log",
        logger=None,
    )

    trainer.run(demo_mode=False)


if __name__ == "__main__":
    test2()
    test3()

import os
import pickle
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional

import flashbax as fbx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optax
import ott.solvers.utils as solver_utils
import ott.utils as ott_utils
from flax import linen as nn
from flax import struct
from flax.training import train_state
from tqdm import tqdm

import method.dynamics as dynamics

__all__ = ["HOTA"]


Callback_t = Callable[[int, ], None]

# multigpu
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

NPROC = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))

P = PartitionSpec
mesh = Mesh(mesh_utils.create_device_mesh((NPROC,)), axis_names=('data',))

def with_mesh(f):
    def wrapper(*args, **kwargs):
        with mesh:
            return f(*args, **kwargs)
    return wrapper

# buffer
class TrajectoryBuffer:
    def __init__(
        self,
        capacity: int,
        dim: int,
        batch_size: int,
        seed: int=0
    ):
        buffer = fbx.make_item_buffer(
            min_length=1,
            max_length=capacity,
            sample_batch_size=batch_size,
            add_batches=True,
        )
        buffer = buffer.replace(
            init = jax.jit(buffer.init),
            add = jax.jit(buffer.add, donate_argnums=0),
            sample = jax.jit(buffer.sample),
            can_sample = jax.jit(buffer.can_sample),
        )

        init_sample = {"x": np.random.randn(dim), "t": np.random.randn()}
        state = buffer.init(init_sample)

        self.buffer = buffer
        self.state = state
        self.rng = jax.random.key(seed)

    def append(self, x: np.ndarray, t: np.ndarray):
        self.state = self.buffer.add(
            self.state,
            {"x": x, "t": t}
        )

    def sample(self):
        self.rng, key = jax.random.split(self.rng)
        return self.buffer.sample(self.state, key).experience

class TimedX(struct.PyTreeNode):
  t: jnp.ndarray
  x: jnp.ndarray

class HOTA:

  def __init__(
      self,
      input_dim: int,
      value_model: nn.Module,
      optimizer: Optional[optax.GradientTransformation],
      flow: dynamics.LagrangianFlow,
      potential_weight: float,
      control_weight: float,
      reg_weight: float,
      acc_weight: float,
      pretrain_steps: float = 5_000,
      time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
      key:Optional[jax.Array] = None,
      **kwargs: Any,
  ):
    self.value_model = value_model
    self.flow = flow
    self.time_sampler = time_sampler
    self.potential_weight = potential_weight
    self.control_weight = control_weight
    self.reg_weight = reg_weight
    self.acc_weight = acc_weight
    self.pretrain_steps = pretrain_steps

    with mesh:
      key, init_key = jax.random.split(key, 2)
      params = value_model.init(
        init_key,
        jnp.ones([1, 1]),
        jnp.ones([1, input_dim]),
        jnp.ones([1, input_dim])
      )

      self.state = train_state.TrainState.create(
        apply_fn=value_model.apply,
        params=params,
        tx=optimizer
      )

      self.target_state = train_state.TrainState.create(
        apply_fn=value_model.apply,
        params=jax.tree.map(lambda x: jnp.copy(x), params),
        tx=optax.identity()
      )

    self.train_step_fast, self.train_step_cost = self._get_step_fn()
    self.inference = self.get_inference()


  def reset(
      self,
      optimizer: Optional[optax.GradientTransformation],
      potential_weight: float,
      control_weight: float,
      reg_weight: float,
      acc_weight: float
  ):

    self.potential_weight = potential_weight
    self.control_weight = control_weight
    self.reg_weight = reg_weight
    self.acc_weight = acc_weight
    self.pretrain_steps = 0

    params = self.state.params

    self.state = train_state.TrainState.create(
      apply_fn=self.value_model.apply,
      params=params,
      tx=optimizer
    )

    self.target_state = train_state.TrainState.create(
      apply_fn=self.value_model.apply,
      params=jax.tree.map(lambda x: jnp.copy(x), params),
      tx=optax.identity()
    )

    self.train_step_fast, self.train_step_cost = self._get_step_fn()
    self.inference = self.get_inference()


  def _get_step_fn(self) -> Callable:
      def am_loss_sample(state, params, key_t, t_sample, x_sample, target_state, reg_weight):
        
        x_t = x_sample
        t = t_sample.reshape(-1, 1)
        At_T = self.flow.compute_inverse_control_matrix(t, x_t).transpose()

        dsdtdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=[1,2])

        dsdt, dsdx = dsdtdx_fn(params, t, x_t, x_t)
        dsdt_tgt, dsdx_tgt = dsdtdx_fn(target_state.params, t, x_t, x_t)
        u = dsdx_tgt
        vt = dsdt_tgt

        dt = 1.0 / 100
        x_dt = x_t - jax.lax.stop_gradient(dsdx) * dt
        x_2dt = x_t - jax.lax.stop_gradient(dsdx) * dt * 2
        U_t = 0.4 * self.flow.compute_potential(t, x_t) + 0.3 * self.flow.compute_potential(t+dt, x_dt) + 0.3 * self.flow.compute_potential(t+dt*2, x_2dt)

        @partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def laplacian(p, t, x, x0):
            grad_fun = jax.grad(lambda __x: state.apply_fn(p, t[None], __x[None], x0[None]).sum())

            def hessian_diag(__x):
                grad_val = jax.jvp(grad_fun, (__x,), (jnp.ones_like(__x),))[1]
                return grad_val

            trace = jnp.sum(hessian_diag(x))
            return trace

        def normalize(x):
          norm = jnp.linalg.norm(x) + 1e-8
          return x / norm

        @partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def acceleration(p, t, x, x0):
            fun = lambda __t, __x: state.apply_fn(p,__t,__x,x0).sum()
            dsdx_fn = jax.grad(fun, argnums=1)
            norm_rev = lambda __t, __x: normalize(jax.jacrev(fun, 1)(__t, __x))
            Dt, Dx = jax.jacfwd(norm_rev, argnums=[0, 1])(t, x)
            acc = Dt.squeeze() - Dx @ dsdx_fn(t, x)
            return acc

        a = acceleration(params, t, x_t, x_t)
        a_cost = jnp.sqrt((a * a).reshape(x_t.shape[0], x_t.shape[1]).sum(-1, keepdims=True) + 1e-8) * self.acc_weight
        D = (0.5 * self.flow.compute_sigma_t(t) ** 2).reshape(-1, 1)
        s_diff_1 = dsdt - 0.5 * ((u @ At_T) * u).sum(-1, keepdims=True) + self.potential_weight * U_t.reshape(-1, 1) + a_cost + D * laplacian(state.params, t, x_t, x_t).reshape(-1, 1)
        s_diff_2 = vt - 0.5 * ((dsdx @ At_T) * dsdx).sum(-1, keepdims=True) + self.potential_weight * U_t.reshape(-1, 1) + a_cost + D * laplacian(params, t, x_t, x_t).reshape(-1, 1)
        loss = jnp.abs(s_diff_1 ** 2).mean() + jnp.abs(s_diff_2 ** 2).mean()
        return loss

      def potential_loss(state, params, key, steps_count, weight, source, target):
        bs = source.shape[0]
        t_0, t_1 = jnp.zeros([bs, 1]), jnp.ones([bs, 1])
        x_0, x_1 = source, target
        dt = 1.0 / steps_count

        dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)

        def move(carry, _):
          t_, x_, key_ = carry
          dsdx = dsdx_fn(state.params, t_, x_, x_0)
          sigma = self.flow.compute_sigma_t(t_)
          key_, key_s = jax.random.split(key_)
          x_next = x_ - dt * dsdx + sigma * jax.random.normal(key_s, shape=x_.shape) * jnp.sqrt(dt)
          t_next = t_ + dt

          return (t_next, x_next, key_), TimedX(t_, x_)

        (_, x_last ,_), result = jax.lax.scan(move, (t_0, x_0, key), None, length=steps_count)
        x_1_pred = jax.lax.stop_gradient(x_last)

        dual_loss = - (-state.apply_fn(params, t_1, x_1, x_0 * 0) + state.apply_fn(params, t_1, x_1_pred, x_0 * 0))
        dual_loss = (dual_loss.mean() * jnp.abs(dual_loss.mean()))

        return dual_loss * weight, result

      @with_mesh
      @jax.jit
      def train_step_cost(state, key, source, target, t_sample, x_sample, target_state, reg_weight, scale):
        source = jax.lax.with_sharding_constraint(source, P('data'))
        target = jax.lax.with_sharding_constraint(target, P('data'))
        t_sample = jax.lax.with_sharding_constraint(t_sample, P('data'))
        x_sample = jax.lax.with_sharding_constraint(x_sample, P('data'))

        grad_fn = jax.value_and_grad(am_loss_sample, argnums=1, has_aux=False)
        loss, control_grads = grad_fn(state, state.params, key, t_sample, x_sample, target_state, reg_weight)

        grad_fn = jax.value_and_grad(potential_loss, argnums=1, has_aux=True)
        (loss_potential, x_seq), potential_grads = grad_fn(state, state.params, key, 30, 1.0, source, target)

        g_norm_control = optax.global_norm(control_grads)
        g_norm_potential = optax.global_norm(potential_grads)
        scale_update = g_norm_potential / g_norm_control
        scale = scale_update * 0.1 + scale * 0.9

        state = state.apply_gradients(
          grads=jax.tree.map(lambda gc, gp: gc * scale * self.control_weight + gp, control_grads, potential_grads)
        )

        new_target_params = optax.incremental_update(state.params, target_state.params, 0.01)
        target_state = target_state.replace(params=new_target_params)

        return state, loss, loss_potential, x_seq, target_state, g_norm_control,  g_norm_potential, scale


      @with_mesh
      @jax.jit
      def train_step_fast(state, key, source, target, t_sample, x_sample, target_state, reg_weight, scale):
        t_sample = jax.lax.with_sharding_constraint(t_sample, P('data'))
        x_sample = jax.lax.with_sharding_constraint(x_sample, P('data'))

        grad_fn = jax.value_and_grad(am_loss_sample, argnums=1, has_aux=False)
        loss, control_grads = grad_fn(state, state.params, key, t_sample, x_sample, target_state, reg_weight)

        state = state.apply_gradients(
          grads=jax.tree.map(lambda gc: gc * scale * self.control_weight, control_grads)
        )

        new_target_params = optax.incremental_update(state.params, target_state.params, 0.01)
        target_state = target_state.replace(params=new_target_params)

        return state, loss, target_state
      return train_step_fast, train_step_cost

  def __call__(  # noqa: D102
      self,
      loader: Iterable[Dict[str, np.ndarray]],
      *,
      n_iters: int,
      rng: Optional[jax.Array] = None,
      callback: Optional[Callback_t] = None,
      eval_every: int = 5_000,
      save_dir: str = None,
  ) -> Dict[str, List[float]]:
    batch_size, input_dim = next(iter(loader))["src_lin"].shape
    self.buffer = TrajectoryBuffer(
      capacity=100_000,
      dim=input_dim,
      batch_size=batch_size,
    )

    loop_key = ott_utils.default_prng_key(rng)
    training_logs = {"cost_loss": [], "potential_loss": [], "g_norm": [], "g_norm_potential" : []}
    it = 0
    g_norm, g_norm_potential = 0, 0
    scale = jnp.array(0.1)
    pbar = tqdm(loader, total=n_iters, colour='green', dynamic_ncols=True)
    for batch in pbar:
      src, tgt = batch["src_lin"], batch["tgt_lin"]
      it_key = jax.random.fold_in(loop_key, it)

      if it > self.pretrain_steps:
          _sample = self.buffer.sample()
          x_sample, t_sample = _sample["x"], _sample["t"]
          reg_weight = self.reg_weight
      else:
          bs = src.shape[0]
          t_sample = self.time_sampler(it_key, bs)
          x_sample = self.flow.compute_xt(it_key, t_sample, src, tgt)
          reg_weight = 0.0
      if it % 2 == 0:
        self.state, loss, loss_potential, tx_seq, self.target_state, g_norm, g_norm_potential, scale = self.train_step_cost(
          self.state, it_key, src, tgt, t_sample, x_sample, self.target_state, reg_weight, scale
        )
      else:
        self.state, loss, self.target_state = self.train_step_fast(
          self.state, it_key, src, tgt, t_sample, x_sample, self.target_state, reg_weight, scale
        )
        tx_seq = None

      if it % 2 == 0:
        training_logs["potential_loss"].append(loss_potential)
        training_logs["cost_loss"].append(loss)
        training_logs["g_norm"].append(g_norm)
        training_logs["g_norm_potential"].append(g_norm_potential)

        traj_idx = np.random.randint(tx_seq.x.shape[1])
        x_seq = tx_seq.x[:, traj_idx].reshape(-1, tx_seq.x.shape[-1])
        t_seq = tx_seq.t[:, traj_idx].reshape(-1)
        self.buffer.append(x=x_seq, t=t_seq)

      if it % 100 == 0 and it > 0:
        pbar.set_postfix({"pot_loss": loss_potential,
                          "cost_loss": loss,
                          "g_norm": g_norm, 
                          "g_norm_potential": g_norm_potential})

      if it % eval_every == 0 and it > 0 and callback is not None:
        callback(it, training_logs, self.transport)
        if save_dir is not None:
          self.save(save_dir, it=it)

      it += 1
      if it >= n_iters:
        break

    return training_logs

  def save(self, save_dir: str, it: int):
    with open(f"{save_dir}/opt_state_step_{it}.pkl", "wb") as file:
      pickle.dump(self.state.opt_state, file)
    with open(f"{save_dir}/params_step_{it}.pkl", "wb") as file:
      pickle.dump(self.state.params, file)
    with open(f"{save_dir}/opt_state_latest.pkl", "wb") as file:
      pickle.dump(self.state.opt_state, file)
    with open(f"{save_dir}/params_latest.pkl", "wb") as file:
      pickle.dump(self.state.params, file)
    with open(f"{save_dir}/step_latest.pkl", "wb") as file:
      pickle.dump(self.state.step, file)
    with open(f"{save_dir}/buffer_state_latest.pkl", "wb") as file:
      pickle.dump(self.buffer.state, file)

  def get_inference(self):

    dt = 1.0 / 30
    t_0 = 0.0
    n = 30

    @jax.jit
    def inference(state, x_0, loop_key):

      dsdx_fn = jax.grad(lambda p, t, x, x0: state.apply_fn(p,t,x,x0).sum(), argnums=2)

      def move(carry, _):
        t_, x_, cost, key_ = carry
        u = dsdx_fn(state.params, t_ * jnp.ones([x_0.shape[0],1]), x_, x_0)
        U_t = self.flow.compute_potential(t_, x_)
        sigma = self.flow.compute_sigma_t(t_)
        key_, key_s = jax.random.split(key_)
        x_ = x_ - dt * u + sigma * jax.random.normal(key_s, shape=x_.shape) * jnp.sqrt(dt)
        t_ = t_ + dt
        cost += 0.5 * (u * u).sum(-1).mean() * dt + U_t.mean() * dt 
        return (t_, x_, cost, key_), x_

      (_, _, cost, _), result = jax.lax.scan(move, (t_0, x_0, 0.0, loop_key), None, length=n)

      return cost, result

    return inference

  def transport(
      self,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      **kwargs: Any,
  ) -> jnp.ndarray:

    loop_key = jax.random.PRNGKey(0)
    t_0 = 0.0
    n = 30
    dt = 1.0 / 30

    cost, result = self.inference(self.state, x, loop_key)
    x_seq = [TimedX(t=t_0, x=x)]

    for i in range(n):
      t_ = x_seq[-1].t + dt
      x_seq.append(TimedX(t=t_, x=result[i]))

    return cost, x_seq

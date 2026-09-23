# Interfaces

Adapters that put the simulator behind other libraries' environment APIs. Every
adapter uses the v2 action space: a v2 policy read as v1 fails silently.

The action is `MultiDiscrete([22, 33, 122, 5, 15, 4])`, meaning
`[op, target, placement, direction, item, amount]`. The legal-action mask is the
per-dimension masks joined in that order, 201 entries in all. Index 0 of every
argument dimension is an UNUSED sentinel that is always legal. The mask is
factorised: every value it allows is legal in its own dimension, but some
combinations still fail to decode, and the simulator counts those as
`decode_failures`.

## Gymnasium

`pip install factory-sim[gym]` (Gymnasium ≥ 1.0). The code is in `fsim/gym_env.py`.

```python
import gymnasium as gym
import fsim.gym_env  # registers the ids

env = gym.make("fsim/BuildLine-v0", split="test")   # held-out family
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

vec = gym.make_vec("fsim/ConstructSmeltingLine-v0", num_envs=64,
                   vectorization_mode="vector_entry_point", threads=8)
```

- **Ids:** `fsim/ConstructSmeltingLine-v0`, `fsim/BuildLine-v0` and `fsim/PlateLine-v0`.
  `split` is `"train"` or `"test"`. The per-task budgets are FactorioRL's:
  600 decisions for the two construction tasks, and 400 decisions with 24,000
  ticks for `plate_line`. The C env truncates on its own, so no `TimeLimit` is added.
- **`FactorySimEnv`** wraps `RlEnv`.
  - The observation space is a `Dict` of `Box`es: `grid (6,65,65) f32 [0,1]`,
    `entities (32,16) f32 [-1,1]`, `entity_mask (32,) i8 {0,1}`,
    `self (12,) f32 [-1,1]`, `inventory (14,) f32 [0,1]` and `goal (12,) f32 [-1,1]`.
  - `reset(seed=s)` seeds Gymnasium's RNG, and each reset draws its scene seed
    for `scenes.sample` from that RNG. `options={"scene_seed": k}` picks a
    scene directly.
  - The environment passes `gymnasium.utils.env_checker.check_env`.
  - `render_mode="rgb_array"` draws the grid planes and entities.
- **Masks:** `info["action_mask"]` and `env.action_masks()` both give the flat
  (201,) boolean mask. `action_masks()` is the method sb3-contrib's
  `MaskablePPO` calls. `split_mask(mask)` returns one array per dimension, and
  `sample_masked(mask, rng)` draws a random legal action.
- **`FactorySimVectorEnv`** is a `gymnasium.vector.VectorEnv` over `VecEnv`
  that steps every environment in C on a thread pool.
  - **Autoreset mode: `AutoresetMode.SAME_STEP`**, which is what `VecEnv` does
    natively. When an episode ends, the observation and mask that `step`
    returns already belong to the next episode. The finished episode's last
    observation is in `info["final_obs"][i]` and its summary (return, length,
    success and so on, plus its last mask) is in `info["final_info"][i]`.
    `info["_final_obs"]` marks the slots that finished.
  - `reset(seed=s)` gives slot `i` its first scene from seed
    `s * 1_000_003 + i`, and later scenes continue that counter.
  - `copy=False` returns the C buffers' strided views without copying. The
    next step overwrites them.

## PufferLib

The code is in `fsim/puffer_env.py`, and it was written against PufferLib 3.0.0.

`FactorySimPufferEnv` is a native `pufferlib.PufferEnv`: one object holding
`num_envs` agents.

```python
import pufferlib, pufferlib.vector
from fsim.puffer_env import FactorySimPufferEnv

env = pufferlib.vector.make(FactorySimPufferEnv, backend=pufferlib.PufferEnv,
                            env_kwargs=dict(num_envs=512, task="build_line", threads=8))
obs, infos = env.reset(seed=0)
obs, rewards, terminals, truncations, infos = env.step(actions)  # (512, 6) int32
masks = env.action_masks                                         # (512, 201) uint8
fields = env.decode()                                            # named arrays
```

- **Zero-copy:** `VecEnv` writes observations straight into PufferLib's
  `observations` buffer, and masks into `env.action_masks`.
- **Observations:** a native PufferEnv needs a single `Box`, so each row holds
  the raw bytes of the C struct as `uint8`. With `compact=True` (the default)
  that is `fsim_obs8`, 9,100 bytes. With `compact=False` it is `fsim_obs`,
  103,632 bytes. `decode_obs(rows, compact)` recovers the fields. On the GPU,
  `fsim.vec.obs_layout` and `unpack_grid(..., xp=torch)` do the same.
- **Action masks:** PufferLib 3.0's trainer does not consume action masks. Its
  `masks` buffer marks live agents, not legal actions. A policy that masks
  its logits should read `env.action_masks`.
- **Resets** happen internally, in the same step. Each finished episode appears
  in the returned info list as one dict of numbers.
- **Other backends:** `Serial` and `Multiprocessing` work too, because the
  constructor accepts `buf` and `seed`. The emulation route,
  `pufferlib.emulation.GymnasiumPufferEnv(env_creator=FactorySimEnv)`, also
  works, but it is one environment per process slot and much slower.
- **Installing:** PufferLib 3.0.0 ships only an sdist, and its `setup.py`
  raises `Unsupported system: Windows`. It also pins `numpy<2` and
  `gymnasium<=0.29.1`, which conflict with this package, so it is not listed as
  a dependency. On Linux, run `pip install factory-sim[puffer]` and then
  `pip install --no-deps pufferlib`. The modules the adapter uses run fine on
  numpy 2 and Gymnasium 1.x. On Windows the adapter and its tests work only
  with the sdist's source tree on `PYTHONPATH`. `import pufferlib` also
  symlinks `resources` into the working directory.

### Throughput

These are laptop figures (16 threads, `construct_smelting_line`, 8 worker
threads, random masked actions). They count environment steps per second, with
the time spent sampling actions left out.

| | 64 envs | 512 envs |
|---|---:|---:|
| raw `VecEnv` (float obs) | 183k | 80k |
| raw `VecEnv` (compact) | 250k | 347k |
| Gymnasium vector, `copy=True` | 42k | 22k |
| Gymnasium vector, `copy=False` | 176k | 80k |
| PufferLib native (compact) | 241k | 416k |
| PufferLib native (float) | 183k | 79k |

The PufferLib adapter costs about as much as raw `VecEnv`, because it adds no
copy. The Gymnasium vector's overhead comes almost entirely from copying the
float grid (about 100 KB per environment per step) out of the strided C
buffers. With `copy=False` it runs at raw speed.

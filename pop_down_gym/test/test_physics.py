from pop_down_gym.physics import TauEScaling
import equinox as eqx
import jax

def test_taue_input():
    foo = TauEScaling()
    trainable, static = eqx.partition(foo, foo.trainable_params_filter_spec())
    # Expect 18 trainable params. 9 for Hmode and 9 for Lmode.
    assert len(jax.tree_leaves(trainable)) == 18
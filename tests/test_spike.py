from spike.gradient_inversion import attack


def test_underparameterized_gradient_matches_but_input_is_ambiguous():
    _, match, err = attack(reps=1, layers=1, restarts=5, seed=0)
    assert match < 1e-10
    assert err > 0.5


def test_overparameterized_input_is_recovered():
    _, match, err = attack(reps=4, layers=1, restarts=5, seed=0)
    assert match < 1e-8
    assert err < 1e-3

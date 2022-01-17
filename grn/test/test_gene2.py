from lib.sde.gene2 import ParamGene


def test_get():
    assert ParamGene.get("init") == 0


def test_length():
    assert ParamGene.length() == 8


def test_contains():
    assert "noise" in ParamGene
    assert "nothing" not in ParamGene


def test_iter():
    assert [x for x in ParamGene] == ["init", "noise", "b", "m", "expr", "deg", "thr", "theta"]

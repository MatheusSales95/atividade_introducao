from pacotesoma.pacotesoma import soma


def test_soma_com_inteiros():
    assert soma(2, 3) == 5


def test_soma_com_floats():
    assert soma(1.5, 2.5) == 4.0


def test_soma_com_inteiro_e_float():
    assert soma(1, 2.5) == 3.5


def test_soma_com_numeros_negativos():
    assert soma(-4, -6) == -10


def test_soma_com_zero():
    assert soma(0, 0) == 0


def test_soma_com_string_no_primeiro_argumento():
    assert soma("a", 2) == "Only integer or float numbers allowed"


def test_soma_com_string_no_segundo_argumento():
    assert soma(2, "b") == "Only integer or float numbers allowed"


def test_soma_com_lista():
    assert soma([1, 2], 3) == "Only integer or float numbers allowed"

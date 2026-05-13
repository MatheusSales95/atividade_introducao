# Metodo soma do pacotesoma
# Criação da função soma, que recebera dois argumentos e retornará a soma destes números
def soma(a, b):
    """Soma dois números (int ou float) e retorna o resultado.

    Parâmetros
    ----------
    a : int | float
        Primeiro número a ser somado.
    b : int | float
        Segundo número a ser somado.

    Retorno
    -------
    int | float
        A soma de ``a`` e ``b`` quando ambos são numéricos.
    str
        Mensagem de erro caso algum dos argumentos não seja numérico.

    Exemplos
    --------
    >>> soma(4, 9)
    13
    >>> soma(1.5, 2.5)
    4.0
    >>> soma("a", 2)
    'Only integer or float numbers allowed'
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "Only integer or float numbers allowed"
    resultado = a + b
    return resultado

def hello():
    print("Hello World!")


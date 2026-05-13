# pacotesoma

Pacote Python simples, criado como **demonstração** dos passos de empacotamento descritos
no notebook `package_creation.ipynb`. Expõe uma única função, `soma`, que soma dois
números inteiros ou de ponto flutuante.

## Estrutura do projeto

```text
pacotesoma/
├── pacotesoma/
│   ├── __init__.py
│   └── pacotesoma.py
├── tests/
│   └── test_soma.py
├── dist/                  # gerado por `python -m build`
│   ├── pacotesoma-0.1.0-py3-none-any.whl
│   └── pacotesoma-0.1.0.tar.gz
├── README.md
├── LICENSE
├── pyproject.toml
├── setup.cfg
└── setup.py
```

## Instalação

A partir do diretório raiz do projeto:

```bash
pip install .
```

Em modo desenvolvimento (recomendado durante a edição do código):

```bash
pip install -e .
```

Ou diretamente a partir do arquivo `.whl` gerado:

```bash
pip install dist/pacotesoma-0.1.0-py3-none-any.whl
```

## Uso

```python
from pacotesoma import soma

print(soma(4, 9))        # 13
print(soma(1.5, 2.5))    # 4.0
print(soma("a", 2))      # 'Only integer or float numbers allowed'
```

Para inspecionar a documentação da função:

```python
help(soma)
```

## Construção do pacote (gerar `.whl` e `.tar.gz`)

Usando o método moderno via `pyproject.toml`:

```bash
pip install build
python -m build
```

Isso gera, dentro de `dist/`:

- `pacotesoma-0.1.0-py3-none-any.whl` — instalador binário
- `pacotesoma-0.1.0.tar.gz` — distribuição de código-fonte

Alternativa legada (com `setup.py`):

```bash
python setup.py bdist_wheel
python setup.py sdist
```

## Testes

```bash
pip install pytest
python -m pytest
```

## Publicação no PyPI

```bash
pip install twine
python -m twine check dist/*
python -m twine upload dist/*
```

Para publicar primeiro no TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

## Desinstalação

```bash
pip uninstall pacotesoma
```

## Licença

Distribuído sob a licença MIT — veja o arquivo [LICENSE](LICENSE) para detalhes.

# hydra-test <!-- omit in toc -->

Facebook AI Researchが公開しているパラメーター管理ツールhydraを試したコード。ベースとして利用したのは[これ](https://github.com/rasbt/python-machine-learning-book-3rd-edition/tree/master/ch12)。
> Ryeではhydra-coreがインストールできない。venvモジュールを用いて仮想環境を生成しpipでインストールする場合は正常にインストールできる。

## Table of Contents <!-- omit in toc -->

- [Requirement](#requirement)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Author](#author)

## Requirement

- hydra
- numpy
- matplotlib

## Usage

必ずsrcに移動してから実行する（データ読み込みの際のディレクトリ判定の都合による）。

```shell
cd src
./main.py
```

## Repository Structure

``` rawtext
.
├─ conf
│   ├── exp001.yaml
│   ├── exp002.yaml
│   ├── exp003.yaml
│   └── exp004.yaml
├─ data
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
├─ src
│   ├── __init__.py
│   ├── data.py
│   ├── main.py
│   └── neuralnet.py
├── .gitattributes
├── .gitignore
└── README.md
```

## Author

<div align="center">
<img src="https://avatars.githubusercontent.com/u/113882060?v=4" width="100" height="100" alt="avator"><br>
<strong>Thinqat(Thinqat1985731)</strong>
</div>

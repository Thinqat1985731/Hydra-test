<a id="readme-top"></a>

# hydra-test <!-- omit in toc -->

Facebook AI Researchが公開しているパラメーター管理ツールhydraを試しに使ってみたコード。ベースとして利用したのは[これ](https://github.com/rasbt/python-machine-learning-book-3rd-edition/tree/master/ch12)。

<details><summary>Table of Contents</summary>

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

</details>

## Getting Started

### Prerequisites

仮想環境について、Ryeではhydra-coreが正常にインストールできないため、venvモジュールの利用を想定している。

```shell
python -m venv .venv
```

### Installation

1. このリポジトリをクローンする。

   ```shell
   git clone https://github.com/Thinqat1985731/hydra-test.git
   ```

2. setup.sh又はsetup.ps1を実行し、pipを経由して必要なpythonライブラリをダウンロードする。

   ```shell
   ./setup.sh
   ./setup.ps1
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

必ずsrcに移動してから実行する（データ読み込みの際のディレクトリ判定の都合による）。

```shell
cd src
./main.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

<div align="center">
   <img src="https://avatars.githubusercontent.com/u/113882060?v=4" width="100" height="100" alt="avator"><br>
   <strong>Thinqat(Thinqat1985731)</strong><br><br>

  <a href="https://github.com/Thinqat1985731" target="_blank">
  <picture>
    <source
      srcset="https://img.shields.io/badge/GitHub-444444.svg?style=for-the-badge&logo=github"
      media="(prefers-color-scheme: dark)"
    />
    <source
      srcset="https://img.shields.io/badge/GitHub-000000.svg?style=for-the-badge&logo=github"
      media="(prefers-color-scheme: light)"
    />
    <img src="https://img.shields.io/badge/-Github-444444.svg?style=for-the-badge&logo=github" alt="Github"/>
  </picture>
  </a>
  <a href="https://huggingface.co/Thinqat1985731" target="_blank">
    <picture>
      <source
        srcset="https://img.shields.io/badge/Hugging_Face-444444.svg?style=for-the-badge&logo=huggingface&logoColor=white"
        media="(prefers-color-scheme: dark)"
      />
      <source
        srcset="https://img.shields.io/badge/Hugging_Face-000000.svg?style=for-the-badge&logo=huggingface&logoColor=white"
        media="(prefers-color-scheme: light)"
      />
      <img src="https://img.shields.io/badge/Hugging_Face-444444.svg?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face"/>
    </picture>
  </a>
  <a href="https://thinqat.vercel.app/" target="_blank">
    <picture>
    <source
      srcset="https://img.shields.io/badge/Myblog-444444.svg?style=for-the-badge&logo=nextdotjs"
      media="(prefers-color-scheme: dark)"
    />
    <source
      srcset="https://img.shields.io/badge/Myblog-000000.svg?style=for-the-badge&logo=nextdotjs"
      media="(prefers-color-scheme: light)"
    />
    <img src="https://img.shields.io/badge/Myblog-444444.svg?style=for-the-badge&logo=nextdotjs" alt="Myblog"/>
  </picture>
  </a>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

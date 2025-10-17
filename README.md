# RAG Project

This repository contains an implementation of Retrieval-Augmented Generation (RAG).

## Features

- Document retrieval using vector embeddings
- Integration with LLMs for answer generation
- Modular design for easy extension

## Installation

```bash
git clone https://github.com/mkhuda/mkhuda-blog-llm-rag.git
cd mkhuda-blog-llm-rag
pip install -r requirements.txt
```

## Usage

1. Prepare your documents in the `data/` folder.
2. Run the retrieval and generation pipeline:

    ```bash
    python rag_build.py
    python main.py --query "Your question here"
    ```

## Configuration

Edit `config.yaml` to adjust model parameters and data paths.

## Contributing

Pull requests are welcome. For major changes, open an issue first.

## License

This project is licensed under the MIT License.
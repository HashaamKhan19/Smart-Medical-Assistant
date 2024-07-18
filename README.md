## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Poetry (Follow this [Poetry installation tutorial](https://python-poetry.org/docs/#installation) to install Poetry on your system)

### Installation

1. Clone the repository

2. Install dependencies using Poetry:

   ```bash
   poetry install --no-root
   ```

3. Set up your environment variables:

   create `.env` file and update the variable (OPENAI_API_KEY) inside with your own value.

4. Activate the Poetry shell to run the examples:

   ```bash
   poetry shell
   ```

5. Run the code examples:

   ```bash
    python assistant/assistant.py
   ```
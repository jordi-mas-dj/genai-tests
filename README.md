# genai-tests

GenAI Retrieval API testing tool for performing searches and analyzing content.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone or download this repository

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables by creating a `.env` file in the project root with the following credentials:
```
GEN_AI_RETRIEVAL_CLIENT_ID=your_client_id
GEN_AI_RETRIEVAL_SERVICE_ACCOUNT=your_service_account
GEN_AI_RETRIEVAL_PASSWORD=your_password
```

If you need credentials, please ask Jordi Mas over slack.

## Usage

Run the tool using Python:

```bash
python genai.py [arguments]
```

The tool supports various command-line arguments for customizing your search. Run with `--help` to see all available options:

```bash
python genai.py --help
```
# AI-Driven Python Project Generator - The God Fathers AI software dev team.

Generates and refines Python projects based on prompts, using OpenAI-powered insights.

## Features
- **Idea Extraction:** Extracts core logic from user inputs.
- **Code Generation:** Creates base code from extracted logic.
- **Refinement:** Refines code via iterative feedback.
- **GitHub Upload:** Uploads final project to GitHub.

## Usage
1. **initiate_project(idea)**: Kickstarts project generation.
2. **generate_base_code()**: Constructs initial code structure.
3. **refine_code()**: Iteratively refines code.
4. **upload_to_github(repo_name)**: Uploads project to GitHub.

## Example
```python
idea = "Create a web scraper"
project = initiate_project(idea)
project.generate_base_code()
project.refine_code()
project.upload_to_github("Web-Scraper")

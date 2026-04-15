# TODOs

- [ ] also per attribute -> not tags
- [ ] fix tags for attributes per case
- [ ] combine evaluators per case with and/ or/ .. logic
- [ ] gif of usage for readme

- [ ] Combine Testcases with and/or/ not/ nor
- [ ] Repeat Runs (either via new ids (with run_number) or merge testsetresults)
- [ ] Repeat LLMJudge Evals, with its own threshold that defaults to 100%
- [ ] Add description to Test-Cases. Could explain idea behind, by whom they were created,... Author could also be a nice example for an additional field in csv.
- [ ] better examples: things that used to stump up llms like counting r`s in strawberry.
- [ ] better examples 2: a banks customer service chatbot tested to provide official financial advise 
- [ ] make this repo mcp-connectable/compatible
- [ ] think about chat-history -> just another "Input-type"?
- [ ] In any case, have an example guide for custom input type
- [ ] More broadly: multi-turn? e.g. two agents conversing? but for
- [ ] explicit decoupling of running tasks and evaluating. There should probably be an intermediary object that holds all the information - or if mlflow is just an interface to the mlflow experiment. 
- [ ] Creating integration tests with mlflow docker test-container to ensure Span-based Evaluation works as expected (with fresh backend-db everytime).
- [ ] Add more content to the documentation, especially in the "How-to Guides" section.
  - [x] Add guides on how to create custom evaluators and custom type evaluators.
  - [ ] Add tutorial
  - [ ] Link to MCP template for full example how to integrate into a project.
- [ ] some automated documentation generation:
  - [ ] Versioning of documentation with mike: https://github.com/jimporter/mike also see https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/#usage
- [ ] Use better template for pyhton project with alternative for mkdocs.
  - [ ] Link to MCP template for full example how to integrate into a project.


realistic Tutorial at some point (pretty involved agent creation)

continue with the tutorial: create a pydantic-ai agent with an async openai client and with base_url and api-key from env.
The agent should have a very simple system prompt to be a customer service chat bot for a (fictional) bank "Ragpill Bank" and has tools (just printing out their commands) for get_account_balance (user is injected from auth- fastapi style)

Done:
- [x] Quotes with ... in the end -> truncate?!
- [x] maybe more whitespace/ bytecode fixes
- [x] metrics for tags only for True -> rename
- [x] Creating integration tests with mlflow docker test-container to ensure Span-based Evaluation works as expected (with fresh backend-db everytime).
- [ ] Add more content to the documentation, especially in the "How-to Guides" section.
  - [x] Add guides on how to create custom evaluators and custom type evaluators.
  - [x] Add tutorial
- [x] some automated documentation generation:
  - [x] Versioning of documentation with mike: https://github.com/jimporter/mike also see https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/#usage
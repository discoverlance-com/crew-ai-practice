# Crew AI Practice

This repository has a couple of crew ai projects and practice projects.

## Background

The following projects are from the [Deep Learning - Multi AI Agent Systems With Crew AI](https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai) course.

I am mostly using GEMINI for the AI model but you can of course reference the [Crew AI documentation on LLMS](https://docs.crewai.com/en/concepts/llms) to choose and configure your preferred LLM or AI model.

Some of the projects require embedded models. For my use case, I use ollama locally. If you are using open ai models, you can remove or delete the `embedding_model` or `embedder` configs as it will just use an embedding model from open ai.

## Course Projects

These are the projects from the course:

- [Tailor Job Applications](./tailor_job_applications/)
- [Multi Agent Customer Support](./multi_agent_customer_support/)
- [Multi Agent Financial Analysis](./multi_agent_financial_analysis/)
- [Article Researcher](./article_researcher/)
- [Customer Outreach campaign](./tools_for_customer_outreach_campaign/)
- [Automate Event Planning](./automate_event_planning/)
- [Estimate Project Planning](./estimate_project_planning/)

Any other projects are tests and practice I am working on with Crew AI.

### Getting Started With Course Projects

- Change directory into the course project
- Install crewai if you have not already installed it by following the [crewai installation guide](https://docs.crewai.com/en/installation) to install uv and crewai.
- Run the command `crewai install` to install the packages.
- Copy the `.env.example` file to `.env`.
- Add your environment variable values to the `.env` file.
- Make sure to replace the model and env variable values with your model choice.
- Make sure to replace the embeddings config with your embedding model details as mentioned above.
- Run the crew with the command `crewai run`.

> For crews that use Serper API, you can visit their [website](https://serper.dev/), signup if you already don't have an account and retrieve an API key from the dashboard. You get a generous free number of queries that should be enough to run the crews in the course projects.
> For gemini, you can get an api key by visiting [Google AI Studio](https://aistudio.google.com/u/0/apikey) with your personal email account and creating a new API Key if you don't already have one.

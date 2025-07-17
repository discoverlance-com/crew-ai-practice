from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import (
  FileReadTool,
  ScrapeWebsiteTool,
  MDXSearchTool,
  SerperDevTool
)

from embedchain.embedder.ollama import OllamaEmbedder,OllamaEmbedderConfig

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path='./fake_resume.md')
semantic_search_resume = MDXSearchTool(
    mdx='./fake_resume.md',
    config={
        "embedding_model": {
            "provider": "ollama", 
            "config": { 
                "model": "mxbai-embed-large"
            }
        },
    }
)

@CrewBase
class TailorJobApplications():
    """TailorJobApplications crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher_agent'],  # type: ignore[index]
            tools=[scrape_tool, search_tool],
            verbose=True
        )

    @agent
    def profiler_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['profiler_agent'],  # type: ignore[index]
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            verbose=True,
        )

    @agent
    def resume_strategist_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['resume_strategist_agent'],  # type: ignore[index]
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            verbose=True,
        )

    @agent
    def interview_preparer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['interview_preparer_agent'],  # type: ignore[index]
            tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
            verbose=True,
        )


    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],  # type: ignore[index]
        )

    @task
    def profile_task(self) -> Task:
        return Task(
            config=self.tasks_config['profile_task'],  # type: ignore[index]
        )

    @task
    def resume_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config['resume_strategy_task'],  # type: ignore[index]
        )

    @task
    def interview_preparation_task(self) -> Task:
        return Task(
            config=self.tasks_config['interview_preparation_task'],  # type: ignore[index]
        )


    @crew
    def crew(self) -> Crew:
        """Creates the TailorJobApplications crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # I am using ollama locally but you can change this
            # either remove it if you are using open ai model or 
            # add respective embed model details
            embedder={
                "provider": "ollama", 
                "config": { 
                    "model": "mxbai-embed-large"
                }
            }

            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

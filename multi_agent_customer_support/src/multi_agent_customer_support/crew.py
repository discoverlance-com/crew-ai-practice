from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os
from crewai_tools import (
   ScrapeWebsiteTool,
)
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://discoverlance.com/blog/articles/getting-started-with-laravel-service-container-providers-facades"
)

@CrewBase
class MultiAgentCustomerSupport():
    """MultiAgentCustomerSupport crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_agent'], # type: ignore[index]
            verbose=True
        )

    @agent
    def support_quality_assurance_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_quality_assurance_agent'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def inquiry_resolution_task(self) -> Task:
        return Task(
            config=self.tasks_config['inquiry_resolution_task'], # type: ignore[index]
            tools=[docs_scrape_tool],
        )

    @task
    def quality_assurance_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_review_task'], # type: ignore[index]
            output_file='outputs/review.md',
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MultiAgentCustomerSupport crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            embedder={
                "provider": "ollama", # I am using ollama locally but you can change this
                # either leave it if you are using open ai or add respective embed odel details
                "config": { 
                    "model": "mxbai-embed-large"
                }
            }
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

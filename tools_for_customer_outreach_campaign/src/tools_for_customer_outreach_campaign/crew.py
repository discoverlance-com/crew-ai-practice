from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import DirectoryReadTool, \
                         FileReadTool, \
                         SerperDevTool
from .tools.sentiment_analysis_tool import SentimentAnalysisTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

directory_read_tool = DirectoryReadTool(directory='./instructions')
file_read_tool = FileReadTool()
search_tool = SerperDevTool()
sentiment_analysis_tool = SentimentAnalysisTool()

@CrewBase
class ToolsForCustomerOutreachCampaign():
    """ToolsForCustomerOutreachCampaign crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def sales_rep_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['sales_rep_agent'], # type: ignore[index]
            verbose=True
        )

    @agent
    def lead_sales_rep_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['lead_sales_rep_agent'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def lead_profiling_task(self) -> Task:
        return Task(
            config=self.tasks_config['lead_profiling_task'], # type: ignore[index]
            tools=[directory_read_tool, file_read_tool, search_tool]
        )

    @task
    def personalized_outreach_task(self) -> Task:
        return Task(
            config=self.tasks_config['personalized_outreach_task'], # type: ignore[index]
            output_file='outputs/outreach .md',
            tools=[sentiment_analysis_tool, search_tool]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ToolsForCustomerOutreachCampaign crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            # I am using ollama locally but you can change this
            # either remove it if you are using open ai api 
            # or add respective embed model details
            embedder={
                "provider": "ollama", 
                "config": { 
                    "model": "mxbai-embed-large"
                }
            }
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

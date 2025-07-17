from crewai import Agent, Crew, Process, Task,LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
import os

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

model = os.environ.get('MODEL')

if not model:
    raise

manager_llm = LLM(model=model,temperature=0.7)

@CrewBase
class MultiAgentFinancialAnalysis():
    """MultiAgentFinancialAnalysis crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def data_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst_agent'], # type: ignore[index]
            verbose=True,
            tools=[scrape_tool,search_tool]
        )

    @agent
    def trading_strategy_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['trading_strategy_agent'], # type: ignore[index]
            verbose=True,
            tools=[scrape_tool,search_tool]
        )
    
    @agent
    def execution_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['execution_agent'], # type: ignore[index]
            verbose=True,
            tools=[scrape_tool,search_tool]
        )
    
    @agent
    def risk_management_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['risk_management_agent'], # type: ignore[index]
            verbose=True,
            tools=[scrape_tool,search_tool]
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def data_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_analysis_task'], # type: ignore[index]
        )
    
    @task
    def strategy_development_task(self) -> Task:
        return Task(
            config=self.tasks_config['strategy_development_task'], # type: ignore[index]
            output_file='outputs/strategy.md'
        )

    @task
    def execution_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['execution_planning_task'], # type: ignore[index]
            output_file='outputs/execution.md'
        )
    
    @task
    def risk_assessment_task(self) -> Task:
        return Task(
            config=self.tasks_config['risk_assessment_task'], # type: ignore[index]
            output_file='outputs/assessment.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MultiAgentFinancialAnalysis crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.hierarchical,
            verbose=True,
            manager_llm=manager_llm,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()


# Define a Pydantic model for venue details 
# (demonstrating Output as Pydantic)
class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: int
    booking_status: str

@CrewBase
class AutomateEventPlanning():
    """AutomateEventPlanning crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def venue_cordinator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['venue_cordinator_agent'], # type: ignore[index]
            verbose=True,
            tools=[search_tool, scrape_tool]
        )

    @agent
    def logistics_manager_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['logistics_manager_agent'], # type: ignore[index]
             tools=[search_tool, scrape_tool],
            verbose=True
        )

    @agent
    def marketing_communications_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['marketing_communications_agent'], # type: ignore[index]
             tools=[search_tool, scrape_tool],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def venue_task(self) -> Task:
        return Task(
            config=self.tasks_config['venue_task'], # type: ignore[index]
            output_json=VenueDetails,
        )

    @task
    def logistics_task(self) -> Task:
        return Task(
            config=self.tasks_config['logistics_task'], # type: ignore[index]
        )
    
    @task
    def marketing_task(self) -> Task:
        return Task(
            config=self.tasks_config['marketing_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AutomateEventPlanning crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

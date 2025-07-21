from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel, Field
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class TaskEstimate(BaseModel):
    task_name: str = Field(..., description="Name of the task")
    estimated_time_hours: float = Field(..., description="Estimated time to complete the task in hours")
    required_resources: List[str] = Field(..., description="List of resources required to complete the task")

class Milestone(BaseModel):
    milestone_name: str = Field(..., description="Name of the milestone")
    tasks: List[str] = Field(..., description="List of task IDs associated with this milestone")

class ProjectPlan(BaseModel):
    tasks: List[TaskEstimate] = Field(..., description="List of tasks with their estimates")
    milestones: List[Milestone] = Field(..., description="List of project milestones")

@CrewBase
class EstimateProjectPlanning():
    """EstimateProjectPlanning crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def project_planning_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['project_planning_agent'], # type: ignore[index]
            verbose=True
        )

    @agent
    def estimation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['estimation_agent'], # type: ignore[index]
            verbose=True
        )
    @agent
    def resource_allocation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['resource_allocation_agent'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def breakdown_task(self) -> Task:
        return Task(
            config=self.tasks_config['breakdown_task'], # type: ignore[index]
            output_file='outputs/breakdown'
        )

    @task
    def time_resource_estimation_task(self) -> Task:
        return Task(
            config=self.tasks_config['time_resource_estimation_task'], # type: ignore[index]
            output_file='outputs/resource_estimation'
        )
    
    @task
    def resource_allocation_task(self) -> Task:
        return Task(
            config=self.tasks_config['resource_allocation_task'], # type: ignore[index]
            output_file='outputs/resource_allocation',
            output_pydantic=ProjectPlan
        )

    @crew
    def crew(self) -> Crew:
        """Creates the EstimateProjectPlanning crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

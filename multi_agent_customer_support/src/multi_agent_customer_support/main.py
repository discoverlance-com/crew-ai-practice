#!/usr/bin/env python
import sys
import warnings

from multi_agent_customer_support.crew import MultiAgentCustomerSupport

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        "customer": "Acme.inc",
        "person": "Sarah Doe",
        "inquiry": "I need help with creating a service provider in the laravel software development framework, specifically how can I ensure that the service provider can be deferred so it's not initially loaded by laravel? Can you provide guidance?"
    }
    
    try:
        MultiAgentCustomerSupport().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "customer": "Acme.inc",
        "person": "Sarah Doe",
        "inquiry": "I need help with creating a service provider in laravel, specifically how can I ensure that the service provider can be deferred so it's not initially loaded by laravel? Can you provide guidance?"
    }
    try:
        MultiAgentCustomerSupport().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MultiAgentCustomerSupport().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "customer": "Acme.inc",
        "person": "Sarah Doe",
        "inquiry": "I need help with creating a service provider in laravel, specifically how can I ensure that the service provider can be deferred so it's not initially loaded by laravel? Can you provide guidance?"
    }
    
    try:
        MultiAgentCustomerSupport().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

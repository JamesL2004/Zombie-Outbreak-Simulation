import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mesa.datacollection import DataCollector
from mesa.visualization import SolaraViz, make_plot_component, make_space_component

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.agents]
    x = sorted(agent_wealths)
    T = model.total_agents
    B = sum(xi * (T - i) for i, xi in enumerate(x)) / (T * sum(x))
    return 1 + (1 / T) - 2 * B

class OutbreakAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        # Create the agent's variable and set the initial values.
        self.isZombie = False
        self.shotsLeft = 15
        self.dead = False

    def step(self):
        self.move()
        if self.isZombie == True:
            self.infect()
            

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
    
    def infect(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        humans = []
        for cell in cellmates:
            if cell.isZombie == False:
                humans.add(cell)
        if len(humans) > 0:
             other = self.random.choice(humans)
             other.isZombie = True


class OutbreakModel(mesa.Model):
    """A model with some number of agents."""
    def __init__(self, totalAgents=100, width=20, height=20):
        super().__init__()
        self.total_agents = 100
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.datacollector = mesa.DataCollector(
            model_reporters={"Gini": compute_gini}, agent_reporters={"Wealth": "wealth"}
        )
        # Create agents
        for i in range(self.total_agents):
            agent = OutbreakAgent(self)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.running = True
        #self.datacollector.collect(self)

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")


model_params = {
    "totalAgents": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
    "width": {
        "type": "SliderInt",
        "value": 20,
        "label": "Width:",
        "min": 10,
        "max": 100,
        "step": 10,
    },
    "height": {
        "type": "SliderInt",
        "value": 20,
        "label": "Height:",
        "min": 10,
        "max": 100,
        "step": 10,
    },
}

#modify this function to change output on grid
def agent_portrayal(agent):
    size = 10
    color = "tab:red"

    if agent.wealth > 3:
        size = 80
        color = "tab:blue"
    elif agent.wealth > 2:
        size = 50
        color = "tab:green"
    elif agent.wealth > 1:
        size = 20
        color = "tab:orange"
    return {"size": size, "color": color}

money_model = OutbreakModel(10, 10, 10)

SpaceGraph = make_space_component(agent_portrayal)
GiniPlot=make_plot_component("Gini")

page = SolaraViz(
    money_model,
    components=[SpaceGraph, GiniPlot],
    model_params=model_params,
    name="Money Model"
)
# This is required to render the visualization in the Jupyter notebook
page
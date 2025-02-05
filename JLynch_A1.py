import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import random as rn
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
            if rn.random() < 0.5:
                self.dropAmmo()
        elif self.isZombie == False:
            self.shootZombie()

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
                humans.append(cell)
        if len(humans) > 0:
             other = self.random.choice(humans)
             other.isZombie = True
    
    def dropAmmo(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        humans = []
        for cell in cellmates:
            if cell.isZombie == False:
                humans.append(cell)
        if len(humans) > 0:
            other = self.random.choice(humans)
            self.shotsLeft -= 3
            other.shotsLeft += 3

    def shootZombie(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        zombies = []
        for cell in cellmates:
            if cell.isZombie == True:
                zombies.append(cell)
        if rn.random() < 0.5:
            if self.shotsLeft > 0:
                other = self.random.choice(zombies)
                other.dead = True
                self.shotsLeft -= 1


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
    size = 20
    color = "tab:red"

    if agent.dead == True:
        size = 40
        color = "tab:blue"
    elif agent.isZombie == True:
        size = 60
        color = "tab:green"
    elif agent.isZombie == False:
        size = 20
        color ="tab:red"
    return {"size": size, "color": color}

outbreak_model = OutbreakModel(10, 10, 10)

SpaceGraph = make_space_component(agent_portrayal)
GiniPlot=make_plot_component("Gini")

page = SolaraViz(
    outbreak_model,
    components=[SpaceGraph, GiniPlot],
    model_params=model_params,
    name="Zmobie Outbreak Model"
)
# This is required to render the visualization in the Jupyter notebook
page
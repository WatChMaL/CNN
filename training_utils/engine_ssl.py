"""
engine_ssl.py

Derived engine class for training a deep generative model
for semi-supervised learning
"""

# WatChMaL imports
from training_utils.engine import Engine

class EngineSSL(Engine):
    
    def __init__(self, model, config):
        super(model, config).__init__()
        
        # Setup the optimizer with the correct parameters
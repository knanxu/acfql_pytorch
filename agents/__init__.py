"""
Agents Collection for Offline-to-Online RL

Available agents:
- fbc: Flow-based Behavior Cloning (standard flow matching)
- mfbc: Mean Flow BC (JVP-based flow matching with t_begin, t_end)
- imfbc: Improved Mean Flow BC (enhanced JVP formulation)
- fql: Flow Q-Learning (full RL with critic)

All agents use dict-based config (similar to JAX version).
"""

from agents.fbc import BCAgent, get_config as fbc_get_config
from agents.mfbc import MFBCAgent, get_config as mfbc_get_config
from agents.imfbc import IMFBCAgent, get_config as imfbc_get_config
from agents.fql import FQLAgent, get_config as fql_get_config

# Agent registry: name -> (AgentClass, get_config_fn)
# All agents use dict-based config (no ConfigClass)
agents = dict(
    fbc=(BCAgent, fbc_get_config),
    mfbc=(MFBCAgent, mfbc_get_config),
    imfbc=(IMFBCAgent, imfbc_get_config),
    fql=(FQLAgent, fql_get_config),
)


def get_agent(agent_name: str):
    """
    Get agent class by name.
    
    Returns:
        (AgentClass, None) - None is for backward compatibility
    """
    if agent_name not in agents:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(agents.keys())}")
    return agents[agent_name][0], None  # Return None as ConfigClass placeholder


def get_agent_config(agent_name: str):
    """Get default config (ml_collections.ConfigDict) for agent by name."""
    if agent_name not in agents:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(agents.keys())}")
    return agents[agent_name][1]()


def list_agents():
    """List all available agents."""
    return list(agents.keys())

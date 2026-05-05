import os
from vram_manager import vram
from config import config

class Lane:
    """
    Represents a single 'lane' or functional role in the Urithiru tower.
    Maps a persona to a functional role and handles the execution.
    """
    def __init__(self, name, persona, role_description):
        self.name = name          # e.g., "critic"
        self.persona = persona    # e.g., "Jasnah"
        self.role = role_description
        self.client = None

    def load(self):
        """Loads the agent into VRAM via the vram_manager."""
        self.client = vram.load_agent(self.name)
        return self.client

    def __repr__(self):
        return f"<Lane {self.name} ({self.persona})>"

class LaneManager:
    """
    Orchestrates the lanes, ensuring correct VRAM management and sequential/parallel execution.
    """
    def __init__(self):
        self.lanes = {
            "developer": Lane("developer", "Navani", "Strategy profiling and recipe extraction"),
            "critic": Lane("critic", "Jasnah", "Pine Script logic audit and auto-fix"),
            "transpiler": Lane("transpiler", "Renarin", "Unknown indicator handling"),
            "quant": Lane("quant", "Pattern", "Optuna backtest optimization"),
            "catfish": Lane("catfish", "Shallan", "Devil's Advocate and dissent"),
            "operator": Lane("operator", "Adolin", "Production readiness audit"),
            "strategist": Lane("strategist", "Dalinar", "Final synthesis and verdict")
        }

    def get_lane(self, name):
        return self.lanes.get(name)

    def run_lane(self, name, func, *args, **kwargs):
        """
        Loads the necessary lane and executes the provided function.
        Handles VRAM cleanup if necessary (though vram_manager handles most of it).
        """
        lane = self.get_lane(name)
        if not lane:
            raise ValueError(f"Unknown lane: {name}")
        
        print(f"  -> Activating Lane: {lane.persona} ({lane.name})...")
        client = lane.load()
        
        # Inject client into kwargs if the function expects it
        kwargs['client'] = client
        return func(*args, **kwargs)

    def unload_all(self):
        vram.unload_all()

# Global orchestrator instance
orchestrator = LaneManager()

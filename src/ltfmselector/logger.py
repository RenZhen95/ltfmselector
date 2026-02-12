import numpy as np

pmSelection = lambda x: 1 if len(x) > 1 else 0

class Logger:
    def __init__(self, env, max_steps=500):
        self.state_dim = env.X.shape[1]*2 + pmSelection(env.pModels)
        self.max_steps = max_steps
        
        # Pre-allocate the "scratchpad" for the current episode
        self.s_buffer = np.zeros((self.max_steps, self.state_dim), dtype=np.float32)
        self.a_buffer = np.zeros((self.max_steps,), dtype=np.int32)
        self.ptr = 0
        
        # Final storage lists
        self.all_states = []
        self.all_actions = []

    def log_step(self, obs, action):
        if self.ptr < self.max_steps:
            self.s_buffer[self.ptr] = obs
            self.a_buffer[self.ptr] = action
            self.ptr += 1

    def log_episode(self):
        """Internal helper to copy the current buffer into the main list."""
        if self.ptr > 0:
            self.all_states.append(self.s_buffer[:self.ptr].copy())
            self.all_actions.append(self.a_buffer[:self.ptr].copy())
            self.ptr = 0

    def save_data(self, filename):
        """Saves all completed and currently-running episodes to disk."""
        # Capture the current episode if it's mid-run
        self.log_episode()
        np.savez_compressed(
            filename, 
            states=np.array(self.all_states, dtype=object),
            actions=np.array(self.all_actions, dtype=object)
        )
        print(f"Saved {len(self.all_states)} episodes (states + actions) to {filename}")

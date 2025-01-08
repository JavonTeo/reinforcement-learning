from Transitions.Transition import Transition

class PrioritizedTransition(Transition):
    def __init__(self, obs, action, reward, next_obs, terminated, priority=None):
        super().__init__(obs, action, reward, next_obs, terminated)
        self.priority = priority
    
    def __init__(self, transition:Transition, priority):
        super().__init__(transition.obs, transition.action, transition.reward, transition.next_obs, transition.terminated)
        self.priority = priority
    
    def __lt__(self, other):
        return self.priority < other.priority
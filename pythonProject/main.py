import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


# Custom environment class for Robot Sumo
class RobotSumoEnv(gym.Env):
    def __init__(self):
        super(RobotSumoEnv, self).__init__()

        # Define the action and observation space
        self.action_space = spaces.Discrete(3)  # Move forward, turn left, turn right
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # Initialize positions of robot and opponent
        self.robot_position = np.array([0.5, 0.5])
        self.opponent_position = np.array([0.7, 0.7])
        self.ring_radius = 1.0

        # Initialize Pygame for rendering
        pygame.init()
        self.window_size = 600
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def reset(self):
        self.robot_position = np.array([0.5, 0.5])
        self.opponent_position = np.array([0.7, 0.7])
        observation = np.concatenate([self.robot_position, self.opponent_position])
        return observation, {}

    def step(self, action):
        if action == 0:  # Move forward
            self.robot_position[1] += 0.1
        elif action == 1:  # Turn left
            self.robot_position[0] -= 0.1
        elif action == 2:  # Turn right
            self.robot_position[0] += 0.1

        distance_to_opponent = np.linalg.norm(self.robot_position - self.opponent_position)
        robot_out = np.linalg.norm(self.robot_position) > self.ring_radius
        opponent_out = np.linalg.norm(self.opponent_position) > self.ring_radius

        if opponent_out:
            reward = 1
            done = True
        elif robot_out:
            reward = -1
            done = True
        else:
            reward = -distance_to_opponent
            done = False

        observation = np.concatenate([self.robot_position, self.opponent_position])
        return observation, reward, done, {}, {}

    def render(self, mode='human'):
        # Convert the robot and opponent positions to Pygame coordinates
        def to_pygame_coords(position):
            return int(self.window_size * position[0]), int(self.window_size * position[1])

        # Fill the screen with white
        self.screen.fill((255, 255, 255))

        # Draw the sumo ring
        pygame.draw.circle(self.screen, (0, 0, 0), (self.window_size // 2, self.window_size // 2),
                           int(self.ring_radius * self.window_size / 2), 2)

        # Draw the robot (blue) and opponent (red)
        robot_coords = to_pygame_coords(self.robot_position)
        opponent_coords = to_pygame_coords(self.opponent_position)
        pygame.draw.circle(self.screen, (0, 0, 255), robot_coords, 20)
        pygame.draw.circle(self.screen, (255, 0, 0), opponent_coords, 20)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        self.clock.tick(30)

    def close(self):
        pygame.quit()


# Main function to test the environment
if __name__ == "__main__":
    env = RobotSumoEnv()

    # Loop for 10 matches
    for match in range(10):
        print(f"Starting match {match + 1}")
        cumulative_score = 0  # Track the cumulative score for each match
        max_score = 1  # The maximum possible score for each match (if robot wins)

        observation, info = env.reset()

        # Each match will run for 50 steps
        for step in range(50):
            action = env.action_space.sample()  # Random action (for now)
            observation, reward, done, _, info = env.step(action)

            cumulative_score += reward  # Add the reward from each step to the cumulative score

            # Handle Pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    exit()

            # Render the current state of the environment
            env.render()

            if done:
                print(f"Match {match + 1} over at step {step + 1}. Resetting environment.")
                break  # Break the step loop when the match is over

        print(f"Match {match + 1} score: {cumulative_score}/{max_score}")

    env.close()
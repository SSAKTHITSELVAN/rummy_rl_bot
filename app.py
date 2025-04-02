# Rummy AI with Reinforcement Learning
# Complete implementation including game state recognition, RL model, and mobile device control

import os
import time
import random
import numpy as np
import cv2
import pytesseract
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import subprocess
from PIL import Image
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURATION
# ======================================================

# Device settings
DEVICE_ID = "emulator-5554"  # Change to your device ID
SCREENSHOT_PATH = "screen.png"
MODEL_SAVE_PATH = "rummy_model.pth"

# Game settings
CARD_VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
CARD_SUITS = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
NUM_CARDS_IN_HAND = 13

# RL Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99  # discount factor
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 10
LEARNING_RATE = 1e-4
MEMORY_SIZE = 10000

# Define screen regions (these need to be calibrated for your specific game)
HAND_REGION = (100, 1200, 980, 1700)  # (x1, y1, x2, y2)
DRAW_PILE_REGION = (500, 800, 600, 900)
DISCARD_PILE_REGION = (700, 800, 800, 900)

# ======================================================
# COMPUTER VISION COMPONENT
# ======================================================

class RummyVision:
    def __init__(self):
        # Initialize card templates for recognition
        self.card_templates = self._load_card_templates()
        
    def _load_card_templates(self):
        # In a real implementation, you would load actual template images
        # This is a placeholder - you would need actual card images from your game
        templates = {}
        # For example: templates[('H', 'A')] = cv2.imread('ace_of_hearts.png')
        print("Card templates would be loaded here")
        return templates
    
    def capture_screen(self):
        """Capture screenshot from connected device"""
        try:
            subprocess.run(f'adb -s {DEVICE_ID} shell screencap -p /sdcard/screen.png', shell=True, check=True)
            subprocess.run(f'adb -s {DEVICE_ID} pull /sdcard/screen.png {SCREENSHOT_PATH}', shell=True, check=True)
            return cv2.imread(SCREENSHOT_PATH)
        except subprocess.CalledProcessError as e:
            print(f"Failed to capture screen: {e}")
            return None
    
    def recognize_cards(self, image):
        """Recognize cards from the game screen"""
        if image is None:
            return [], [], []
            
        # Extract regions
        hand_region = image[HAND_REGION[1]:HAND_REGION[3], HAND_REGION[0]:HAND_REGION[2]]
        discard_pile = image[DISCARD_PILE_REGION[1]:DISCARD_PILE_REGION[3], DISCARD_PILE_REGION[0]:DISCARD_PILE_REGION[2]]
        
        # Detect individual cards in hand (placeholder implementation)
        hand_cards = self._detect_cards_in_region(hand_region)
        
        # Detect top card in discard pile
        discard_top = self._detect_cards_in_region(discard_pile)
        discard_top = discard_top[0] if discard_top else None
        
        # For game state analysis
        draw_pile_available = self._check_draw_pile_available(image)
        
        return hand_cards, discard_top, draw_pile_available
    
    def _detect_cards_in_region(self, region):
        """Detect and identify cards in a region using template matching and OCR"""
        # This is a simplified version - real implementation would:
        # 1. Use contour detection to find card boundaries
        # 2. For each card, identify suit and value with template matching or OCR
        
        # Placeholder implementation that returns random cards for testing
        num_cards = random.randint(10, 13)  # Simulating 10-13 cards in hand
        cards = []
        
        for _ in range(num_cards):
            suit = random.choice(CARD_SUITS)
            value = random.choice(CARD_VALUES)
            cards.append((suit, value))
            
        return cards
    
    def _check_draw_pile_available(self, image):
        """Check if draw pile has cards available"""
        # In a real implementation, analyze the draw pile region
        # Return True if cards are available, False otherwise
        return True

# ======================================================
# RUMMY GAME LOGIC AND ENVIRONMENT
# ======================================================

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        
    def __repr__(self):
        return f"{self.value}{self.suit}"
        
    def get_rank(self):
        """Convert card value to numerical rank (for sorting and set detection)"""
        if self.value == 'A':
            return 1
        elif self.value in ['J', 'Q', 'K']:
            return {'J': 11, 'Q': 12, 'K': 13}[self.value]
        else:
            return int(self.value)

class RummyGameState:
    def __init__(self):
        self.hand = []  # Cards in player's hand
        self.discard_top = None  # Top card of discard pile
        self.draw_pile_available = True
        
    def update_state(self, hand_cards, discard_top, draw_pile_available):
        """Update game state with new observations"""
        self.hand = [Card(suit, value) for suit, value in hand_cards]
        self.discard_top = Card(discard_top[0], discard_top[1]) if discard_top else None
        self.draw_pile_available = draw_pile_available
        
    def get_valid_actions(self):
        """Return list of valid actions in current state"""
        actions = []
        
        # Always valid to draw from stock (if available)
        if self.draw_pile_available:
            actions.append(('draw_stock', None))
            
        # Can pick from discard if available
        if self.discard_top:
            actions.append(('draw_discard', None))
            
        # Can discard any card from hand
        for i, card in enumerate(self.hand):
            actions.append(('discard', i))
            
        return actions
    
    def get_state_representation(self):
        """Convert current state to a numerical representation for the RL model"""
        # Encode hand as a 4×13 grid (4 suits × 13 values)
        hand_matrix = np.zeros((4, 13))
        
        for card in self.hand:
            suit_idx = CARD_SUITS.index(card.suit)
            value_idx = CARD_VALUES.index(card.value)
            hand_matrix[suit_idx][value_idx] = 1
            
        # Encode discard top card
        discard_matrix = np.zeros((4, 13))
        if self.discard_top:
            suit_idx = CARD_SUITS.index(self.discard_top.suit)
            value_idx = CARD_VALUES.index(self.discard_top.value)
            discard_matrix[suit_idx][value_idx] = 1
            
        # Encode draw pile availability
        draw_pile = np.array([1.0 if self.draw_pile_available else 0.0])
        
        # Flatten and concatenate all features
        state_vector = np.concatenate([
            hand_matrix.flatten(),
            discard_matrix.flatten(),
            draw_pile
        ])
        
        return state_vector
    
    def analyze_melds(self):
        """Analyze potential melds in hand (sets and runs)"""
        # Sort hand by suit and rank
        hand_by_suit = {}
        for card in self.hand:
            if card.suit not in hand_by_suit:
                hand_by_suit[card.suit] = []
            hand_by_suit[card.suit].append(card)
            
        for suit in hand_by_suit:
            hand_by_suit[suit].sort(key=lambda card: card.get_rank())
            
        # Group cards by rank for sets
        hand_by_rank = {}
        for card in self.hand:
            rank = card.get_rank()
            if rank not in hand_by_rank:
                hand_by_rank[rank] = []
            hand_by_rank[rank].append(card)
            
        # Find potential sets (3+ cards of same rank)
        potential_sets = []
        for rank, cards in hand_by_rank.items():
            if len(cards) >= 3:
                potential_sets.append(cards)
                
        # Find potential runs (3+ consecutive cards of same suit)
        potential_runs = []
        for suit, cards in hand_by_suit.items():
            if len(cards) < 3:
                continue
                
            # Check for consecutive cards
            for i in range(len(cards) - 2):
                run = [cards[i]]
                for j in range(i + 1, len(cards)):
                    if cards[j].get_rank() == run[-1].get_rank() + 1:
                        run.append(cards[j])
                    elif cards[j].get_rank() > run[-1].get_rank() + 1:
                        break
                
                if len(run) >= 3:
                    potential_runs.append(run)
                    
        return potential_sets, potential_runs
    
    def calculate_state_score(self):
        """Calculate a heuristic score of the current hand's quality"""
        potential_sets, potential_runs = self.analyze_melds()
        
        # Count cards that are part of potential melds
        meld_cards = set()
        for meld in potential_sets + potential_runs:
            for card in meld:
                meld_cards.add((card.suit, card.value))
                
        # Basic score is the ratio of cards that can form melds
        meld_ratio = len(meld_cards) / max(1, len(self.hand))
        
        # Bonus for having nearly complete melds
        meld_bonus = 0
        for meld in potential_sets + potential_runs:
            if len(meld) >= 3:
                meld_bonus += 0.2 * (len(meld) - 2)
                
        # Higher score for more organized hand
        return meld_ratio + meld_bonus

# ======================================================
# REINFORCEMENT LEARNING MODEL
# ======================================================

# Define the neural network for Deep Q-learning
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

class RummyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize Q-networks (policy and target)
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        
        # Initialize replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Initialize step counter and exploration parameters
        self.steps_done = 0
        
        # Action mapping for output neurons
        self.action_map = [
            ('draw_stock', None),
            ('draw_discard', None)
        ]
        # Add discard actions for each potential card position
        for i in range(NUM_CARDS_IN_HAND):
            self.action_map.append(('discard', i))
            
    def select_action(self, state, valid_actions):
        """Select an action using epsilon-greedy policy"""
        # Convert valid actions to indices
        valid_indices = []
        for action in valid_actions:
            if action[0] == 'draw_stock':
                valid_indices.append(0)
            elif action[0] == 'draw_discard':
                valid_indices.append(1)
            elif action[0] == 'discard':
                valid_indices.append(2 + action[1])  # +2 for the two draw actions
                
        # Calculate epsilon threshold
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            # Exploit: select best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                
                # Filter for valid actions only
                valid_q_values = torch.tensor([q_values[0][i].item() for i in valid_indices])
                best_valid_index = valid_indices[valid_q_values.argmax().item()]
                
                return self.action_map[best_valid_index]
        else:
            # Explore: select random valid action
            return random.choice(valid_actions)
            
    def optimize_model(self):
        """Train the model with a batch from replay memory"""
        if len(self.memory) < BATCH_SIZE:
            return
            
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Compute a mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                     dtype=torch.bool)
        non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None],
                                           dtype=torch.float32)
        
        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.long)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def update_target_network(self):
        """Update the target network with the policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self, path):
        """Save the policy network"""
        torch.save(self.policy_net.state_dict(), path)
        
    def load_model(self, path):
        """Load model weights"""
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}, starting fresh")

# ======================================================
# DEVICE INTERACTION
# ======================================================

class DeviceController:
    def __init__(self, device_id=DEVICE_ID):
        self.device_id = device_id
        
        # Define tap locations for different actions (need to be calibrated)
        self.draw_pile_location = (550, 850)  # x, y coordinates for tapping draw pile
        self.discard_pile_location = (750, 850)  # for tapping discard pile
        self.hand_card_locations = []  # Will store locations of cards in hand
        
        # Generate evenly spaced hand card locations (will need calibration)
        card_width = 70  # approximate card width in pixels
        hand_start_x = HAND_REGION[0] + 35  # middle of first card
        for i in range(NUM_CARDS_IN_HAND):
            self.hand_card_locations.append((hand_start_x + i * card_width, HAND_REGION[1] + 100))
            
    def execute_action(self, action):
        """Execute the given action on the device"""
        action_type, action_param = action
        
        if action_type == 'draw_stock':
            self._tap(self.draw_pile_location)
            
        elif action_type == 'draw_discard':
            self._tap(self.discard_pile_location)
            
        elif action_type == 'discard' and action_param is not None:
            # Tap the specified card in hand
            card_idx = min(action_param, len(self.hand_card_locations) - 1)
            self._tap(self.hand_card_locations[card_idx])
            
        # Wait for animation
        time.sleep(1)
            
    def _tap(self, location):
        """Tap at the specified location"""
        x, y = location
        try:
            subprocess.run(f'adb -s {self.device_id} shell input tap {x} {y}', shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute tap: {e}")

# ======================================================
# MAIN GAME LOOP WITH REINFORCEMENT LEARNING
# ======================================================

class RummyRLPlayer:
    def __init__(self):
        # State size: 4×13 (hand) + 4×13 (discard top) + 1 (draw pile available)
        state_size = 4 * 13 * 2 + 1
        
        # Action size: 2 (draw actions) + 13 (discard actions)
        action_size = 2 + NUM_CARDS_IN_HAND
        
        # Initialize components
        self.vision = RummyVision()
        self.game_state = RummyGameState()
        self.agent = RummyAgent(state_size, action_size)
        self.controller = DeviceController()
        
        # Training metrics
        self.episode_rewards = []
        self.episode_durations = []
        self.current_episode = 0
        self.total_steps = 0
        
        # Try to load a saved model
        if os.path.exists(MODEL_SAVE_PATH):
            self.agent.load_model(MODEL_SAVE_PATH)
            
    def map_action_to_index(self, action):
        """Map an action to its index in the action space"""
        action_type, action_param = action
        
        if action_type == 'draw_stock':
            return 0
        elif action_type == 'draw_discard':
            return 1
        elif action_type == 'discard':
            return 2 + action_param
            
        return 0  # Default
        
    def run_episode(self, training=True):
        """Run one episode of the game"""
        episode_reward = 0
        episode_steps = 0
        
        # Initial observation
        screen = self.vision.capture_screen()
        hand_cards, discard_top, draw_pile_available = self.vision.recognize_cards(screen)
        self.game_state.update_state(hand_cards, discard_top, draw_pile_available)
        
        # Get initial state score for reward calculation
        initial_score = self.game_state.calculate_state_score()
        
        # Main game loop
        while True:
            # Get valid actions
            valid_actions = self.game_state.get_valid_actions()
            
            # Convert state to vector representation
            state_vector = self.game_state.get_state_representation()
            
            # Select action
            action = self.agent.select_action(state_vector, valid_actions)
            
            # Execute action
            self.controller.execute_action(action)
            
            # Wait for game to update
            time.sleep(1.5)
            
            # Get new observation
            screen = self.vision.capture_screen()
            hand_cards, discard_top, draw_pile_available = self.vision.recognize_cards(screen)
            self.game_state.update_state(hand_cards, discard_top, draw_pile_available)
            
            # Calculate reward
            new_score = self.game_state.calculate_state_score()
            reward = new_score - initial_score
            initial_score = new_score
            
            # Check for game end (placeholder - need real end game detection)
            is_terminal = False  # This would be determined by game state
            terminal_reward = 0
            
            if is_terminal:
                # Terminal state reached - calculate final reward
                # Winning would give high positive reward, losing negative
                terminal_reward = 10.0 if True else -5.0  # Placeholder for win/lose check
                reward += terminal_reward
                next_state_vector = None
            else:
                next_state_vector = self.game_state.get_state_representation()
            
            # Store the transition in memory
            if training:
                action_idx = self.map_action_to_index(action)
                self.agent.memory.push(
                    state_vector,
                    action_idx,
                    next_state_vector,
                    reward
                )
                
                # Train the model
                self.agent.optimize_model()
            
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Update the target network
            if training and self.total_steps % TARGET_UPDATE == 0:
                self.agent.update_target_network()
                
            # Check if episode is done
            if is_terminal or episode_steps >= 100:  # Limit episode length
                break
                
        # Save training metrics
        self.episode_rewards.append(episode_reward)
        self.episode_durations.append(episode_steps)
        self.current_episode += 1
        
        # Periodically save the model
        if training and self.current_episode % 10 == 0:
            self.agent.save_model(MODEL_SAVE_PATH)
            self.plot_training_progress()
            
        return episode_reward, episode_steps
    
    def train(self, num_episodes=1000):
        """Train the agent for the given number of episodes"""
        print("Starting training...")
        
        for episode in range(num_episodes):
            # Run one episode
            reward, steps = self.run_episode(training=True)
            
            print(f"Episode {episode+1}/{num_episodes} - Reward: {reward:.2f}, Steps: {steps}")
            
            # Save model every 50 episodes
            if (episode + 1) % 50 == 0:
                self.agent.save_model(MODEL_SAVE_PATH)
                self.plot_training_progress()
                
        print("Training complete!")
        self.agent.save_model(MODEL_SAVE_PATH)
        self.plot_training_progress()
    
    def play(self, num_games=10):
        """Play games using the trained model"""
        print("Playing with trained model...")
        
        total_reward = 0
        for game in range(num_games):
            reward, steps = self.run_episode(training=False)
            total_reward += reward
            print(f"Game {game+1}/{num_games} - Reward: {reward:.2f}, Steps: {steps}")
            
        avg_reward = total_reward / num_games
        print(f"Average reward over {num_games} games: {avg_reward:.2f}")
    
    def plot_training_progress(self):
        """Plot training progress metrics"""
        plt.figure(figsize=(12, 5))
        
        # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot durations
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_durations)
        plt.title('Episode Durations')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

# ======================================================
# MAIN FUNCTION
# ======================================================

def main():
    # Initialize the RL player
    player = RummyRLPlayer()
    
    # Training mode
    print("Starting in training mode. Press Ctrl+C to stop.")
    try:
        player.train(num_episodes=100)  # Adjust number of episodes as needed
    except KeyboardInterrupt:
        print("Training interrupted.")
    
    # Play mode
    print("\nPlaying games with trained model...")
    player.play(num_games=5)
    
    print("Program complete.")

if __name__ == "__main__":
    main()
from pynput.keyboard import Controller
import time
import cv2
import pyautogui
import numpy as np
import gym.spaces as spaces
import keyboard

'''This code acts as a digital orchestrator for a classic fighting game. Think of it as a virtual assistant that assists in controlling a character in a pixelated world. It initiates the game, observes the players' health, and makes decisions on how the character should move.

Firstly, it presses the 'h' key to start the game. Then, it captures the screen, somewhat like taking a picture, and applies some adjustments to better understand what's happening. It pays special attention to the characters' health bars, as if it were monitoring their well-being.

When it's time to take action, the code translates decisions from a virtual "brain" into keyboard keys. It's like giving commands for the in-game character to dance. Occasionally, it surprises the character with unexpected moves to keep things interesting.

The story continues until the code realizes that one of the characters is about to be defeated, concluding that particular "performance" of the game. In essence, this code functions as an assistant that helps control a character in a fighting game, making the experience more enjoyable and interactive.
'''
class CustomEnvironment:
    def __init__(self):
        self.done = False
        self.keyboard = Controller()
        self.left = None
        self.top = None
        self.width = None
        self.height = None
        # players health
        self.vidaplayer1 = [0]
        self.vidaplayer2 = [0]
        #set num of action
        self.num_actions = 9
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def reset(self):
        # Mapeamento das ações da IA para teclas
        reset_mapping = {
            0: 'h'  # set key for load game in this case is h.
        }

        # press the key "h" for load game
        key_to_press = reset_mapping.get(0)
        self.keyboard.press(key_to_press)
        time.sleep(0.1)
        self.keyboard.release(key_to_press)
        
        observation = self.observation_space

        return observation
        
    def observation_space(self, left, top, width, height):
        # observation space
        observation = self.capture_and_process(left, top, width, height)
        shape = observation.shape  
        observation_space = spaces.Box(0, 255, shape, dtype=np.uint8)
        return observation_space 

    # catpura e processa imagem               
    def capture_and_process(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        screenshot1 = pyautogui.screenshot(region=(left, top, width, height))
        screenshot = cv2.resize(np.array(screenshot1), (width // 2, height // 2))
        observation = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        observation = cv2.GaussianBlur(observation, (5, 5), 0)
                
        # Take screenshot of the health bar
        hsv_atual = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
 
        # Set the green color range in HSV
        verde_baixo = np.array([90, 255, 255])  # cores para barra street 2 x
        verde_alto = np.array([90, 255, 255])  #
        
          
        # Create a Mask for the Color Green
        mascara = cv2.inRange(hsv_atual, verde_baixo, verde_alto)

        # Find contours in the mask
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the outlines to ignore the small ones
        contornos_maiores = [cnt for cnt in contornos if cv2.contourArea(cnt) > 3]

        
        # Show Outline Width            
        for idx, contorno in enumerate(contornos_maiores):
            x, y, w, h = cv2.boundingRect(contorno)
            cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if idx == 0:
                self.vidaplayer2[0] = w
            elif idx == 1:
                self.vidaplayer1[0] = w       
                
        info = {
            'player1_health': self.vidaplayer1[0],
            'player2_health': self.vidaplayer2[0],
        }      
        # for show image
        
        cv2.imshow('Game Screen', screenshot)
        cv2.waitKey(1)
        return observation, info


    def action_to_array(self, action_index):
        # Mapping AI actions to keys
        action_mapping = {
            0: ['w'],    # up
            1: ['s'],  # down
            2: ['a'],  # left
            3: ['d'],  # right
            4: ['z'],  # Numpad 1
            5: ['x'],  # Numpad 2
            6: ['c'],  # Numpad 3
            7: ['v'],  # Numpad 4
            8: ['c', 'v'], # Numpad 5
            9: ['x', 'z'], # Numpad 6        
        }
        
        if action_index in action_mapping:
            return action_mapping[action_index]
        else:
            print("Índice de ação inválido:", action_index)
            return [0] * 9 

    def step(self, nnOutput, exploration_rate=0.4):
        # Chooses the action based on the neural network's probabilities
        action = np.argmax(nnOutput)

        # exploration_rate
        if np.random.rand() < exploration_rate:
            action = np.random.randint(0, self.num_actions)  # choice random action

        # execute action 
        keys_to_press = self.action_to_array(action)
        try:
            for key in keys_to_press:
                # press key
                keyboard.press(key)
                time.sleep(0.1)
        finally:
            for key in keys_to_press:
                keyboard.release(key)
        
        # env update
        observation, info = self._update_observation()
        # episode is done
        done = self.is_episode_done()
        
        return observation, done, info

    
    #observation update
    def _update_observation(self):
        observation, info = self.capture_and_process(self.left, self.top, self.width, self.height)
        return observation, info

    
    def is_episode_done(self):
        if self.vidaplayer2[0] < 8:
            return True
        else:
            return False
   

    #function do execute action
    def execute_actions(self, actions):
        keyboard = Controller()  # Initialize keyboard control
        for action in actions:
            for key in action:
                keyboard.press(key)  # key press
            time.sleep(0.1)  
            for key in action:
                keyboard.release(key)  # release key

        # wait for release key
        time.sleep(0.1)

        for key in keys_to_press:
            self.keyboard.release(key)
            

   

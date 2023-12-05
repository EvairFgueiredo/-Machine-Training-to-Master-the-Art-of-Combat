import retro
import numpy as np
import cv2
import neat
import pygetwindow as gw
import pickle
import threading
import ambiente_visao
from concurrent.futures import ThreadPoolExecutor
import time

# function for load model "winner.pkl" 
'''def play_game_with_winner(model_file):
    with open('winner.pkl', 'rb') as input_file:
        winner_genome = pickle.load(input_file)'''


# load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

env = ambiente_visao.CustomEnvironment()

nome_da_janela = "Fightcade FBNeo v0.2.97.44-54 • Super Street Fighter II X - grand master challenge (super street fighter 2 X 940223 Japan)"


janela = gw.getWindowsWithTitle(nome_da_janela)

if janela:
    # window is check
    janela = janela[0]  # Use the first window you find, if there are multiple with the same name
    left, top, width, height = janela.left, janela.top, janela.width, janela.height
    # Initialize a thread for capture and process the window
else:
    print("Window not found") 
    
# generation
generation = -1


# funtion to evaluate genomes during training process
def eval_genomes(genomes, config,):

    # generation
    global generation
    generation += 1
    for genome_id, genome in genomes:
        
        #######################################################################################################
        ###########################################control variables###########################################
        #######################################################################################################

        log = True
        log_size = 300

        #######################################################################################################
        ###########################################control variables###########################################
        #######################################################################################################

        # get environment print screen
        env.reset()
        observation, info = env.capture_and_process(left, top, width, height)
        
        inx, iny = observation.shape
        inx = int(inx / 8)
        iny = int(iny / 8)

        # create the neural network
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        

        # initialize variables
        done = False
        fitness_current = 0
        fitness_current_max = 0
        counter = 0
        frame = 0
        
        # main loop
        reward = 0
        
        while not done:
            # frame count
            frame += 1
            
            observation, info = env.capture_and_process(left, top, width, height)
            
            # prepare the print screen to use as neural network input
            observation = cv2.resize(observation, (inx, iny))
        
            input_data = np.ndarray.flatten(observation)

            # Execute a ação da rede neural
            nnOutput = net.activate(input_data)
            
            #lifes before action
            health_before_action = info['player1_health']
            health_before_action2 = info['player2_health']
            
            
            #action
            observation, done, info = env.step(nnOutput)
            
            #lifes after action
            health_after_action = info['player1_health']
            health_after_action2 = info['player2_health']

            fitness_current += reward   
            reward = 0
            
            # set counter to stop
            if fitness_current > fitness_current_max:
                fitness_current_max = fitness_current
                counter = 0
            else:
                counter += 1
            
            reason_stopped = ''             
            if health_after_action <= 5:
                reason_stopped = 'Lose'
                reward -= 10
                done = True
                
            if health_after_action2 <= 5:
                reason_stopped = 'Win'
                reward += 10
                done = True
                
            # reward system
            diference = health_before_action - health_after_action
            
            # counter to stop
            if counter >= 110:
                reason_stopped = 'Maximum frames without reward'
                done = True
            '''if health_before_action == health_after_action:
                reward += 0.1
                print("defendeu e ganhou >>", f"{reward}")'''
    
            diference2 = health_before_action2 - health_after_action2
            if health_before_action2 > health_after_action2:
                # Player 2's health decreased after player 1's action
                if diference2 <= 80.0:
                    reward += diference2
            
            if health_before_action > health_after_action:
                # Player 1's health decreased after player 2's action
                if diference <= 80.0:
                    reward -= diference
       
            fitness_current += reward
            
            genome.fitness = fitness_current
            
            
            # show game monitor
            #cv2.imshow('Game Screen', observation)
            #cv2.waitKey(1)

            # logs
            if log and frame % log_size == 0:
                print('generation: ', generation, 'genome_id: ', genome_id, 'frame: ', frame, 'counter', counter, 'fitness_current: ', fitness_current)
            if done and log:
                print('------------------------------------------------------------------------------------------------------------------')
                print('generation: ', generation, 'genome_id: ', genome_id, 'frame: ', frame, 'counter', counter, 'fitness_current: ', fitness_current, 'Reason: ', reason_stopped)
                print('------------------------------------------------------------------------------------------------------------------')


# set configuration
p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-15')

# report trainning
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10)) #every x generations save a checkpoint

# run trainning
winner = p.run(eval_genomes)

# save the winner
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

# close environment
env.close()
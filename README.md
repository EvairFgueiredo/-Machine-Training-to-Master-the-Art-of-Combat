# -Machine-Training-to-Master-the-Art-of-Combat

In this thrilling codebase, we venture into a realm of machine learning where a virtual agent is trained to master the complex nuances of an epic fighting game. Harnessing the powerful NEAT library, our digital hero undergoes generations of evolution, refining its skills through a unique blend of genetic algorithms and neural networks.

At the heart of the training is an intricate neural network, dynamically created through genetic evolution. Each genome, representing a unique neural structure, is put to the test in a custom vision and action environment, providing the virtual agent with the opportunity to learn tactics, strategies, and special moves to overcome virtual opponents.

Magic unfolds as the agent interprets the game screen, makes rapid decisions, and receives rewards or penalties as feedback. The NEAT algorithm, with its ability to evolve neural topologies, sculpts the digital mind of the agent, transforming it from a novice to a master in the virtual combat arts.

The journey is accompanied by informative logs detailing the agent's progress across generations, revealing moments of triumph, challenges overcome, and emerging strategies. With each generation, the code persists, saving the most promising genomes until, finally, a winner emerges - a model that encapsulates perfection in virtual fighting skills.

Upon concluding its mission, the code draws the curtains on the game, leaving behind a trained digital hero ready to face even greater challenges.


# Usage Requirements:
To adapt this code to play any fighting game on Fightcade or other platforms, considering only the life bar colors of the specific game, please follow the guidelines below:

# Game Configuration:
Open the target fighting game on Fightcade or the desired platform.
Identify the specific colors of the life bars for both players.

# Monitor Configuration:
The code assumes a monitor size of 19 inches.

# Window Setup:
Adjust the code to match the title of the game window. Modify the variable nome_da_janela accordingly.
Ensure the game window is in a 2x size configuration.

# Color Recognition:
Analyze the life bars' colors in the game and modify the code to recognize these colors. Adapt the color recognition part in the eval_genomes function.

# Additional Libraries:
Ensure the required Python libraries are installed. You can install them using the following command:
bash
Copy code
pip install retro numpy opencv-python neat-python pygetwindow pillow pythonnet concurrent-futures
# Execution:
Run the code in an environment that supports the specified game setup and monitor configuration.

# Patience:
The training process may take some time as the algorithm evolves and refines its strategies over multiple generations.
By meeting these requirements, you'll embark on an exciting journey of machine learning as the virtual agent learns to master the art of combat in the specified game environment. May your digital warrior emerge victorious!

# Requirements:
plaintext
Copy code
retro
numpy
opencv-python
neat-python
pygetwindow
pillow
pythonnet
concurrent-futures
Be prepared to witness the birth of a virtual master, a digital warrior traversing the boundaries between code and game, bringing with it the promise of a future where machines learn the art of battle.

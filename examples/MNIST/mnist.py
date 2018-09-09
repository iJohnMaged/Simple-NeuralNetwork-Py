import sys, os
from neural_network import NeuralNetwork
from examples.MNIST.load_data import *
from activation_functions import A_FUNCTIONS
import pygame
sys.path.append(f'..{os.sep}..')


FPS = 60

def main():

    # Train the neural network on the MNIST dataset

    train_index = 0

    # Neural Network info:
    # Using 784 inputs nodes which are every pixel of the 28*28 image.
    # Using 300 hidden nodes
    # Using 10 output nodes, where output[i] is the probability of digit = i
    nn = NeuralNetwork(784, 300, 10, lr=0.1, activation_func=A_FUNCTIONS['sigmoid'])

    while train_index < len(X_train):
        nn.train(X_train[train_index], y_train[train_index])
        train_index += 1
        print("training ", train_index)

    # You don't have to train the NN every run. You can save the NN model after training using this commented code:
    # nn.save_to_file('nn.nno')

    # And then you can remove the training loop and load the data:
    # nn = NeuralNetwork.load_from_file('nn.nno')

    # Initialize PyGame
    # I'm currently using PyGame to show the current image being tested.
    # Later I'll add a canvas to draw digits for the model to predict!
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Comic Sans MS', 30)

    screen = pygame.display.set_mode((400, 500))
    base = pygame.Surface((400, 100))
    base.fill((0, 0, 0))

    correct = 0
    test_index = 0
    accuracy = 0

    while True:

        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        accuracy_color = (255, 0, 0)

        # Testing the neural network with out testing data!

        if test_index < len(X_test):
            # Returns the output array and the digit with highest probability
            output = nn.predict(X_test[test_index], True)
            # Check if the digit with highest probability is the same as the ground truth.
            if output[1] == y_test[test_index]:
                correct += 1
                accuracy_color = (0, 255, 0)

            test_index += 1
            # Calculate accuracy
            accuracy = 100 * (correct / (test_index))

            if test_index >= len(X_test):
                print('Finished the test set')
                print(f'Final accuracy: {accuracy}')

        # Drawing the image and text, all PyGame stuff.
        img = X_test_images[test_index-1]
        img_surf = pygame.surfarray.make_surface(img)
        img_surf = pygame.transform.scale(img_surf, (400, 400))

        textsurface = font.render(f'Accuracy: {accuracy:.2f}', False, accuracy_color)

        screen.blit(img_surf, (0, 0))
        screen.blit(base, (0, 400))
        screen.blit(textsurface, ((200 - textsurface.get_width() // 2), (450 - textsurface.get_height() // 2)))
        pygame.display.flip()

if __name__ == '__main__':
    main()
import numpy as np
# import tensorflow as tf
# import keras
import random
import time
import pygame
import sys

LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3
gridSize = (8, 8)

class Game():
  def __init__(self):
    self.direction = LEFT
    self.length = 1
    self.segments = [
      (gridSize[0]//2, gridSize[1]//2)
      ]

    self.apple = (gridSize[0]//2-1, gridSize[1]//2-1)
    # self.apple = (0, 0)
    self.gameOver = False
    self.score = 0
  
  def update(self, action):
    #take an action (direction) and move the snake
    #additionally, check if game is ended by collision
    self.changeDirection(action)

    oldHead = self.segments[0]
    newHead0, newHead1 = oldHead[0], oldHead[1]
    if self.direction == LEFT:
      newHead1 = oldHead[1] - 1
    elif self.direction == RIGHT:
      newHead1 = oldHead[1] + 1
    elif self.direction == UP:
      newHead0 = oldHead[0] - 1
    elif self.direction == DOWN:
      newHead0 = oldHead[0] + 1
    newHead = (newHead0, newHead1)

    #check if self collision or head off screen
    if newHead in self.segments[:-1]:
      self.gameOver = True
    if newHead0 == -1 or newHead0 == gridSize[0]:
      self.gameOver = True
    if newHead1 == -1 or newHead1 == gridSize[1]:
      self.gameOver = True

    #check if apple eat
    #length extension on apple eat
    if newHead == self.apple:
      self.score += 1
      self.resetApple() #have to insert new head before resetting apple otherwise it bugs out if apple's spawned on new head location
    else:
      self.segments.pop()

    if not self.gameOver:
      self.segments.insert(0, newHead)

  def evaluate(self):
    if len(self.segments) == 0:
      return -100
    return 100*len(self.segments)-self.dist(self.segments[0], self.apple)
  
  def dist(self, a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

  def changeDirection(self, newDir):
    # self.direction = (self.direction + newDir) % 4
    self.direction = newDir

  def resetApple(self):
    while True:
      self.apple = (random.randrange(0, gridSize[0]), random.randrange(0, gridSize[1]))
      if self.apple not in self.segments:
        break

  def getScore(self):
    return self.score

class SnakeEnv():
  def __init__(self):
    self.game = Game()

  def reset(self):
    self.game = Game()
    return self.observe(self.game)

  def step(self, action):
    scoreBefore = self.game.evaluate()
    self.game.update(action)
    scoreAfter = self.game.evaluate()
    reward = scoreAfter - scoreBefore
    obs = self.observe(self.game)
    done = self.game.gameOver
    return obs, reward, done

  def observe(self, game):
    """
    return board state either as board or in form snake state, apple pos
    """
    #board
    # observation = np.zeros((gridSize[0], gridSize[1]))
    # for segment in game.segments:
    #   observation[segment[0], segment[1]] = 1
    # observation[game.apple[0], game.apple[1]] = -1
    # return observation

    #actor states
    observation = [
      game.segments,
      game.direction,
      game.apple,
    ]
    return observation

def dumb_move_from_obs(obs):
  """
  used to test SnakeEnv
  simplistic move returned from relative positions of head, apple, direction
  """
  def get_dist(A, B):
    return abs(A[0]-B[0]) + abs(A[1]-B[1])

  def evaluate(obs):
    head = obs[0][0]
    if (
      head in obs[0][1:-1] or
      head[0] < 0 or 
      head[0] >= gridSize[0] or 
      head[1] < 0 or
      head[1] >= gridSize[1]
    ):
      return -(gridSize[0]+gridSize[1])
    else:
      return -get_dist(head, obs[2])

  def move_in_dir(head, dir):
    if dir == LEFT:
      step = (0, -1)
    elif dir == RIGHT:
      step = (0, +1)
    elif dir == UP:
      step = (-1, 0)
    elif dir == DOWN:
      step = (+1, 0)
    return (head[0] + step[0], head[1] + step[1])

  segments = obs[0].copy()
  direction = obs[1]
  apple = obs[2]

  scores = np.array([0, 0, 0])
  for i in range(3):
    move = i-1
    new_dir = (obs[1] + move) % 4
    new_head_location = move_in_dir(obs[0][0], new_dir)
    new_segments = segments.copy()
    new_segments.insert(0, new_head_location)
    new_obs = [new_segments, new_dir, apple]
    scores[i] = evaluate(new_obs)
  a = np.argmax(scores)
  # a = np.random.choice(np.flatnonzero(scores == scores.max()))
  return (obs[1] + a - 1) % 4  

def frame_from_obs(obs):
  segments = obs[0]
  apple = obs[2]
  observation = np.zeros((gridSize[0], gridSize[1]))
  for segment in segments:
    observation[segment[0], segment[1]] = 1
  observation[apple[0], apple[1]] = -1
  return observation

class gameDisplayer():
  """
  takes frames and displays
  """
  def __init__(self, frames):
    self.blockSize = 30
    self.fps = 15
    self.clock = pygame.time.Clock()

    pygame.init()
    self.surface = pygame.display.set_mode([gridSize[0]*self.blockSize, gridSize[1]*self.blockSize])
    pygame.display.set_caption("")

    self.frames = frames
    self.currentFrame = 0
  
  def run(self):
    while True:
      self.renderFrame(self.currentFrame)
      if self.currentFrame < len(self.frames)-1:
        self.currentFrame += 1
      self._checkEvents()
      self.clock.tick(self.fps)

  def renderFrame(self, i):
    self.surface.fill((230, 230, 230))
    frame = self.frames[i]
    for x in range(gridSize[0]):
      for y in range(gridSize[1]):
        currentSquare = pygame.Rect(self.blockSize*x, self.blockSize*y, self.blockSize, self.blockSize)
        if frame[x][y] == -1:
          pygame.draw.rect(self.surface, "red", currentSquare)
        elif frame[x][y] == 1:
          pygame.draw.rect(self.surface, "green", currentSquare)
    pygame.display.flip()

  def displayFrames(frames):
    pass

  def _checkEvents(self):
    events = pygame.event.get()
    for event in events:
      if event.type == pygame.QUIT:
        sys.exit() 

def main():
  frames = []
  env = SnakeEnv()
  obs = env.reset()
  frames = []
  while True:
    obs, reward, done = env.step(dumb_move_from_obs(obs))
    frames.append(frame_from_obs(obs))
    if done:
      break
  g = gameDisplayer(frames)
  g.run()

  # n_inputs = gridSize[0]*gridSize[1]
  # model = keras.models.Sequential([
  #   keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
  #   keras.layers.Dense(3, activation="softmax"),
  # ])

  # def play_one_step(env, obs, model, loss_fn):
  #   with tf.GradientTape() as tape:
  #       dir = model(obs.flatten()[np.newaxis])
  #       roll = tf.random.uniform([1,1])
  #       if roll < dir[0][0]:
  #         action = -1
  #       elif roll < dir[0][0] + dir[0][1]:
  #         action = 0
  #       else:
  #         action = 1
  #       y_target = np.array([0, 0, 0])
  #       y_target[np.argmax(dir)] = 1
  #       y_target = y_target[np.newaxis]
  #       loss = tf.reduce_mean(loss_fn(y_target, dir))
  #   grads = tape.gradient(loss, model.trainable_variables)
  #   obs, reward, done = env.step(action)
  #   return obs, reward, done, grads

  # def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
  #     all_rewards = []
  #     all_grads = []
  #     for episode in range(n_episodes):
  #         current_rewards = []
  #         current_grads = []
  #         obs = env.reset()
  #         for step in range(n_max_steps):
  #             obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
  #             current_rewards.append(reward)
  #             current_grads.append(grads)
  #             if done:
  #                 break
  #         all_rewards.append(current_rewards)
  #         all_grads.append(current_grads)
  #     return all_rewards, all_grads

  # def discount_rewards(rewards, discount_factor):
  #     discounted = np.array(rewards)
  #     for step in range(len(rewards) - 2, -1, -1):
  #         discounted[step] += discounted[step + 1] * discount_factor
  #     return discounted

  # def discount_and_normalize_rewards(all_rewards, discount_factor):
  #     all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
  #     flat_rewards = np.concatenate(all_discounted_rewards)
  #     reward_mean = flat_rewards.mean()
  #     reward_std = flat_rewards.std()
  #     if reward_std == 0:
  #       print(flat_rewards)
  #       print("div by 0")
  #     return all_discounted_rewards
  #     # return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

  # n_iterations = 50
  # n_episodes_per_update = 50
  # n_max_steps = 100
  # discount_factor = 0.95

  # optimizer = keras.optimizers.Adam(lr=0.01)
  # # loss_fn = keras.losses.binary_crossentropy
  # loss_fn = keras.losses.categorical_crossentropy

  # for iteration in range(n_iterations):
  #     print(f"working on iteration {iteration}...")
  #     all_rewards, all_grads = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn)
  #     all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

  #     all_mean_grads = []
  #     for var_index in range(len(model.trainable_variables)):
  #         mean_grads = tf.reduce_mean([
  #             final_reward*all_grads[episode_index][step][var_index]
  #             for episode_index, final_rewards in enumerate(all_final_rewards)
  #             for step, final_reward in enumerate(final_rewards)
  #         ], axis=0)
  #         all_mean_grads.append(mean_grads)
  #     optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

  # def render_policy_net(model, n_max_steps=200, seed=42):
  #   frames = []
  #   env = SnakeEnv()
  #   obs = env.reset()
  #   for step in range(n_max_steps):
  #       frames.append(obs)
  #       dir = model.predict(obs.flatten()[np.newaxis])
  #       roll = tf.random.uniform([1,1])
  #       if roll < dir[0][0]:
  #         action = -1
  #       elif roll < dir[0][0] + dir[0][1]:
  #         action = 0
  #       else:
  #         action = 1
  #       obs, reward, done = env.step(action)
  #       if done:
  #           break
  #   return frames
  
  # frames = render_policy_net(model)
  # g = gameDisplayer(frames)
  # g.run()

  # # while True:
  # #   obs, reward, done = env.step(random.randrange(-1,2))
  # #   frames.append(obs)
  # #   if done:
  # #     break
  # # g = gameDisplayer(frames)
  # # g.run()



  # # while True:
  # #   head = env.game.segments[0]
  # #   apple = env.game.apple
  # #   if head[0] > apple[0]:
  # #     dir = LEFT
  # #   else:
  # #     dir = RIGHT
  # #   if head[1] > apple[1]:
  # #     dir = UP
  # #   elif head[1] < apple[1]:
  # #     dir = DOWN
  # #   obs, reward = env.step(dir)
  # #   print(obs, reward)
  # #   time.sleep(.1)


if __name__ == "__main__":
  main()
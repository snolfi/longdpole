### Long Double Pole Balancing Problem

This is an environment for [OpenAI Gym](https://github.com/openai/gym) where the goal is to train a controller for a cart with two poles attached on the top with passive joints
see: Pagliuca P., Milano N. and Nolfi S. (2018). Maximizing adaptive power in neuroevolution. PLoS ONE 13(7): e0198788, available from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198788
It is an harder version of the classic pole balancing problem described in:
Wieland A. (1991). Evolving controls for unstable systems. In Proceedings of the International Joint Conference on Neural Networks, Volume II, Seattle, WA, USA: IEEE Press. pp. 667–673.

The installation requires python3, cython, a command line C compiler, and the GNU GSL Scientific Library  (https://www.gnu.org/software/gsl)

Thanks to the usage of a C++ library, the environment run much faster than pole-balancing problem included in OpenAI Gym.
Moreover, it constitutes a much harder version of the standard double-pole balancing problem commonly used in the literature. 
Consequently it can be used to benchmark modern algorithms. 

To install this environment use the following instructions:

    git clone https://github.com/snolfi/longdpole
    cd longdpole
    cd longdpolelib
    python setupErDpole.py build_ext --inplace  
    cp ErDpole*.so ../
    cd ..    
    pip install -e .
    
In case of error during the execution of the command included in the fourth line, please check and eventually update the name of directory that includes your gsl library and associated include file
specified in the file ./longdpolelib/setupevonet.py file. 
The *.so or *.dll file library should be included in the directory in which you launch your python application

The environment come in three version that vary with respect to the length and the mass of the second pole:
LongdpoleEnv-v0: the length and the mass of the second pole is 50% that of the first pole
LongdpoleEnv-v1: the length and the mass of the second pole is 60% that of the first pole
LongdpoleEnv-v2: the length and the mass of the second pole is 70% that of the first pole

A basic script testing the environment:

    import gym
    import longdpole  # this instruction imports the c++ library compiled with cython
    
    env = gym.make('LongdpoleEnv-v0')
    env.reset()
    for _ in range (1000):
        observation, reward, done, info = env.step(env.actions_space.sample())
        print(observation)
        print(reward)
        env.render()
        if (done):
            break
    env.close()


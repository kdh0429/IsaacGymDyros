from rl_games.common import object_factory
import rl_games.algos_torch
from rl_games.algos_torch import network_builder
from isaacgymenvs.learning.rl_games_custom import network_builder_dyros
from rl_games.algos_torch import models
from rl_games.algos_torch.model_builder import ModelBuilder
from isaacgymenvs.learning.rl_games_custom import models_dyros

NETWORK_REGISTRY = {}

def register_network(name, target_class):
    NETWORK_REGISTRY[name] = lambda **kwargs : target_class()

class ModelBuilderDyros:
    def __init__(self):

        self.model_factory = object_factory.ObjectFactory()
        self.model_factory.register_builder('discrete_a2c', lambda network, **kwargs : models.ModelA2C(network))
        self.model_factory.register_builder('multi_discrete_a2c', lambda network, **kwargs : models.ModelA2CMultiDiscrete(network))
        self.model_factory.register_builder('continuous_a2c', lambda network, **kwargs : models.ModelA2CContinuous(network))
        self.model_factory.register_builder('continuous_a2c_logstd', lambda network, **kwargs : models.ModelA2CContinuousLogStd(network))
        self.model_factory.register_builder('soft_actor_critic', lambda network, **kwargs : models.ModelSACContinuous(network))
        self.model_factory.register_builder('continuous_a2c_logstd_dyros', lambda network, **kwargs : models_dyros.ModelA2CContinuousLogStdDYROS(network))
        #self.model_factory.register_builder('dqn', lambda network, **kwargs : models.AtariDQN(network))

        self.network_factory = object_factory.ObjectFactory()
        self.network_factory.set_builders(NETWORK_REGISTRY)
        self.network_factory.register_builder('actor_critic', lambda **kwargs : network_builder.A2CBuilder())
        self.network_factory.register_builder('resnet_actor_critic', lambda **kwargs : network_builder.A2CResnetBuilder())
        self.network_factory.register_builder('rnd_curiosity', lambda **kwargs : network_builder.RNDCuriosityBuilder())
        self.network_factory.register_builder('soft_actor_critic', lambda **kwargs: network_builder.SACBuilder())
        self.network_factory.register_builder('actor_critic_dyros', lambda **kwargs: network_builder_dyros.A2CDYROSBuilder())

    def load(self, params):
        self.model_name = params['model']['name']
        self.network_name = params['network']['name']

        network = self.network_factory.create(self.network_name)
        network.load(params['network'])
        model = self.model_factory.create(self.model_name, network=network)

        return model